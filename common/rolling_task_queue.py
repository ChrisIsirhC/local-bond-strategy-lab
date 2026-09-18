"""Durable queue for expensive rolling-research jobs.

The dashboard deliberately submits jobs to a separate Python process.  A full
V2 rolling search can take long enough that holding a Streamlit request open
would otherwise block the page and make a second click start another costly
search.  Queue state is stored under ``backtest_outputs`` so it is visible and
recoverable alongside the research artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Iterator

from common.config import strategy_config_from_dict
from common.rolling_research import RollingResearchCancelled, RollingResearchConfig, run_rolling_research


QUEUE_RELATIVE_PATH = Path("backtest_outputs") / "rolling_task_queue.json"
MAX_QUEUE_WORKERS = 2
DATA_LOCK_SUFFIX = ".data.lock"
WORKER_LOCK_SUFFIX = ".worker.lock"
WORKER_LOG_NAME = "rolling_task_queue.log"
_LOCK_STALE_SECONDS = 120


def queue_path(root: Path) -> Path:
    return root / QUEUE_RELATIVE_PATH


def list_rolling_tasks(root: Path) -> list[dict[str, Any]]:
    """Return queue records in submission order, newest fields kept intact."""
    with _data_lock(root):
        return _read_records(queue_path(root))


def enqueue_rolling_task(root: Path, config: RollingResearchConfig) -> dict[str, Any]:
    """Persist a job without starting a duplicate worker.

    The first job is visibly launched immediately.  The detached worker still
    claims it before doing work, so queue ownership remains single-process.
    """
    created_at = datetime.now().isoformat(timespec="seconds")
    with _data_lock(root):
        path = queue_path(root)
        records = _read_records(path)
        status, stage = _submission_state(records)
        task = {
            "task_id": f"Q{datetime.now():%Y%m%d%H%M%S%f}",
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "status": status,
            "stage": stage,
            "task_name": config.task_name or config.base_config.name,
            "task_kind": "dashboard_rolling",
            "config": _serialize_config(config),
            "output_dir": None,
            "experiment_dir": None,
            "error": None,
            "completed_periods": 0,
            "total_periods": None,
            "cancel_requested": False,
        }
        records.append(task)
        _write_records(path, records)
    return task


def enqueue_factor_rolling_task(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Queue a selected factor universe without first creating an F study.

    The payload is already JSON-shaped and contains the exact factor list and
    strategy settings chosen on the factor-research page.  Persisting it here
    freezes those inputs before the background worker starts.
    """
    base = payload.get("base_config")
    if not isinstance(base, dict):
        raise ValueError("因子滚动任务缺少基线配置")
    created_at = datetime.now().isoformat(timespec="seconds")
    with _data_lock(root):
        path = queue_path(root)
        records = _read_records(path)
        status, stage = _submission_state(records)
        task = {
            "task_id": f"Q{datetime.now():%Y%m%d%H%M%S%f}",
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "status": status,
            "stage": stage,
            "task_name": str(payload.get("task_name") or base.get("name") or "因子完整滚动"),
            "task_kind": "factor_rolling",
            "config": payload,
            "output_dir": None,
            "experiment_dir": None,
            "error": None,
            "completed_periods": 0,
            "total_periods": None,
            "cancel_requested": False,
        }
        records.append(task)
        _write_records(path, records)
    return task


def retry_rolling_task(root: Path, task_id: str) -> dict[str, Any]:
    """Clone one failed task into a new queued task without altering its record.

    A retry deliberately receives a new Q identifier.  The failed execution is
    still useful audit evidence, while the new task is an independently
    recoverable attempt using the exact serialized settings of the failed one.
    """
    source_id = str(task_id)
    with _data_lock(root):
        path = queue_path(root)
        records = _read_records(path)
        source = next((item for item in records if str(item.get("task_id")) == source_id), None)
        if source is None:
            raise ValueError(f"未找到滚动任务：{source_id}")
        if source.get("status") != "失败":
            raise ValueError("只有失败任务可以再次执行")
        raw_config = source.get("config")
        # Validate before persisting so a damaged historic row cannot clog the queue.
        _deserialize_config(raw_config)
        created_at = datetime.now().isoformat(timespec="seconds")
        status, stage = _submission_state(records)
        task = {
            "task_id": f"Q{datetime.now():%Y%m%d%H%M%S%f}",
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "status": status,
            "stage": "正在启动" if status == "启动中" else f"等待前序任务完成（重试 {source_id}）",
            "task_name": source.get("task_name") or "未命名任务",
            "task_kind": source.get("task_kind", "dashboard_rolling"),
            "config": raw_config,
            "output_dir": None,
            "experiment_dir": None,
            "error": None,
            "completed_periods": 0,
            "total_periods": None,
            "cancel_requested": False,
            "retry_of": source_id,
        }
        records.append(task)
        _write_records(path, records)
    return task


def remove_failed_rolling_task(root: Path, task_id: str) -> None:
    """Remove one failed queue record without touching any archived result."""
    source_id = str(task_id)
    with _data_lock(root):
        path = queue_path(root)
        records = _read_records(path)
        source = next((item for item in records if str(item.get("task_id")) == source_id), None)
        if source is None:
            raise ValueError(f"未找到滚动任务：{source_id}")
        if source.get("status") != "失败":
            raise ValueError("只有失败任务可以删除")
        _write_records(path, [item for item in records if str(item.get("task_id")) != source_id])


def cancel_rolling_task(root: Path, task_id: str) -> None:
    """Request cooperative cancellation of a queued or active rolling task.

    A running search is stopped at its next safe period boundary, preserving the
    single worker and preventing a partially-written registry or archive.
    """
    source_id = str(task_id)
    with _data_lock(root):
        path = queue_path(root)
        records = _read_records(path)
        source = next((item for item in records if str(item.get("task_id")) == source_id), None)
        if source is None:
            raise ValueError(f"未找到滚动任务：{source_id}")
        if source.get("status") not in {"启动中", "运行中"}:
            raise ValueError("只能终止正在启动或运行中的任务")
        if source.get("status") == "启动中":
            source.update({"status": "已终止", "stage": "已终止", "finished_at": datetime.now().isoformat(timespec="seconds")})
        else:
            source.update({"cancel_requested": True, "stage": "正在终止（将在当前期结束后停止）"})
        _write_records(path, records)


def start_rolling_queue_worker(root: Path) -> bool:
    """Start up to two detached workers and report whether the queue was idle.

    Two short-lived contenders are started on every submission.  Slot locks
    ensure that at most two workers own a task; extra contenders exit after
    failing to acquire their slot, while a finishing slot can immediately
    claim the next waiting task.
    """
    was_idle = not rolling_worker_running(root)
    script = root / "run_rolling_task_queue.py"
    if not script.exists():
        raise FileNotFoundError(f"滚动任务队列启动脚本不存在：{script}")
    log_path = root / "backtest_outputs" / WORKER_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for slot in range(MAX_QUEUE_WORKERS):
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen(
                [sys.executable, "-u", str(script), "--root", str(root), "--slot", str(slot)],
                cwd=str(root),
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                close_fds=False,
            )
    return was_idle


def rolling_worker_running(root: Path) -> bool:
    running = False
    for slot in range(MAX_QUEUE_WORKERS):
        lock_path = _worker_lock_path(root, slot)
        if not lock_path.exists():
            continue
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            _remove_if_exists(lock_path)
            continue
        if _process_exists(pid):
            running = True
        else:
            _remove_if_exists(lock_path)
    return running


def process_rolling_queue(
    root: Path,
    run_job: Callable[..., dict[str, Any]] = run_rolling_research,
    *,
    worker_slot: int = 0,
) -> None:
    """Run pending jobs. Called only by one detached worker slot."""
    with _worker_lock(root, worker_slot) as acquired:
        if not acquired:
            return
        while True:
            task = _claim_next_task(root)
            if task is None:
                return
            task_id = str(task["task_id"])
            try:
                progress = lambda message, completed, total: _set_progress(
                    root, task_id, message, completed, total,
                )
                cancelled = lambda: _cancel_requested(root, task_id)
                if task.get("task_kind") == "factor_rolling":
                    from common.factor_rolling_task import run_factor_rolling_task
                    raw = task.get("config")
                    if not isinstance(raw, dict):
                        raise ValueError("因子滚动任务配置损坏")
                    result = run_factor_rolling_task(root, raw, progress=progress, cancel_requested=cancelled)
                else:
                    config = _deserialize_config(task["config"])
                    # ``run_rolling_research`` supports cooperative cancellation.
                    # Keep the queue compatible with older, injected test runners
                    # that only accept the historical ``progress`` keyword.  This
                    # is inspected before invocation rather than catching
                    # ``TypeError`` so a real failure raised *inside* a job is not
                    # mistakenly retried with different arguments.
                    run_arguments: dict[str, Any] = {"progress": progress}
                    if "cancel_requested" in inspect.signature(run_job).parameters:
                        run_arguments["cancel_requested"] = cancelled
                    result = run_job(root, config, **run_arguments)
            except RollingResearchCancelled:
                _complete_task(root, task_id, status="已终止", stage="已终止")
                continue
            except Exception as exc:  # Keep the next queued task runnable.
                _complete_task(
                    root,
                    task_id,
                    status="失败",
                    stage="运行失败",
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
                )
                continue
            _complete_task(
                root,
                task_id,
                status="完成",
                stage="已完成",
                output_dir=str(result.get("output_dir") or "") or None,
                experiment_dir=str(result.get("experiment_dir") or "") or None,
            )


def _serialize_config(config: RollingResearchConfig) -> dict[str, Any]:
    return {
        "base_config": config.base_config.as_dict(),
        "search_mode": config.search_mode,
        "training_mode": config.training_mode,
        "first_training_end": config.first_training_end,
        "recalibration_interval": config.recalibration_interval,
        "recalibration_unit": config.recalibration_unit,
        "rolling_window_months": config.rolling_window_months,
        "min_objective_improvement": config.min_objective_improvement,
        "task_name": config.task_name,
        "weight_search_version": config.weight_search_version,
        "minimum_training_months": config.minimum_training_months,
        "resume_from": str(config.resume_from) if config.resume_from else None,
    }


def _deserialize_config(raw: object) -> RollingResearchConfig:
    if not isinstance(raw, dict):
        raise ValueError("滚动任务配置损坏")
    base_raw = raw.get("base_config")
    if not isinstance(base_raw, dict):
        raise ValueError("滚动任务缺少基线配置")
    resume_raw = raw.get("resume_from")
    return RollingResearchConfig(
        base_config=strategy_config_from_dict(base_raw),
        search_mode=str(raw.get("search_mode", "combined")),  # type: ignore[arg-type]
        training_mode=str(raw.get("training_mode", "expanding")),  # type: ignore[arg-type]
        first_training_end=str(raw["first_training_end"]) if raw.get("first_training_end") else None,
        recalibration_interval=int(raw.get("recalibration_interval", 3)),
        recalibration_unit=str(raw.get("recalibration_unit", "months")),  # type: ignore[arg-type]
        rolling_window_months=int(raw["rolling_window_months"]) if raw.get("rolling_window_months") is not None else None,
        min_objective_improvement=float(raw.get("min_objective_improvement", 0.0)),
        task_name=str(raw["task_name"]) if raw.get("task_name") else None,
        weight_search_version=str(raw.get("weight_search_version", "v2")),  # type: ignore[arg-type]
        minimum_training_months=int(raw["minimum_training_months"]) if raw.get("minimum_training_months") is not None else None,
        resume_from=Path(str(resume_raw)) if resume_raw else None,
    )


def _claim_next_task(root: Path) -> dict[str, Any] | None:
    with _data_lock(root):
        path = queue_path(root)
        records = _read_records(path)
        active_count = sum(1 for item in records if item.get("status") in {"启动中", "运行中"})
        if active_count >= MAX_QUEUE_WORKERS:
            return None
        for task in records:
            if task.get("status") in {"启动中", "等待中"}:
                task["status"] = "运行中"
                task["stage"] = "正在准备数据"
                task["started_at"] = datetime.now().isoformat(timespec="seconds")
                _write_records(path, records)
                return dict(task)
    return None


def _submission_state(records: list[dict[str, Any]]) -> tuple[str, str]:
    """Choose the presentation state for a newly submitted task.

    ``启动中`` is intentionally claimable by the worker.  It only exists for
    the brief hand-off after an otherwise idle queue receives its first task.
    """
    busy_states = {"启动中", "运行中", "等待中"}
    if not any(str(item.get("status") or "") in busy_states for item in records):
        return "启动中", "正在启动"
    return "等待中", "等待前序任务完成"


def _set_progress(
    root: Path,
    task_id: str,
    stage: str,
    completed_periods: int | None,
    total_periods: int | None,
) -> None:
    with _data_lock(root):
        path = queue_path(root)
        records = _read_records(path)
        changed = False
        for task in records:
            if task.get("task_id") == task_id and task.get("status") == "运行中":
                if task.get("cancel_requested"):
                    task["stage"] = "正在终止（将在当前期结束后停止）"
                else:
                    task["stage"] = str(stage)
                    if completed_periods is not None:
                        task["completed_periods"] = max(0, int(completed_periods))
                    if total_periods is not None:
                        task["total_periods"] = max(0, int(total_periods))
                changed = True
                break
        if changed:
            _write_records(path, records)


def _cancel_requested(root: Path, task_id: str) -> bool:
    """Read cancellation state without modifying the task record."""
    with _data_lock(root):
        records = _read_records(queue_path(root))
        task = next((item for item in records if str(item.get("task_id")) == task_id), None)
        return bool(task and task.get("cancel_requested"))


def _complete_task(root: Path, task_id: str, *, status: str, stage: str, output_dir: str | None = None,
                   experiment_dir: str | None = None, error: str | None = None) -> None:
    with _data_lock(root):
        path = queue_path(root)
        records = _read_records(path)
        for task in records:
            if task.get("task_id") == task_id:
                task.update({
                    "status": status,
                    "stage": stage,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "output_dir": output_dir,
                    "experiment_dir": experiment_dir,
                    "error": error,
                })
                break
        _write_records(path, records)


@contextmanager
def _data_lock(root: Path) -> Iterator[None]:
    lock_path = Path(f"{queue_path(root)}{DATA_LOCK_SUFFIX}")
    _acquire_file_lock(lock_path)
    try:
        yield
    finally:
        _remove_if_exists(lock_path)


@contextmanager
def _worker_lock(root: Path, worker_slot: int = 0) -> Iterator[bool]:
    lock_path = _worker_lock_path(root, worker_slot)
    try:
        _acquire_file_lock(lock_path)
    except TimeoutError:
        yield False
        return
    try:
        yield True
    finally:
        _remove_if_exists(lock_path)


def _worker_lock_path(root: Path, worker_slot: int = 0) -> Path:
    suffix = "" if worker_slot == 0 else f".{int(worker_slot)}"
    return Path(f"{queue_path(root)}{WORKER_LOCK_SUFFIX}{suffix}")


def _acquire_file_lock(path: Path, timeout_seconds: float = 3.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "created_at": datetime.now().isoformat()}, handle)
            return
        except FileExistsError:
            if _is_stale_lock(path):
                _remove_if_exists(path)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"无法获取队列锁：{path.name}")
            time.sleep(0.05)


def _is_stale_lock(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", 0))
        return not _process_exists(pid)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        try:
            return time.time() - path.stat().st_mtime > _LOCK_STALE_SECONDS
        except OSError:
            return True


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [dict(item) for item in raw] if isinstance(raw, list) else []


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
