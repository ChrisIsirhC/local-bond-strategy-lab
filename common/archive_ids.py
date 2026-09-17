from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import time
from typing import Iterator


REGISTRY_PATH = Path("backtest_outputs") / "archive_id_registry.json"
_LOCK_SUFFIX = ".lock"
_LOCK_STALE_SECONDS = 120
_IDENTIFIER_RE = re.compile(r"^[A-Z][0-9]{3,}$")


def archive_id_category(prefix: str) -> str:
    """Return the shared registry namespace for a displayed run prefix."""
    normalized = str(prefix).upper()
    if normalized == "F":
        # Factor-study batches and their archived strategy results share one
        # sequence so every visible F identifier is unique.
        return "factor_studies"
    return f"experiments_{normalized}"


def ensure_short_archive_ids(
    root: Path,
    category: str,
    archive_names: list[str] | tuple[str, ...],
    prefix: str,
) -> dict[str, str]:
    """Return stable short IDs while preserving immutable archive directory names."""
    path = Path(root) / REGISTRY_PATH
    with _registry_lock(path):
        try:
            registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            registry = {}
        normalized_prefix = str(prefix).upper().strip()
        if not re.fullmatch(r"[A-Z]", normalized_prefix):
            raise ValueError(f"归档编号前缀无效：{prefix!r}")
        section = registry.get(category)
        if not isinstance(section, dict):
            section = {}
        saved_prefix = str(section.get("prefix", "")).upper().strip()
        if saved_prefix and saved_prefix != normalized_prefix:
            raise ValueError(
                f"归档编号注册表类别 {category} 已绑定前缀 {saved_prefix}，不能改为 {normalized_prefix}"
            )
        items = section.get("items")
        if not isinstance(items, dict):
            items = {}
        items = {str(key): str(value) for key, value in items.items()}
        owners: dict[str, str] = {}
        for name, identifier in items.items():
            if not _IDENTIFIER_RE.fullmatch(identifier) or not identifier.startswith(normalized_prefix):
                raise ValueError(f"归档编号注册表损坏：{name} 的编号 {identifier!r} 不属于 {normalized_prefix} 序列")
            previous = owners.setdefault(identifier, name)
            if previous != name:
                raise ValueError(
                    f"归档编号注册表损坏：{identifier} 同时指向 {previous} 与 {name}"
                )
        # The complete identifier (for example F200) is the immutable key,
        # not merely a display label.  Reject a collision even if a caller
        # accidentally uses a second registry category for the same prefix.
        global_owners: dict[str, tuple[str, str]] = {}
        for existing_category, existing_section in registry.items():
            if not isinstance(existing_section, dict):
                continue
            existing_items = existing_section.get("items", {})
            if not isinstance(existing_items, dict):
                continue
            for existing_name, existing_identifier in existing_items.items():
                identifier_text = str(existing_identifier)
                owner = global_owners.setdefault(identifier_text, (str(existing_category), str(existing_name)))
                if owner != (str(existing_category), str(existing_name)):
                    raise ValueError(
                        f"归档编号注册表损坏：{identifier_text} 同时指向 "
                        f"{owner[0]}/{owner[1]} 与 {existing_category}/{existing_name}"
                    )
        used_numbers = [
            int(value[len(normalized_prefix):])
            for value in items.values()
            if value.startswith(normalized_prefix) and value[len(normalized_prefix):].isdigit()
        ]
        next_number = max(used_numbers, default=0) + 1
        changed = False
        for name in sorted({str(value) for value in archive_names if str(value)}):
            if name in items:
                continue
            candidate = f"{normalized_prefix}{next_number:03d}"
            if candidate in global_owners:
                owner_category, owner_name = global_owners[candidate]
                raise ValueError(
                    f"归档编号冲突：{candidate} 已属于 {owner_category}/{owner_name}，拒绝覆盖或复用"
                )
            items[name] = candidate
            global_owners[candidate] = (category, name)
            next_number += 1
            changed = True
        if changed or category not in registry:
            registry[category] = {"prefix": normalized_prefix, "next_number": next_number, "items": items}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        return {name: items[name] for name in archive_names if name in items}


def short_archive_id(root: Path, category: str, archive_name: str, prefix: str) -> str:
    return ensure_short_archive_ids(root, category, [archive_name], prefix)[archive_name]


def existing_short_archive_id(root: Path, category: str, archive_name: str) -> str | None:
    """Read an already assigned identifier without acquiring the write lock.

    UI-only paths must never create an ID while a worker is committing an
    archive.  Missing or temporarily unreadable registry data simply means the
    caller displays the strategy name without a short ID for that render.
    """
    path = Path(root) / REGISTRY_PATH
    try:
        registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        section = registry.get(category, {}) if isinstance(registry, dict) else {}
        items = section.get("items", {}) if isinstance(section, dict) else {}
        value = items.get(str(archive_name)) if isinstance(items, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
    return str(value) if value else None


@contextmanager
def _registry_lock(path: Path, timeout_seconds: float = 5.0) -> Iterator[None]:
    """Serialize registry read-modify-write across dashboard and workers."""
    lock_path = Path(f"{path}{_LOCK_SUFFIX}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "created_at": time.time()}, handle)
            break
        except FileExistsError:
            if _registry_lock_is_stale(lock_path):
                _remove_lock(lock_path)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("归档编号注册表正被其他任务写入，请稍后重试")
            time.sleep(0.05)
    try:
        yield
    finally:
        _remove_lock(lock_path)


def _registry_lock_is_stale(lock_path: Path) -> bool:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", 0))
        if pid > 0:
            try:
                os.kill(pid, 0)
                return False
            except OSError:
                return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    try:
        return time.time() - lock_path.stat().st_mtime > _LOCK_STALE_SECONDS
    except OSError:
        return True


def _remove_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
