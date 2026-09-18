"""Columnar storage for immutable archived strategy results.

Metadata lives in SQLite.  Large numerical evidence lives once in this
Parquet/ZSTD store, keyed solely by the immutable strategy ID.  Archive-local
CSV remains a compatibility fallback during migration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

import pandas as pd

from common.strategy_repository import mark_artifact_staged, register_external_artifact, strategy_id_for_archive


RESULT_STORE_PATH = Path("backtest_outputs") / "result_store"
_SUPPORTED = {"strategy_nav", "signal_score", "rolling_periods"}
_ROLLING_REPRODUCTION_DIR = "rolling_reproduction"


def _root_for_archive(archive_dir: Path) -> Path:
    archive_dir = Path(archive_dir).resolve()
    # <root>/backtest_outputs/experiments/<archive-name>
    if archive_dir.parent.name == "experiments" and archive_dir.parent.parent.name == "backtest_outputs":
        return archive_dir.parent.parent.parent
    raise ValueError(f"无法识别策略归档目录：{archive_dir}")


def _store_dir(root: Path, strategy_id: str) -> Path:
    return Path(root).resolve() / RESULT_STORE_PATH / strategy_id


def rolling_reproduction_dir(root: Path, strategy_id: str) -> Path:
    """Return the compact, strategy-owned rolling replay contract directory."""
    return _store_dir(root, strategy_id) / _ROLLING_REPRODUCTION_DIR


def write_rolling_reproduction_bundle(
    root: Path,
    strategy_id: str,
    *,
    rolling_dir: Path | None,
    archive_config_path: Path,
) -> Path | None:
    """Persist everything needed to reuse rolling periods without a work directory.

    Rolling workspaces are disposable execution scratch space.  The immutable
    strategy archive instead owns a small replay contract: run settings,
    baseline, period table and the selected configuration for every period.
    This intentionally stores JSON rather than a copy of daily numerical data;
    the latter already lives in the Parquet result store.
    """
    root = Path(root).resolve()
    archive_config_path = Path(archive_config_path)
    source_dir = Path(rolling_dir).resolve() if rolling_dir else None
    destination = rolling_reproduction_dir(root, strategy_id)
    destination.mkdir(parents=True, exist_ok=True)

    periods: pd.DataFrame | None = None
    source_manifest: dict[str, object] = {}
    if source_dir is not None and source_dir.is_dir():
        manifest_path = source_dir / "滚动配置.json"
        if manifest_path.exists():
            try:
                source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                source_manifest = {}
        period_path = source_dir / "逐期定参与样本外表现.csv"
        if period_path.exists():
            try:
                periods = pd.read_csv(period_path, encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError, ValueError):
                periods = None
    if periods is None:
        parquet = _store_dir(root, strategy_id) / "rolling_periods.parquet"
        if parquet.exists():
            periods = pd.read_parquet(parquet, engine="pyarrow")
    if periods is None or periods.empty:
        return None

    # ``基线配置`` is required by the rolling engine solely to prove the
    # reusable period contract belongs to the strategy being replayed.  The
    # source copy wins; an archive config is an equivalent immutable fallback.
    baseline_source = source_dir / "基线配置.json" if source_dir is not None else None
    baseline_target = destination / "基线配置.json"
    if baseline_source is not None and baseline_source.exists():
        shutil.copy2(baseline_source, baseline_target)
    elif archive_config_path.exists():
        shutil.copy2(archive_config_path, baseline_target)
    else:
        return None

    normalized = periods.copy()
    config_column = "参数配置文件"
    factor_period_contract = config_column not in normalized.columns and "权重" in normalized.columns
    period_root = destination / "periods"
    period_root.mkdir(parents=True, exist_ok=True)
    fallback_configs = (
        sorted((source_dir / "periods").glob("*/参数配置.json"))
        if source_dir is not None and (source_dir / "periods").exists()
        else []
    )
    copied = 0
    for index, (_, row) in enumerate(normalized.iterrows()):
        candidates: list[Path] = []
        raw = row.get(config_column, "")
        if isinstance(raw, str) and raw.strip():
            recorded = Path(raw)
            candidates.append(recorded)
            if source_dir is not None and not recorded.is_absolute():
                candidates.append(source_dir / recorded)
        if index < len(fallback_configs):
            candidates.append(fallback_configs[index])
        config_source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if config_source is None:
            continue
        label = str(row.get("期数", index + 1)).strip() or str(index + 1)
        config_target = period_root / f"{int(float(label)):02d}" / "参数配置.json"
        config_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_source, config_target)
        normalized.at[normalized.index[index], config_column] = str(config_target.relative_to(destination)).replace("\\", "/")
        copied += 1

    # A partial bundle would silently turn a replay into new searches, which
    # violates reproducibility.  Leave the evidence in place but do not mark
    # this contract usable until every historical period has its parameters.
    if not factor_period_contract and copied != len(normalized):
        return None
    normalized.to_csv(destination / "逐期定参与样本外表现.csv", index=False, encoding="utf-8-sig")
    contract = {
        "strategy_id": strategy_id,
        "format": "rolling-reproduction-v1",
        "contract_kind": "factor-period-values" if factor_period_contract else "parameter-config-files",
        "source": "rolling-workspace" if source_dir is not None and source_dir.is_dir() else "archived-periods",
        "period_count": int(len(normalized)),
        "manifest": source_manifest,
    }
    (destination / "滚动配置.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for filename, artifact_type in {
        "滚动配置.json": "rolling_reproduction_manifest",
        "基线配置.json": "rolling_reproduction_baseline",
        "逐期定参与样本外表现.csv": "rolling_reproduction_periods",
    }.items():
        path = destination / filename
        register_external_artifact(
            root,
            strategy_id=strategy_id,
            artifact_type=artifact_type,
            relative_path=str(path.relative_to(root)).replace("\\", "/"),
            byte_size=int(path.stat().st_size),
            sha256=_sha256_file(path),
            row_count=int(len(normalized)) if filename.endswith(".csv") else None,
            is_primary=False,
        )
    return destination


def read_rolling_reproduction_manifest(root: Path, strategy_id: str) -> dict[str, object]:
    """Read a central rolling replay contract without falling back to scratch files."""
    path = rolling_reproduction_dir(root, strategy_id) / "滚动配置.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _fingerprint(frame: pd.DataFrame) -> str:
    # Stable across Parquet round trips while keeping the check independent of
    # Parquet metadata, compression version and row-group layout.
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    content = normalized.to_csv(index=False, lineterminator="\n", na_rep="<NA>").encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def write_result_frame(root: Path, strategy_id: str, artifact_type: str, frame: pd.DataFrame) -> Path:
    if artifact_type not in _SUPPORTED:
        raise ValueError(f"不支持的结果类型：{artifact_type}")
    destination = _store_dir(root, strategy_id) / f"{artifact_type}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    restored = pd.read_parquet(temporary, engine="pyarrow")
    if list(restored.columns) != list(frame.columns) or len(restored) != len(frame) or _fingerprint(restored) != _fingerprint(frame):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Parquet 校验失败：{strategy_id}/{artifact_type}")
    temporary.replace(destination)
    _write_manifest(root, strategy_id, artifact_type, frame, destination)
    register_external_artifact(
        root,
        strategy_id=strategy_id,
        artifact_type=f"{artifact_type}_parquet",
        relative_path=str(destination.relative_to(Path(root).resolve())),
        byte_size=int(destination.stat().st_size),
        sha256=_sha256_file(destination),
        row_count=int(len(frame)),
        is_primary=artifact_type in {"strategy_nav", "signal_score"},
    )
    return destination


def _write_manifest(root: Path, strategy_id: str, artifact_type: str, frame: pd.DataFrame, path: Path) -> None:
    manifest_path = _store_dir(root, strategy_id) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        manifest = {}
    manifest.setdefault("strategy_id", strategy_id)
    manifest.setdefault("format", "Parquet/ZSTD")
    manifest.setdefault("artifacts", {})
    manifest["artifacts"][artifact_type] = {
        "path": str(path.relative_to(Path(root).resolve())).replace("\\", "/"),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "fingerprint": _fingerprint(frame),
        "bytes": int(path.stat().st_size),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_archive_frame(archive_dir: Path, artifact_type: str, *, csv_name: str | None = None) -> pd.DataFrame:
    """Read columnar results first and use legacy CSV only during migration."""
    if artifact_type not in _SUPPORTED:
        raise ValueError(f"不支持的结果类型：{artifact_type}")
    archive_dir = Path(archive_dir).resolve()
    root = _root_for_archive(archive_dir)
    strategy_id = strategy_id_for_archive(root, archive_dir.name)
    if strategy_id:
        parquet_path = _store_dir(root, strategy_id) / f"{artifact_type}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path, engine="pyarrow")
    legacy = archive_dir / (csv_name or f"{artifact_type}.csv")
    if not legacy.exists():
        raise FileNotFoundError(f"策略结果缺失：{artifact_type}")
    return pd.read_csv(legacy, encoding="utf-8-sig")


def migrate_archive_result(root: Path, archive_dir: Path) -> list[Path]:
    """Copy result evidence to Parquet and verify it before any CSV is staged."""
    root = Path(root).resolve()
    archive_dir = Path(archive_dir).resolve()
    strategy_id = strategy_id_for_archive(root, archive_dir.name)
    if not strategy_id:
        raise ValueError(f"归档尚未登记策略 ID：{archive_dir.name}")
    written: list[Path] = []
    for artifact_type in ("strategy_nav", "signal_score"):
        source = archive_dir / f"{artifact_type}.csv"
        if not source.exists():
            raise FileNotFoundError(f"归档缺少 {source.name}：{archive_dir.name}")
        frame = pd.read_csv(source, encoding="utf-8-sig")
        written.append(write_result_frame(root, strategy_id, artifact_type, frame))
    return written


def migrate_rolling_periods(root: Path, strategy_id: str, source: Path) -> Path | None:
    source = Path(source)
    if not source.exists():
        return None
    return write_result_frame(root, strategy_id, "rolling_periods", pd.read_csv(source, encoding="utf-8-sig"))


def sync_result_store_metadata(root: Path) -> int:
    """Register already migrated Parquet files without rewriting numerical data."""
    root = Path(root).resolve()
    registered = 0
    for manifest_path in sorted((root / RESULT_STORE_PATH).glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            strategy_id = str(manifest["strategy_id"])
            artifacts = manifest.get("artifacts", {})
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(artifacts, dict):
            continue
        for artifact_type, payload in artifacts.items():
            if artifact_type not in _SUPPORTED or not isinstance(payload, dict):
                continue
            path = root / str(payload.get("path", ""))
            if not path.exists():
                continue
            register_external_artifact(
                root,
                strategy_id=strategy_id,
                artifact_type=f"{artifact_type}_parquet",
                relative_path=str(path.relative_to(root)),
                byte_size=int(path.stat().st_size),
                sha256=_sha256_file(path),
                row_count=int(payload.get("rows", 0)),
                is_primary=artifact_type in {"strategy_nav", "signal_score"},
            )
            registered += 1
    return registered


def stage_legacy_archive_frames(root: Path, archive_dir: Path, *, quarantine_root: Path) -> list[dict[str, object]]:
    """Move only parity-checked legacy primary CSVs into the deletion queue."""
    root = Path(root).resolve()
    archive_dir = Path(archive_dir).resolve()
    strategy_id = strategy_id_for_archive(root, archive_dir.name)
    if not strategy_id:
        raise ValueError(f"归档尚未登记策略 ID：{archive_dir.name}")
    records: list[dict[str, object]] = []
    for artifact_type in ("strategy_nav", "signal_score"):
        source = archive_dir / f"{artifact_type}.csv"
        if not source.exists():
            continue
        original = pd.read_csv(source, encoding="utf-8-sig")
        parquet = _store_dir(root, strategy_id) / f"{artifact_type}.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"缺少已验证的 Parquet：{strategy_id}/{artifact_type}")
        restored = pd.read_parquet(parquet, engine="pyarrow")
        if list(restored.columns) != list(original.columns) or len(restored) != len(original) or _fingerprint(restored) != _fingerprint(original):
            raise ValueError(f"拒绝暂存：{strategy_id}/{artifact_type} 与原 CSV 校验不一致")
        target = Path(quarantine_root) / archive_dir.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        relative_source = str(source.relative_to(root)).replace("\\", "/")
        relative_target = str(target.relative_to(root)).replace("\\", "/")
        checksum = _sha256_file(source)
        shutil.move(str(source), str(target))
        mark_artifact_staged(
            root,
            strategy_id=strategy_id,
            relative_path=relative_source,
            staged_relative_path=relative_target,
        )
        records.append({
            "strategy_id": strategy_id,
            "source": relative_source,
            "staged": relative_target,
            "sha256": checksum,
            "bytes": int(target.stat().st_size),
        })
    return records
