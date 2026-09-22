"""Create the compact public archive view for one already-verified strategy.

The local workstation keeps timestamped experiment folders, while the public
Streamlit package addresses result pages through their immutable strategy IDs.
Only immutable archive metadata is copied here; numerical result evidence stays
in the central ``result_store/<strategy_id>`` directory.  The long timestamped
archive remains the canonical cross-machine identity and is never replaced.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from common.experiments import _normalise_index_rows
from common.strategy_repository import archive_uid_from_value, strategy_id_for_archive


def bridge(root: Path, strategy_id: str, *, base_index: Path | None = None) -> Path:
    root = Path(root).resolve()
    normalized_id = strategy_id.upper().strip()
    index_path = root / "backtest_outputs" / "experiments_index.json"
    local_payload = json.loads(index_path.read_text(encoding="utf-8"))
    local_rows = local_payload.get("rows", []) if isinstance(local_payload, dict) else []
    local_rows = _normalise_index_rows(root, local_rows)
    row = next((item for item in local_rows if str(item.get("strategy_id") or item.get("运行ID", "")).upper() == normalized_id), None)
    if not isinstance(row, dict):
        raise ValueError(f"历史索引未找到策略 {normalized_id}")
    archive_uid = str(row.get("archive_uid") or "").strip()
    if not archive_uid:
        raise ValueError(f"策略 {normalized_id} 缺少唯一归档键，拒绝公开同步")
    source = Path(str(row.get("实验目录", "")))
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    # Windows history indexes written by older builds may lowercase the
    # archive component.  Resolve the actual directory name before querying
    # SQLite, whose archive_name comparison is intentionally exact.
    if source.parent.is_dir():
        canonical = next(
            (candidate for candidate in source.parent.iterdir() if candidate.name.casefold() == source.name.casefold()),
            None,
        )
        if canonical is not None:
            source = canonical
    # A previously published index points at the compact directory.  Resolve
    # its canonical timestamped archive through the immutable repository row
    # before copying, so republishing never loses the long identity.
    if source.name.upper() == normalized_id:
        metadata = root / "backtest_outputs" / "strategy_metadata.sqlite"
        try:
            with sqlite3.connect(metadata) as connection:
                found = connection.execute(
                    "SELECT archive_name FROM strategy_identity WHERE strategy_id = ?",
                    (normalized_id,),
                ).fetchone()
        except sqlite3.Error:
            found = None
        if found and str(found[0]).strip():
            candidate = source.parent / str(found[0]).strip()
            if candidate.is_dir():
                source = candidate
    if not source.is_dir() or strategy_id_for_archive(root, source.name) != normalized_id:
        raise ValueError(f"策略 {normalized_id} 的本地归档不完整或 ID 不匹配")

    # On import, the timestamp UID is compared first.  A short-ID collision
    # across local and remote repositories is expected and must allocate the
    # next number in the target series instead of replacing either strategy.
    public_id = normalized_id
    base_payload: dict[str, object] | None = None
    base_rows: list[dict[str, object]] = []
    if base_index is not None:
        raw_payload = json.loads(Path(base_index).read_text(encoding="utf-8"))
        base_payload = raw_payload if isinstance(raw_payload, dict) else {}
        base_rows = [item for item in base_payload.get("rows", []) if isinstance(item, dict)]
        same_uid = next((item for item in base_rows if str(item.get("archive_uid") or "") == archive_uid), None)
        if same_uid is not None:
            public_id = str(same_uid.get("strategy_id") or same_uid.get("运行ID") or normalized_id).upper().strip()
        elif any(str(item.get("strategy_id") or item.get("运行ID") or "").upper().strip() == public_id for item in base_rows):
            prefix = public_id[:1]
            used = {
                int(identifier[1:])
                for item in base_rows
                for identifier in [str(item.get("strategy_id") or item.get("运行ID") or "").upper().strip()]
                if identifier.startswith(prefix) and identifier[1:].isdigit()
            }
            next_number = max(used, default=0) + 1
            while next_number in used:
                next_number += 1
            public_id = f"{prefix}{next_number:03d}"

    target = root / "backtest_outputs" / "experiments" / public_id
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "run_manifest.json"):
        file = source / name
        if not file.is_file():
            raise FileNotFoundError(f"策略 {normalized_id} 缺少 {name}")
        destination = target / name
        if file.resolve() != destination.resolve():
            shutil.copy2(file, destination)
    # The copied public manifest is a target-repository view.  Preserve its
    # immutable archive UID while assigning the target repository's local ID.
    manifest_file = target / "run_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"策略 {normalized_id} 的运行清单格式无效")
    manifest["short_id"] = public_id
    manifest["archive_uid"] = archive_uid
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if base_payload is not None:
        # The public baseline already contains the older compact-ID rows.  Add
        # this new row to that baseline instead of replacing every historical
        # long path with workstation-specific paths.
        payload = base_payload
        rows = [item for item in base_rows if str(item.get("archive_uid") or "") != archive_uid]
        public_row = dict(row)
        public_row["实验目录"] = str(target.relative_to(root)).replace("\\", "/")
        public_row["storage_path"] = public_row["实验目录"]
        public_row["strategy_id"] = public_id
        public_row["运行ID"] = public_id
        # The timestamp-only key disambiguates copies between repositories.
        # Do not put a title-bearing archive directory into the public index.
        public_row["archive_uid"] = archive_uid
        public_row.pop("archive_name", None)
        public_row.pop("archive_key", None)
        rows.append(public_row)
        rows = _normalise_index_rows(root, rows)
        payload["version"] = 9
        payload["rows"] = rows
        index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    # Without a base index, keep the long archive path exactly as-is.  The
    # compact directory is only an additional entry point, not a replacement.
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="生成公开部署的短策略 ID 归档桥接目录")
    parser.add_argument("strategy_id")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--base-index", type=Path, default=None)
    args = parser.parse_args()
    print(bridge(args.root, args.strategy_id, base_index=args.base_index))


if __name__ == "__main__":
    main()
