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

from common.strategy_repository import strategy_id_for_archive


def bridge(root: Path, strategy_id: str, *, base_index: Path | None = None) -> Path:
    root = Path(root).resolve()
    normalized_id = strategy_id.upper().strip()
    index_path = root / "backtest_outputs" / "experiments_index.json"
    local_payload = json.loads(index_path.read_text(encoding="utf-8"))
    local_rows = local_payload.get("rows", []) if isinstance(local_payload, dict) else []
    row = next((item for item in local_rows if str(item.get("运行ID", "")).upper() == normalized_id), None)
    if not isinstance(row, dict):
        raise ValueError(f"历史索引未找到策略 {normalized_id}")
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

    target = root / "backtest_outputs" / "experiments" / normalized_id
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "run_manifest.json"):
        file = source / name
        if not file.is_file():
            raise FileNotFoundError(f"策略 {normalized_id} 缺少 {name}")
        destination = target / name
        if file.resolve() != destination.resolve():
            shutil.copy2(file, destination)

    if base_index is not None:
        # The public baseline already contains the older compact-ID rows.  Add
        # this new row to that baseline instead of replacing every historical
        # long path with workstation-specific paths.
        payload = json.loads(Path(base_index).read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        # Normalize every public row with its immutable long archive key.  Old
        # deployments only stored ``实验目录=.../<short-id>``; their manifest
        # still carries ``archive_key`` and is the authoritative bridge back
        # to the timestamped local directory.
        for item in rows:
            if not isinstance(item, dict):
                continue
            archive_name = str(item.get("archive_name") or item.get("archive_key") or "").strip()
            if not archive_name:
                compact = root / str(item.get("实验目录", "")).replace("\\", "/")
                manifest = compact / "run_manifest.json"
                try:
                    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest_payload = {}
                archive_name = str(manifest_payload.get("archive_key") or "").strip()
            if not archive_name:
                # Older compact manifests predate ``archive_key``.  The
                # repository database remains the authoritative reverse map.
                try:
                    with sqlite3.connect(root / "backtest_outputs" / "strategy_metadata.sqlite") as connection:
                        found = connection.execute(
                            "SELECT archive_name FROM strategy_identity WHERE strategy_id = ?",
                            (str(item.get("运行ID", "")).upper().strip(),),
                        ).fetchone()
                    archive_name = str(found[0]).strip() if found else ""
                except sqlite3.Error:
                    archive_name = ""
            if archive_name:
                item["archive_name"] = archive_name
                item["archive_key"] = archive_name
        rows = [item for item in rows if str(item.get("运行ID", "")).upper() != normalized_id]
        public_row = dict(row)
        public_row["实验目录"] = str(target.relative_to(root)).replace("\\", "/")
        # Keep the immutable, timestamped archive identity beside the compact
        # public entry point.  ``strategy_id`` is the stable lookup key, while
        # ``archive_name`` disambiguates copies between local and remote
        # workspaces and lets an importer verify that they are the same run.
        public_row["archive_name"] = source.name
        public_row["archive_key"] = source.name
        rows.append(public_row)
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
