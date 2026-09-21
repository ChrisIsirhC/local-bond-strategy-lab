"""SQLite metadata repository for immutable archived strategies.

The backtest engine still keeps large numerical artifacts in files.  This
module is intentionally limited to the relational facts that must remain
unique and queryable: strategy identity, presentation state, archive metadata
and artifact fingerprints.  In particular, a strategy ID is the sole public
identifier of an archived strategy; it is never derived by scanning folders.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator


METADATA_PATH = Path("backtest_outputs") / "strategy_metadata.sqlite"
LEGACY_ID_REGISTRY_PATH = Path("backtest_outputs") / "archive_id_registry.json"
_IDENTIFIER_RE = re.compile(r"^[A-Z][0-9]{3,}$")


def metadata_path(root: Path) -> Path:
    return Path(root).resolve() / METADATA_PATH


@contextmanager
def _connection(root: Path) -> Iterator[sqlite3.Connection]:
    path = metadata_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        _create_schema(connection)
        _ensure_schema_extensions(connection)
        _import_legacy_registry_if_needed(connection, Path(root).resolve())
        # Schema creation and the one-time legacy import may write metadata.
        # Commit those setup writes before callers acquire their own explicit
        # IMMEDIATE transaction for ID allocation or archive registration.
        connection.commit()
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS repository_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS strategy_identity (
            strategy_id TEXT PRIMARY KEY
                CHECK (strategy_id GLOB '[A-Z][0-9][0-9][0-9]*'),
            archive_name TEXT NOT NULL UNIQUE,
            source_category TEXT NOT NULL,
            id_prefix TEXT NOT NULL CHECK (length(id_prefix) = 1),
            created_at TEXT NOT NULL,
            archive_path TEXT UNIQUE,
            config_sha256 TEXT,
            manifest_sha256 TEXT
        );

        CREATE TABLE IF NOT EXISTS strategy_sequence (
            id_prefix TEXT PRIMARY KEY CHECK (length(id_prefix) = 1),
            next_number INTEGER NOT NULL CHECK (next_number > 0)
        );

        CREATE TABLE IF NOT EXISTS strategy_presentation (
            strategy_id TEXT PRIMARY KEY REFERENCES strategy_identity(strategy_id)
                ON DELETE CASCADE,
            display_name TEXT,
            note TEXT,
            is_favorite INTEGER NOT NULL DEFAULT 0 CHECK (is_favorite IN (0, 1)),
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS strategy_snapshot (
            strategy_id TEXT PRIMARY KEY REFERENCES strategy_identity(strategy_id)
                ON DELETE CASCADE,
            strategy_name TEXT NOT NULL,
            source TEXT,
            signal_frequency TEXT,
            created_at TEXT,
            config_json TEXT NOT NULL,
            manifest_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS strategy_metric (
            strategy_id TEXT PRIMARY KEY REFERENCES strategy_identity(strategy_id)
                ON DELETE CASCADE,
            metrics_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS strategy_artifact (
            strategy_id TEXT NOT NULL REFERENCES strategy_identity(strategy_id)
                ON DELETE CASCADE,
            artifact_type TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            row_count INTEGER,
            is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0, 1)),
            storage_state TEXT NOT NULL DEFAULT 'active' CHECK (storage_state IN ('active', 'staged')),
            staged_relative_path TEXT,
            PRIMARY KEY (strategy_id, artifact_type, relative_path)
        );

        CREATE TABLE IF NOT EXISTS strategy_lineage (
            strategy_id TEXT NOT NULL REFERENCES strategy_identity(strategy_id)
                ON DELETE CASCADE,
            sequence_no INTEGER NOT NULL,
            parent_strategy_id TEXT REFERENCES strategy_identity(strategy_id),
            operation TEXT NOT NULL,
            changes_json TEXT NOT NULL,
            arguments_json TEXT NOT NULL,
            note TEXT NOT NULL,
            PRIMARY KEY (strategy_id, sequence_no)
        );

        CREATE TRIGGER IF NOT EXISTS strategy_id_is_immutable
        BEFORE UPDATE OF strategy_id ON strategy_identity
        BEGIN
            SELECT RAISE(ABORT, 'strategy_id is immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS strategy_archive_name_is_immutable
        BEFORE UPDATE OF archive_name ON strategy_identity
        BEGIN
            SELECT RAISE(ABORT, 'strategy archive identity is immutable');
        END;

        CREATE INDEX IF NOT EXISTS idx_strategy_identity_archive_path
            ON strategy_identity(archive_path);
        CREATE INDEX IF NOT EXISTS idx_strategy_artifact_sha256
            ON strategy_artifact(sha256);
        """
    )


def _ensure_schema_extensions(connection: sqlite3.Connection) -> None:
    """Additive schema upgrades keep a partially migrated repository usable."""
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(strategy_artifact)")}
    if "storage_state" not in columns:
        connection.execute(
            "ALTER TABLE strategy_artifact ADD COLUMN storage_state TEXT NOT NULL DEFAULT 'active'"
        )
    if "staged_relative_path" not in columns:
        connection.execute("ALTER TABLE strategy_artifact ADD COLUMN staged_relative_path TEXT")


def _import_legacy_registry_if_needed(connection: sqlite3.Connection, root: Path) -> None:
    imported = connection.execute(
        "SELECT value FROM repository_meta WHERE key = 'legacy_id_registry_imported'"
    ).fetchone()
    if imported is not None:
        return
    path = root / LEGACY_ID_REGISTRY_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取旧归档编号注册表，拒绝在不完整编号基础上分配新策略 ID") from exc
    if not isinstance(raw, dict):
        raise ValueError("旧归档编号注册表格式无效")

    owners: dict[str, tuple[str, str]] = {}
    now = datetime.now().isoformat(timespec="seconds")
    max_by_prefix: dict[str, int] = {}
    for category, section in raw.items():
        if not isinstance(section, dict):
            continue
        prefix = str(section.get("prefix", "")).upper().strip()
        items = section.get("items", {})
        if not re.fullmatch(r"[A-Z]", prefix) or not isinstance(items, dict):
            raise ValueError(f"旧归档编号注册表类别 {category!r} 损坏")
        for archive_name, strategy_id in items.items():
            archive_name = str(archive_name)
            strategy_id = str(strategy_id).upper().strip()
            if not _IDENTIFIER_RE.fullmatch(strategy_id) or not strategy_id.startswith(prefix):
                raise ValueError(f"旧归档编号 {strategy_id!r} 不属于 {prefix} 序列")
            owner = owners.setdefault(strategy_id, (str(category), archive_name))
            if owner != (str(category), archive_name):
                raise ValueError(
                    f"旧归档编号重复：{strategy_id} 同时指向 {owner[0]}/{owner[1]} 与 {category}/{archive_name}"
                )
            existing = connection.execute(
                "SELECT strategy_id FROM strategy_identity WHERE archive_name = ?", (archive_name,)
            ).fetchone()
            if existing is not None and existing["strategy_id"] != strategy_id:
                raise ValueError(f"归档 {archive_name} 已绑定 {existing['strategy_id']}，不能改为 {strategy_id}")
            connection.execute(
                """
                INSERT INTO strategy_identity(strategy_id, archive_name, source_category, id_prefix, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(strategy_id) DO NOTHING
                """,
                (strategy_id, archive_name, str(category), prefix, now),
            )
            max_by_prefix[prefix] = max(max_by_prefix.get(prefix, 0), int(strategy_id[1:]))

    for prefix, maximum in max_by_prefix.items():
        connection.execute(
            """
            INSERT INTO strategy_sequence(id_prefix, next_number) VALUES (?, ?)
            ON CONFLICT(id_prefix) DO UPDATE SET next_number = MAX(next_number, excluded.next_number)
            """,
            (prefix, maximum + 1),
        )
    connection.execute(
        "INSERT INTO repository_meta(key, value) VALUES ('legacy_id_registry_imported', ?)",
        (datetime.now().isoformat(timespec="seconds"),),
    )


def ensure_strategy_ids(
    root: Path,
    category: str,
    archive_names: list[str] | tuple[str, ...],
    prefix: str,
) -> dict[str, str]:
    """Atomically allocate immutable strategy IDs for archive names.

    SQLite's write transaction is the only allocator.  We never infer a new
    number from filenames and we never update a previously assigned ID.
    """
    normalized_prefix = str(prefix).upper().strip()
    if not re.fullmatch(r"[A-Z]", normalized_prefix):
        raise ValueError(f"归档编号前缀无效：{prefix!r}")
    names = sorted({str(name) for name in archive_names if str(name).strip()})
    if not names:
        return {}
    with _connection(root) as connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            f"SELECT archive_name, strategy_id, id_prefix FROM strategy_identity WHERE archive_name IN ({','.join('?' for _ in names)})",
            names,
        ).fetchall()
        assigned = {str(row["archive_name"]): str(row["strategy_id"]) for row in rows}
        for row in rows:
            if str(row["id_prefix"]) != normalized_prefix:
                raise ValueError(
                    f"归档 {row['archive_name']} 已绑定 {row['strategy_id']}，不能改为 {normalized_prefix} 序列"
                )
        sequence = connection.execute(
            "SELECT next_number FROM strategy_sequence WHERE id_prefix = ?", (normalized_prefix,)
        ).fetchone()
        next_number = int(sequence["next_number"]) if sequence is not None else 1
        now = datetime.now().isoformat(timespec="seconds")
        for archive_name in names:
            if archive_name in assigned:
                continue
            strategy_id = f"{normalized_prefix}{next_number:03d}"
            connection.execute(
                """
                INSERT INTO strategy_identity(strategy_id, archive_name, source_category, id_prefix, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (strategy_id, archive_name, str(category), normalized_prefix, now),
            )
            assigned[archive_name] = strategy_id
            next_number += 1
        connection.execute(
            """
            INSERT INTO strategy_sequence(id_prefix, next_number) VALUES (?, ?)
            ON CONFLICT(id_prefix) DO UPDATE SET next_number = excluded.next_number
            """,
            (normalized_prefix, next_number),
        )
        result = {name: assigned[name] for name in archive_names if name in assigned}
    # Archive lookups are a hot UI path.  A just-created binding must become
    # visible immediately, including when an earlier render cached ``None``.
    _strategy_id_lookup.cache_clear()
    _strategy_id_by_identifier_lookup.cache_clear()
    return result


@lru_cache(maxsize=4096)
def _strategy_id_lookup(root_text: str, archive_name: str) -> str | None:
    try:
        with _connection(Path(root_text)) as connection:
            row = connection.execute(
                "SELECT strategy_id FROM strategy_identity WHERE archive_name = ?", (str(archive_name),)
            ).fetchone()
            return str(row["strategy_id"]) if row is not None else None
    except (sqlite3.Error, ValueError):
        return None


@lru_cache(maxsize=4096)
def _strategy_id_by_identifier_lookup(root_text: str, strategy_id: str) -> str | None:
    """Read a known ID without turning every public-page cell into a DB open."""
    try:
        with _connection(Path(root_text)) as connection:
            row = connection.execute(
                "SELECT strategy_id FROM strategy_identity WHERE strategy_id = ?", (strategy_id,)
            ).fetchone()
            return str(row["strategy_id"]) if row is not None else None
    except (sqlite3.Error, ValueError):
        return None


def strategy_id_for_archive(root: Path, archive_name: str) -> str | None:
    """Read an immutable strategy identity without reopening SQLite on each UI render."""
    # Public deployments may address an archive by its immutable strategy ID
    # rather than the original (often very long) timestamped folder name.
    # Accept that compact form only when it is already present in the registry;
    # this never creates or mutates an ID.
    normalized = str(archive_name).upper().strip()
    if _IDENTIFIER_RE.fullmatch(normalized):
        return _strategy_id_by_identifier_lookup(str(Path(root).resolve()), normalized)
    return _strategy_id_lookup(str(Path(root).resolve()), str(archive_name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_strategy_archive(
    root: Path,
    *,
    strategy_id: str,
    archive_dir: Path,
    manifest: dict[str, Any],
    config_payload: dict[str, Any],
    strategy_metrics: dict[str, Any],
) -> None:
    """Attach immutable archive facts to an already allocated strategy ID."""
    archive_dir = Path(archive_dir).resolve()
    root = Path(root).resolve()
    archive_name = archive_dir.name
    manifest_path = archive_dir / "run_manifest.json"
    config_path = archive_dir / "config.json"
    if not manifest_path.exists() or not config_path.exists():
        raise FileNotFoundError("策略归档缺少 config.json 或 run_manifest.json")
    with _connection(root) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT strategy_id, archive_name FROM strategy_identity WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        if row is None or str(row["archive_name"]) != archive_name:
            raise ValueError(f"策略 ID {strategy_id} 未绑定当前归档 {archive_name}")
        archive_relative = str(archive_dir.relative_to(root)).replace("\\", "/")
        connection.execute(
            """
            UPDATE strategy_identity
            SET archive_path = ?, config_sha256 = ?, manifest_sha256 = ?
            WHERE strategy_id = ?
            """,
            (archive_relative, _sha256(config_path), _sha256(manifest_path), strategy_id),
        )
        connection.execute(
            """
            INSERT INTO strategy_snapshot(strategy_id, strategy_name, source, signal_frequency, created_at, config_json, manifest_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(strategy_id) DO UPDATE SET
                strategy_name = excluded.strategy_name,
                source = excluded.source,
                signal_frequency = excluded.signal_frequency,
                created_at = excluded.created_at,
                config_json = excluded.config_json,
                manifest_json = excluded.manifest_json
            """,
            (
                strategy_id,
                str(manifest.get("策略名称", config_payload.get("name", ""))),
                str(manifest.get("运行来源", "")),
                str(manifest.get("信号频率", "")),
                str(manifest.get("运行时间", "")),
                json.dumps(config_payload, ensure_ascii=False, sort_keys=True, default=str),
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, default=str),
            ),
        )
        connection.execute(
            """
            INSERT INTO strategy_metric(strategy_id, metrics_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(strategy_id) DO UPDATE SET metrics_json = excluded.metrics_json, updated_at = excluded.updated_at
            """,
            (strategy_id, json.dumps(strategy_metrics, ensure_ascii=False, sort_keys=True, default=str), datetime.now().isoformat(timespec="seconds")),
        )
        connection.execute("DELETE FROM strategy_artifact WHERE strategy_id = ?", (strategy_id,))
        for path in sorted(archive_dir.iterdir()):
            if not path.is_file():
                continue
            row_count = None
            if path.suffix.lower() == ".csv":
                try:
                    with path.open("rb") as handle:
                        row_count = max(sum(1 for _ in handle) - 1, 0)
                except OSError:
                    pass
            artifact_type = path.stem
            connection.execute(
                """
                INSERT INTO strategy_artifact(strategy_id, artifact_type, relative_path, sha256, byte_size, row_count, is_primary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    artifact_type,
                    str(path.relative_to(root)).replace("\\", "/"),
                    _sha256(path),
                    int(path.stat().st_size),
                    row_count,
                    1 if path.name in {"strategy_nav.csv", "signal_score.csv", "config.json", "run_manifest.json"} else 0,
                ),
            )


def mark_artifact_staged(
    root: Path,
    *,
    strategy_id: str,
    relative_path: str,
    staged_relative_path: str,
) -> None:
    """Keep the metadata audit trail when a derived artifact enters quarantine."""
    with _connection(root) as connection:
        connection.execute(
            """
            UPDATE strategy_artifact
            SET storage_state = 'staged', staged_relative_path = ?
            WHERE strategy_id = ? AND relative_path = ?
            """,
            (staged_relative_path, strategy_id, relative_path.replace("\\", "/")),
        )


def register_external_artifact(
    root: Path,
    *,
    strategy_id: str,
    artifact_type: str,
    relative_path: str,
    byte_size: int,
    sha256: str,
    row_count: int | None = None,
    is_primary: bool = False,
) -> None:
    """Record a columnar artifact that is stored outside the legacy folder."""
    with _connection(root) as connection:
        connection.execute(
            """
            INSERT INTO strategy_artifact(
                strategy_id, artifact_type, relative_path, sha256, byte_size, row_count, is_primary,
                storage_state, staged_relative_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', NULL)
            ON CONFLICT(strategy_id, artifact_type, relative_path) DO UPDATE SET
                sha256 = excluded.sha256,
                byte_size = excluded.byte_size,
                row_count = excluded.row_count,
                is_primary = excluded.is_primary,
                storage_state = 'active',
                staged_relative_path = NULL
            """,
            (strategy_id, artifact_type, relative_path.replace("\\", "/"), sha256, byte_size, row_count, int(is_primary)),
        )


def repository_summary(root: Path) -> dict[str, int]:
    with _connection(root) as connection:
        return {
            "strategies": int(connection.execute("SELECT COUNT(*) FROM strategy_identity").fetchone()[0]),
            "snapshots": int(connection.execute("SELECT COUNT(*) FROM strategy_snapshot").fetchone()[0]),
            "artifacts": int(connection.execute("SELECT COUNT(*) FROM strategy_artifact").fetchone()[0]),
        }
