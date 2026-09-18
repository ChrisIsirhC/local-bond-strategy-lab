from __future__ import annotations

from pathlib import Path
import json
import re

from common.strategy_repository import ensure_strategy_ids, strategy_id_for_archive


REGISTRY_PATH = Path("backtest_outputs") / "archive_id_registry.json"
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
    """Return stable IDs from the SQLite source of truth.

    ``archive_id_registry.json`` is imported once for backward compatibility,
    but no longer participates in allocation or concurrent writes.
    """
    return ensure_strategy_ids(Path(root), category, archive_names, prefix)


def short_archive_id(root: Path, category: str, archive_name: str, prefix: str) -> str:
    return ensure_short_archive_ids(root, category, [archive_name], prefix)[archive_name]


def existing_short_archive_id(root: Path, category: str, archive_name: str) -> str | None:
    """Read an already assigned identifier without acquiring the write lock.

    UI-only paths must never create an ID while a worker is committing an
    archive.  Missing or temporarily unreadable registry data simply means the
    caller displays the strategy name without a short ID for that render.
    """
    return strategy_id_for_archive(Path(root), archive_name)
