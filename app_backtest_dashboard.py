from __future__ import annotations

from datetime import datetime
from dataclasses import replace
from html import escape
import json
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from common.archive_ids import archive_id_category, ensure_short_archive_ids, existing_short_archive_id, short_archive_id
from common.config import (
    CONFIG_DIR,
    DashboardStrategyConfig,
    ObjectiveConfig,
    load_strategy_config,
    save_strategy_config,
    strategy_config_from_dict,
)
from common.combined_research import run_combined_search
from common.experiments import archive_dashboard_experiment, archive_id_prefix, list_experiments, load_experiment_result
from common.market_data import CONDITIONAL_BENCHMARK_NAME, GOV_10Y, load_market_data, market_date_bounds
from common.period_evaluation import DEFAULT_TRAINING_END, evaluate_period, evaluate_periods, generalization_summary
from common.performance import performance_metrics
from common.provenance import (
    MISSING_PROVENANCE,
    UNKNOWN_HISTORY,
    config_parameters,
    display_provenance,
    relative_path,
    record_manual_changes,
    record_step,
)
from common.reporting import _build_period_diagnostics
from common.runner import run_dashboard_config
from common.runner import run_dashboard_weight_search_v1, run_dashboard_weight_search_v2
from common.rolling_research import RollingResearchConfig, run_rolling_research
from common.rolling_task_queue import cancel_rolling_task, enqueue_factor_rolling_task, enqueue_rolling_task, list_rolling_tasks, remove_failed_rolling_task, retry_rolling_task, start_rolling_queue_worker
from common.threshold_research import run_threshold_research
from common.trade_metrics import capital_gain_trade_metrics, capital_gain_trade_table
from common.factor_expansion_features import ALL_FACTOR_COLUMNS, BASE_FACTOR_COLUMNS, EXPANDED_FACTOR_COLUMNS, FACTOR_DISPLAY_COLUMNS, FACTOR_GROUPS, FACTOR_LABELS
from common.factor_expansion_research import OBJECTIVES, ROOT_POSITIONS, ROOT_THRESHOLDS, ExpansionResearchConfig, build_expansion_signals
from common.factor_expansion_research import rebuild_expansion_stitched_cumulatives
from run_factor_expansion_research import run_factor_expansion_static_research
from run_factor_expansion_rolling import run_factor_expansion_rolling
from strategies.dashboard_signal_v1 import DashboardThresholds, DashboardWeights, FactorWindowConfig, build_dashboard_signal, signal_file_for_frequency
from strategies.position_policy import DashboardPositionPolicy


ROOT = Path(__file__).resolve().parent
RESULT_STATE_KEY = "dashboard_backtest_result"
SEARCH_DRAFT_FILE = Path("backtest_outputs") / "search_page_draft.json"
FAVORITES_FILE = Path("backtest_outputs") / "experiment_favorites.json"
DISPLAY_NAMES_FILE = Path("backtest_outputs") / "experiment_display_names.json"
EXPERIMENT_NOTES_FILE = Path("backtest_outputs") / "experiment_notes.json"
FACTOR_RESEARCH_DIR = Path("backtest_outputs") / "因子增加研究"
HOME_BASELINE_FILES = [
    "基线01_重收益_日频.json",
    "基线02_重胜率_日频.json",
    "基线03_激进看多_周频.json",
    "基线04_重胜率_周频.json",
    "基线06_v2上限50_重收益_日频_候选.json",
    "基线07_v2上限50_重胜率_周频_推荐.json",
]

INK = "#18201d"
MUTED = "#66716c"
GRID = "#d9ddd8"
GREEN = "#176b5b"
GREEN_LIGHT = "#8fb8ad"
CORAL = "#bb654f"
GOLD = "#9a7b38"
PAPER = "#f3f2ed"
CHART_COLORS = [GREEN, INK, CORAL, GOLD, GREEN_LIGHT, "#7b8180"]
DIAGNOSTIC_BAD_TYPES = ["做空做反", "低仓少吃", "该空没空"]
DIAGNOSTIC_GOOD_TYPES = ["有效防守", "有效进攻"]
DIAGNOSTIC_TYPE_ORDER = DIAGNOSTIC_BAD_TYPES + DIAGNOSTIC_GOOD_TYPES

WEIGHT_LABELS = {
    "supply_amount": "供给|发行量",
    "supply_ratio": "供给|发行占比",
    "supply_long": "供给|10Y以上发行",
    "fly_penalty": "发行结果|发飞惩罚（数据不足，停用）",
    "bank_demand": "需求|银行需求",
    "spread_gov": "估值|地方债-国债利差",
    "spread_change": "估值|利差一周变化",
    "spread_ncd": "估值|地方债-NCD利差",
    "nonbank_sentiment": "情绪|非银情绪",
}


def _all_strategy_config_paths(*, include_factor_archives: bool = False) -> list[Path]:
    candidates = [
        *sorted((ROOT / CONFIG_DIR).glob("*.json")),
        *sorted((ROOT / CONFIG_DIR / "baselines").glob("*.json")),
        *sorted((ROOT / CONFIG_DIR / "experiments").glob("*.json")),
        *sorted((ROOT / "backtest_outputs" / "experiments").glob("*/config.json")),
    ]
    paths: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        try:
            config = load_strategy_config(path)
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        provenance = config.research_provenance or {}
        is_factor_archive = isinstance(provenance, dict) and provenance.get("研究类型") == "因子增加单一策略归档"
        if is_factor_archive and not include_factor_archives:
            continue
        seen.add(resolved)
        paths.append(path)
    return paths


def _is_archived_experiment_config(path: Path) -> bool:
    return path.name == "config.json" and path.parent.parent.name == "experiments" and path.parent.parent.parent.name == "backtest_outputs"


def _strategy_config_label(path: Path, *, favorites: set[str] | None = None) -> str:
    """Return the shared searchable label for a strategy configuration.

    Archive configs are the only selectable configs that can be collected.  A
    fixed-width prefix keeps the strategy names aligned in every picker.  The
    browser-side picker bridge replaces the saved marker with the same orange
    Material bookmark used by result pages.
    """
    config = load_strategy_config(path)
    marker = "\u00a0\u00a0\u00a0"
    if _is_archived_experiment_config(path):
        is_favorite = path.parent.name in (favorites or set())
        marker = "__local_bond_favorite__" if is_favorite else marker
        label = _run_display_name(path.parent, config.name)
    elif path.parent.name == "experiments":
        label = f"搜索结果 · {path.stem}"
    elif path.parent.name == "baselines":
        label = f"保留基线 · {path.stem}"
    else:
        label = f"基础配置 · {path.stem}"
    return f"{marker}{label}"


def _strategy_config_source_label(path: Path) -> str:
    if _is_archived_experiment_config(path):
        return "历史实验"
    if path.parent.name == "experiments":
        return "搜索结果"
    if path.parent.name == "baselines":
        return "保留基线"
    return "基础配置"


def _fuzzy_config_matches(paths: list[Path], query: str) -> list[Path]:
    normalized_query = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", query.lower())
    if not normalized_query:
        return paths
    ranked: list[tuple[int, int, Path]] = []
    for index, path in enumerate(paths):
        normalized_label = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", _strategy_config_label(path).lower())
        if normalized_query in normalized_label:
            score = 0
        else:
            cursor = iter(normalized_label)
            score = 1 if all(character in cursor for character in normalized_query) else -1
        if score >= 0:
            ranked.append((score, index, path))
    return [path for _, _, path in sorted(ranked)]


def _strategy_config_picker(
    label: str,
    key: str,
    *,
    default: Path | None = None,
    surface: object = st,
    help_text: str | None = None,
    include_factor_archives: bool = False,
) -> Path:
    paths = _all_strategy_config_paths(include_factor_archives=include_factor_archives)
    if not paths:
        surface.error("没有可加载的策略配置。")
        st.stop()
    matched = paths
    default_index = matched.index(default) if default in matched else 0
    current = st.session_state.get(key)
    if current not in matched:
        st.session_state[key] = matched[default_index]
    favorites = _favorite_experiment_ids()
    selected = surface.selectbox(
        label,
        matched,
        index=default_index,
        format_func=lambda path: _strategy_config_label(path, favorites=favorites),
        key=key,
        help=help_text,
    )
    suffix = "因子研究归档会保留其已启用因子集合，并自动使用对应因子引擎复现。" if include_factor_archives else ""
    surface.caption(f"共 {len(paths)} 个兼容配置；打开下拉框后可直接输入策略名称、短编号或来源关键字筛选。{suffix}")
    return selected


def _favorite_experiment_ids() -> set[str]:
    """Read the user's local, display-only experiment collection."""
    path = ROOT / FAVORITES_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    entries = payload.get("experiments", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return set()
    return {Path(str(entry)).name for entry in entries if str(entry).strip()}


def _set_experiment_favorite(experiment_dir: Path, favorite: bool) -> None:
    """Persist a favorite by immutable archive directory name, never by title."""
    archive_name = Path(experiment_dir).name
    if not archive_name:
        return
    favorites = _favorite_experiment_ids()
    if favorite:
        favorites.add(archive_name)
    else:
        favorites.discard(archive_name)
    path = ROOT / FAVORITES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"experiments": sorted(favorites)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _experiment_display_names() -> dict[str, str]:
    """Load user-assigned archive titles without altering frozen results."""
    path = ROOT / DISPLAY_NAMES_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    names = payload.get("names", {}) if isinstance(payload, dict) else {}
    if not isinstance(names, dict):
        return {}
    return {
        Path(str(archive_name)).name: str(title).strip()
        for archive_name, title in names.items()
        if str(archive_name).strip() and str(title).strip()
    }


def _normalise_display_name(value: object) -> str:
    """Keep the run identifier generated by the archive, never by user input."""
    name = re.sub(r"^\s*\[[A-Za-z]\d{3,}\]\s*", "", str(value or "")).strip()
    return re.sub(r"\s+", " ", name)


def _set_experiment_display_name(experiment_dir: Path, title: object, original_name: object) -> None:
    """Persist a reversible display alias keyed by immutable archive directory."""
    archive_name = Path(experiment_dir).name
    if not archive_name:
        return
    names = _experiment_display_names()
    normalized = _normalise_display_name(title)
    original = _normalise_display_name(original_name)
    if not normalized or normalized == original:
        names.pop(archive_name, None)
    else:
        names[archive_name] = normalized
    path = ROOT / DISPLAY_NAMES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "names": dict(sorted(names.items()))}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _experiment_notes() -> dict[str, str]:
    """Load local researcher notes keyed by immutable archive directory."""
    path = ROOT / EXPERIMENT_NOTES_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    notes = payload.get("notes", {}) if isinstance(payload, dict) else {}
    if not isinstance(notes, dict):
        return {}
    return {
        Path(str(archive_name)).name: re.sub(r"\s+", " ", str(note)).strip()
        for archive_name, note in notes.items()
        if str(archive_name).strip() and str(note).strip()
    }


def _set_experiment_note(experiment_dir: Path, note: object) -> None:
    """Persist a display-only archive note without touching frozen results."""
    archive_name = Path(experiment_dir).name
    if not archive_name:
        return
    notes = _experiment_notes()
    cleaned = re.sub(r"\s+", " ", str(note or "")).strip()[:280]
    if cleaned:
        notes[archive_name] = cleaned
    else:
        notes.pop(archive_name, None)
    path = ROOT / EXPERIMENT_NOTES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "notes": dict(sorted(notes.items()))}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _archive_run_id(experiment_dir: Path) -> str | None:
    archive_name = Path(experiment_dir).name
    if "__" not in archive_name:
        return None
    try:
        manifest = json.loads((Path(experiment_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        saved_id = str(manifest.get("short_id", "")).strip()
        saved_prefix = str(manifest.get("short_id_prefix", "")).upper().strip()
        id_prefix = saved_prefix if len(saved_prefix) == 1 and saved_prefix.isalpha() else archive_id_prefix(str(manifest.get("运行来源", "")))
    except (OSError, json.JSONDecodeError):
        saved_id = ""
        id_prefix = "B"
    registered_id = existing_short_archive_id(ROOT, archive_id_category(id_prefix), archive_name)
    if registered_id and saved_id and registered_id != saved_id:
        raise ValueError(f"归档 ID 已被篡改：{archive_name} 保存为 {saved_id}，注册表为 {registered_id}")
    return registered_id or saved_id


@st.dialog("重命名历史实验")
def _render_rename_experiment_dialog(experiment_dir: Path) -> None:
    """Edit an archive's presentation title while retaining all frozen inputs."""
    config_path = Path(experiment_dir) / "config.json"
    try:
        original_name = load_strategy_config(config_path).name
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        st.error("无法读取该历史实验的原始策略名称。")
        return
    archive_id = _archive_run_id(experiment_dir) or "未分配"
    custom_title = _experiment_display_names().get(Path(experiment_dir).name, "")
    default_title = custom_title or _normalise_display_name(original_name)
    st.caption(f"运行 ID：{archive_id}。编号、归档目录、配置和研究溯源不会变更。")
    title = st.text_input(
        "显示名称",
        value=default_title,
        max_chars=80,
        key=f"rename_title_{Path(experiment_dir).name}",
        help="结果页、历史实验、首页和策略选择栏会统一显示 [运行ID] 策略名称。",
    )
    st.caption(f"归档原名：{_normalise_display_name(original_name)}")
    save_col, reset_col, cancel_col = st.columns([1.2, 1.25, 1])
    if save_col.button("保存名称", type="primary", use_container_width=True):
        cleaned = _normalise_display_name(title)
        if not cleaned:
            st.error("显示名称不能为空。")
        else:
            _set_experiment_display_name(experiment_dir, cleaned, original_name)
            st.query_params.pop("rename", None)
            st.rerun()
    if reset_col.button("恢复原名", use_container_width=True):
        _set_experiment_display_name(experiment_dir, original_name, original_name)
        st.query_params.pop("rename", None)
        st.rerun()
    if cancel_col.button("取消", use_container_width=True):
        st.query_params.pop("rename", None)
        st.rerun()


def _render_result_rename_bridge() -> None:
    """Edit display-only archive metadata without navigating the result page."""
    components.html(
        """
        <script>
        (() => {
          const host = window.parent.document;
          // Keep this bridge local to the result header.  The versioned key
          // ensures a hot-reloaded dashboard installs the repaired listener
          // even when an earlier zero-height component is still alive.
          const bridgeKey = '__localBondResultMetadataBridgeV5';
          if (host.defaultView[bridgeKey]) return;
          host.defaultView[bridgeKey] = true;

          const writer = host.createElement('iframe');
          writer.setAttribute('aria-hidden', 'true');
          writer.style.cssText = 'display:none;width:0;height:0;border:0;';
          host.body.appendChild(writer);

          const style = host.createElement('style');
          style.textContent = `
            .local-bond-rename-overlay { display:none; position:fixed; inset:0; z-index:999999; align-items:center; justify-content:center; padding:1rem; background:rgba(24,32,29,.18); }
            .local-bond-rename-overlay.is-open { display:flex; }
            .local-bond-rename-dialog { width:min(25rem,100%); padding:1.1rem; border:1px solid #d9ddd8; border-radius:7px; background:#fbfaf6; color:#18201d; box-shadow:0 14px 35px rgba(24,32,29,.16); font-family:Geist,"Microsoft YaHei",Arial,sans-serif; }
            .local-bond-rename-heading { font-size:.95rem; font-weight:650; }
            .local-bond-rename-dialog p { margin:.35rem 0 .85rem; color:#66716c; font-size:.74rem; line-height:1.55; }
            .local-bond-rename-input { width:100%; padding:.58rem .65rem; border:1px solid #cdd4cf; border-radius:4px; background:#fff; color:#18201d; font:500 .82rem Geist,"Microsoft YaHei",Arial,sans-serif; outline:none; }
            .local-bond-rename-input:focus { border-color:#176b5b; box-shadow:0 0 0 2px rgba(23,107,91,.13); }
            .local-bond-rename-actions { display:flex; justify-content:flex-end; gap:.45rem; margin-top:.85rem; }
            .local-bond-rename-actions button { min-height:2rem; padding:0 .7rem; border:1px solid #d5d9d5; border-radius:4px; background:#f7f7f7; color:#59635f; cursor:pointer; font:600 .76rem Geist,"Microsoft YaHei",Arial,sans-serif; }
             .local-bond-rename-actions button.primary { border-color:#176b5b; background:#176b5b; color:#fff; }`;
          host.head.appendChild(style);

          const overlay = host.createElement('div');
          overlay.className = 'local-bond-rename-overlay';
          overlay.innerHTML = `
            <section class="local-bond-rename-dialog" role="dialog" aria-modal="true" aria-label="重命名历史实验">
              <div class="local-bond-rename-heading">重命名历史实验</div>
              <p>仅修改显示名称。运行 ID、归档目录、参数和研究溯源不会变更。</p>
              <input class="local-bond-rename-input" maxlength="80" aria-label="显示名称">
              <div class="local-bond-rename-actions">
                <button type="button" data-action="cancel">取消</button>
                <button type="button" class="primary" data-action="save">保存名称</button>
              </div>
            </section>`;
          host.body.appendChild(overlay);
          const input = overlay.querySelector('input');
          let activeButton = null;
          const close = () => { overlay.classList.remove('is-open'); activeButton = null; };
          const save = () => {
            const title = input.value.trim();
            if (!activeButton || !title) { input.focus(); return; }
            const archive = activeButton.dataset.archive;
            const prefix = (activeButton.closest('.archive-breadcrumb').querySelector('strong').textContent.match(/^\\[[A-Za-z]\\d{3,}\\]\\s*/) || [''])[0];
            activeButton.closest('.archive-breadcrumb').querySelector('strong').textContent = prefix + title;
            activeButton.dataset.title = title;
            writer.src = '/history?rename_save=' + encodeURIComponent(archive) + '&rename_title=' + encodeURIComponent(title) + '&_=' + Date.now();
            close();
          };
          overlay.addEventListener('click', event => {
            if (event.target === overlay || event.target.closest('[data-action="cancel"]')) close();
            if (event.target.closest('[data-action="save"]')) save();
          });
          input.addEventListener('keydown', event => {
            if (event.key === 'Enter') { event.preventDefault(); save(); }
            if (event.key === 'Escape') close();
          });
          const handleRenameClick = event => {
            // A click can originate on the button's icon or on its padded
            // edge.  Resolve from an Element defensively so the full rounded
            // button, rather than only the icon's visual area, is actionable.
            // `event.target` belongs to the parent document, whereas this
            // Streamlit component executes in an iframe. `instanceof Element`
            // therefore fails across realms even for a real button. Duck-type
            // `closest` instead so the icon, padding and button all work.
            const source = event.target;
            const trigger = source && typeof source.closest === 'function'
              ? source.closest('.archive-rename')
              : null;
            if (!trigger) return;
            event.preventDefault();
            event.stopPropagation();
            activeButton = trigger;
            input.value = trigger.dataset.title || '';
            overlay.classList.add('is-open');
            window.setTimeout(() => { input.focus(); input.select(); }, 0);
          };
          host.addEventListener('click', handleRenameClick, true);

          const noteOverlay = host.createElement('div');
          noteOverlay.className = 'local-bond-rename-overlay';
          noteOverlay.innerHTML = `
            <section class="local-bond-rename-dialog" role="dialog" aria-modal="true" aria-label="编辑备注">
              <div class="local-bond-rename-heading">策略备注</div>
              <p>备注只用于研究工作台展示，不修改运行 ID、策略参数或冻结归档。</p>
              <input class="local-bond-rename-input" maxlength="280" aria-label="策略备注">
              <div class="local-bond-rename-actions">
                <button type="button" data-action="cancel">取消</button>
                <button type="button" class="primary" data-action="save">保存备注</button>
              </div>
            </section>`;
          host.body.appendChild(noteOverlay);
          const noteInput = noteOverlay.querySelector('input');
          let activeNoteButton = null;
          const closeNote = () => { noteOverlay.classList.remove('is-open'); activeNoteButton = null; };
          const saveNote = () => {
            if (!activeNoteButton) return;
            const note = noteInput.value.trim();
            activeNoteButton.dataset.note = note;
            activeNoteButton.classList.toggle('has-note', Boolean(note));
            activeNoteButton.setAttribute('title', note || '添加备注');
            activeNoteButton.setAttribute('aria-label', note ? '编辑备注：' + note : '添加备注');
            writer.src = '/history?note_save=' + encodeURIComponent(activeNoteButton.dataset.archive) + '&note_text=' + encodeURIComponent(note) + '&_=' + Date.now();
            closeNote();
          };
          noteOverlay.addEventListener('click', event => {
            if (event.target === noteOverlay || event.target.closest('[data-action="cancel"]')) closeNote();
            if (event.target.closest('[data-action="save"]')) saveNote();
          });
          noteInput.addEventListener('keydown', event => {
            if (event.key === 'Enter') { event.preventDefault(); saveNote(); }
            if (event.key === 'Escape') closeNote();
          });
          const handleNoteClick = event => {
            const source = event.target;
            const trigger = source && typeof source.closest === 'function'
              ? source.closest('.archive-note')
              : null;
            if (!trigger) return;
            event.preventDefault();
            event.stopPropagation();
            activeNoteButton = trigger;
            noteInput.value = trigger.dataset.note || '';
            noteOverlay.classList.add('is-open');
            window.setTimeout(() => {{ noteInput.focus(); noteInput.select(); }}, 0);
          };
          host.addEventListener('click', handleNoteClick, true);
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def _render_favorite_button(experiment_dir: Path, *, key: str, container: object = st) -> None:
    """Show the common orange bookmark without rerunning the result page."""
    archive_name = Path(experiment_dir).name
    if not archive_name or archive_name.startswith("本次结果"):
        return

    # A fragment re-runs only this icon after a click.  The charts and result
    # workspace therefore remain in place instead of visibly rebuilding.
    @st.fragment
    def render_control() -> None:
        is_favorite = archive_name in _favorite_experiment_ids()
        state_key = f"{key}_bookmark"
        safe_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", key)
        state_class = "saved" if is_favorite else "empty"
        wrapper_key = f"favorite_control_{state_class}_{safe_key}"
        with container.container(key=wrapper_key):
            if container.button(
                "",
                icon=":material/bookmark:" if is_favorite else ":material/bookmark_border:",
                key=state_key,
                help="已收藏，点击取消收藏" if is_favorite else "收藏",
                type="tertiary",
                width="content",
            ):
                _set_experiment_favorite(experiment_dir, not is_favorite)
                st.rerun(scope="fragment")

    render_control()


def main() -> None:
    st.set_page_config(page_title="10Y地方债策略工作台", layout="wide")
    _inject_styles()
    _render_strategy_picker_favorite_bridge()
    st.logo(
        ROOT / "assets" / "local_bond_logo.svg",
        icon_image=ROOT / "assets" / "local_bond_icon.svg",
        size="large",
    )
    home_page = st.Page(_render_home_page, title="首页", url_path="home", default=True)
    history_page = st.Page(_render_history_route, title="历史实验", url_path="history")
    search_page = st.Page(_render_search_page, title="搜索研究", url_path="search")
    rolling_page = st.Page(_render_rolling_research_page, title="滚动定参", url_path="rolling")
    factor_research_page = st.Page(_render_factor_research_page, title="因子研究", url_path="factor-research")
    st.session_state["history_navigation_page"] = history_page
    st.session_state["factor_research_navigation_page"] = factor_research_page
    selected_page = st.navigation([home_page, history_page, search_page, rolling_page, factor_research_page], position="top")
    selected_page.run()


def _render_home_page() -> None:
    if st.query_params.get("home") == "1":
        st.session_state.pop(RESULT_STATE_KEY, None)
        st.query_params.clear()
        st.rerun()

    config_path = _select_config()
    base_config = load_strategy_config(config_path)
    # Baseline files use their filename as the editable display name. Archived
    # experiment configs must retain the archived strategy name (their filename
    # is always just ``config.json``).
    display_name = base_config.name if _is_archived_experiment_config(config_path) else config_path.stem
    base_config = DashboardStrategyConfig(
        name=display_name,
        weights=base_config.weights,
        thresholds=base_config.thresholds,
        positions=base_config.positions,
        objective=base_config.objective,
        factor_windows=base_config.factor_windows,
        backtest_start=base_config.backtest_start,
        backtest_end=base_config.backtest_end,
        benchmark_id=base_config.benchmark_id,
        signal_frequency=base_config.signal_frequency,
        research_provenance=base_config.research_provenance,
        source_config_path=base_config.source_config_path or str(config_path.resolve()),
    )
    config, run_clicked, save_clicked = _sidebar_config(base_config, key_prefix=f"main_{config_path.stem}")

    if save_clicked:
        _save_config(config)
    if run_clicked:
        experiment_dir = _run_backtest(config)
        if experiment_dir is not None:
            _switch_to_history_experiment(experiment_dir)

    result = st.session_state.get(RESULT_STATE_KEY)
    if result is None:
        _render_launch_state(config)
        _render_home_research_snapshot()
        _render_launch_details(config)
        return

    if len(result) == 6:
        daily, signals, strategy_metrics, benchmark_metrics, diagnostics, result_name = result
        experiment_dir = Path("本次结果生成于自动归档功能启用前")
        result_config = config
    elif len(result) == 7:
        daily, signals, strategy_metrics, benchmark_metrics, diagnostics, result_name, experiment_dir = result
        result_config = config
    else:
        daily, signals, strategy_metrics, benchmark_metrics, diagnostics, result_name, experiment_dir, result_config = result
    _render_summary(strategy_metrics, benchmark_metrics, signals)
    _render_workspace(
        daily,
        signals,
        strategy_metrics,
        benchmark_metrics,
        diagnostics,
        result_name,
        result_config,
        experiment_dir,
    )


def _render_strategy_picker_favorite_bridge() -> None:
    """Render favorite markers inside every native strategy search result.

    Streamlit selectbox labels are plain text, so its standard formatter cannot
    render an orange Material icon.  This tiny, local-only bridge decorates the
    marker generated by ``_strategy_config_label`` in the parent document.  It
    deliberately observes future popovers too, which covers searchable lists
    in the sidebar and in the research pages without duplicating controls.
    """
    components.html(
        """
        <script>
        (() => {
          const host = window.parent.document;
          if (host.defaultView.__localBondPickerFavoriteBridge) return;
          host.defaultView.__localBondPickerFavoriteBridge = true;

          const style = host.createElement('style');
          style.textContent = `
            .local-bond-picker-label { display:inline-flex; align-items:center; min-width:0; }
            .local-bond-picker-label .local-bond-picker-bookmark-slot { display:inline-flex; flex:0 0 1.1rem; width:1.1rem; height:1.1rem; align-items:center; justify-content:center; margin-right:.12rem; }
            .local-bond-picker-label .local-bond-picker-bookmark-slot.is-saved { color:#bb654f; }
            .local-bond-picker-label .local-bond-picker-bookmark-slot .material-symbols-rounded { font-size:1rem; font-variation-settings:'FILL' 1,'wght' 400,'GRAD' 0,'opsz' 20; }
          `;
          host.head.appendChild(style);

          const marker = '__local_bond_favorite__';
          const decorate = () => {
            host.querySelectorAll('div, span, p').forEach(node => {
              if (node.childElementCount || node.dataset.localBondPickerDecorated === '1') return;
              const text = node.textContent || '';
              if (!text.includes(marker)) return;
              node.dataset.localBondPickerDecorated = '1';
              const name = text.replace(marker, '').replace(/^\\s+/, '');
              node.textContent = '';
              node.classList.add('local-bond-picker-label');
              const slot = host.createElement('span');
              slot.className = 'local-bond-picker-bookmark-slot is-saved';
              slot.innerHTML = '<span class="material-symbols-rounded" aria-hidden="true">bookmark</span>';
              const title = host.createElement('span');
              title.textContent = name;
              node.append(slot, title);
            });
          };
          decorate();
          new MutationObserver(decorate).observe(host.body, {childList:true, subtree:true, characterData:true});
        })();
        </script>
        """,
        height=0,
    )


def _render_history_route() -> None:
    rename_save_target = str(st.query_params.get("rename_save", "")).strip()
    if rename_save_target:
        experiment_root = (ROOT / "backtest_outputs" / "experiments").resolve()
        experiment_dir = (experiment_root / Path(rename_save_target).name).resolve()
        rename_title = str(st.query_params.get("rename_title", "")).strip()
        if experiment_dir.parent == experiment_root and (experiment_dir / "run_manifest.json").exists():
            try:
                original_name = load_strategy_config(experiment_dir / "config.json").name
                if _normalise_display_name(rename_title):
                    _set_experiment_display_name(experiment_dir, rename_title, original_name)
            except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
                pass
        # This route is loaded only by a hidden iframe after a local UI
        # confirmation.  Stop here so saving a display alias never renders a
        # second dashboard or refreshes the visible page.
        st.stop()
    note_save_target = str(st.query_params.get("note_save", "")).strip()
    if note_save_target:
        experiment_root = (ROOT / "backtest_outputs" / "experiments").resolve()
        experiment_dir = (experiment_root / Path(note_save_target).name).resolve()
        note_text = str(st.query_params.get("note_text", ""))
        if experiment_dir.parent == experiment_root and (experiment_dir / "run_manifest.json").exists():
            _set_experiment_note(experiment_dir, note_text)
        # The note writer is an invisible iframe inside the history table.
        # Never render a second result page or refresh the visible table.
        st.stop()
    factor_study = str(st.query_params.get("factor_study", "")).strip()
    if factor_study:
        _render_factor_strategy_history(factor_study)
        return
    experiment_id = str(
        st.query_params.get("experiment", "")
        or st.session_state.pop("pending_history_experiment", "")
    ).strip()
    if experiment_id:
        _render_historical_result_page(experiment_id)
    else:
        _render_history_page()
    rename_target = str(st.query_params.get("rename", "")).strip()
    if rename_target:
        experiment_root = (ROOT / "backtest_outputs" / "experiments").resolve()
        experiment_dir = (experiment_root / Path(rename_target).name).resolve()
        if experiment_dir.parent == experiment_root and (experiment_dir / "run_manifest.json").exists():
            _render_rename_experiment_dialog(experiment_dir)
        else:
            st.warning("未找到要重命名的历史实验。")
            st.query_params.pop("rename", None)


def _switch_to_history_experiment(experiment_dir: Path) -> None:
    st.switch_page(
        st.session_state["history_navigation_page"],
        query_params={"experiment": Path(experiment_dir).name},
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0..1,0&display=swap');

        :root {
            --ink: #18201d;
            --muted: #66716c;
            --paper: #f3f2ed;
            --surface: #fbfaf6;
            --line: #d9ddd8;
            --green: #176b5b;
            --coral: #bb654f;
            --gold: #9a7b38;
        }
        html { scroll-behavior: smooth; }
        body, [class*="css"] { font-family: "Geist", "Microsoft YaHei", sans-serif; }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at 82% 0%, rgba(23, 107, 91, .08), transparent 28rem),
                radial-gradient(circle at 12% 42%, rgba(187, 101, 79, .055), transparent 24rem),
                var(--paper);
        }
        /* Every bookmark has one orange outlined shape. A filled glyph is the
           only saved-state distinction, as in normal collection controls. */
        [class*="st-key-favorite_"] [data-testid="stIconMaterial"] {
            color: var(--coral) !important;
            font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 22 !important;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .18;
            z-index: 0;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.08'/%3E%3C/svg%3E");
        }
        [data-testid="stHeader"] {
            min-height: 3.75rem;
            background: rgba(243, 242, 237, .88);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid rgba(217,221,216,.82);
        }
        [data-testid="stMainBlockContainer"] { max-width: 92rem; padding-top: 2.1rem; padding-bottom: 5rem; animation: pageReveal .45s ease both; }
        main [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { animation: cascadeReveal .58s cubic-bezier(.22,.8,.25,1) both; }
        main [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(2) { animation-delay: .04s; }
        main [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(3) { animation-delay: .08s; }
        main [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(n+4) { animation-delay: .12s; }
        [data-testid="stSidebar"] { background: #e9e9e3; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--muted) !important; opacity: 1; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--ink); letter-spacing: 0; }
        [data-testid="stSidebar"] details { background: rgba(251,250,246,.62); border: 1px solid var(--line); border-radius: 6px; }
        [data-testid="stSidebar"] details:hover { border-color: #abb5af; }
        [data-testid="stSidebar"] .stButton button {
            min-height: 2.7rem;
            border-radius: 4px;
            font-weight: 600;
            border-color: #b8beb9;
            transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
        }
        [data-testid="stSidebar"] .stButton button:hover { transform: translateY(-1px); box-shadow: 0 8px 22px rgba(24,32,29,.10); }
        [data-testid="stSidebar"] .stButton button:active { transform: translateY(1px) scale(.99); }
        [data-testid="stSidebar"] .stButton button[kind="primary"] { background: var(--green); color: #ffffff !important; border-color: var(--green); }
        [data-testid="stSidebar"] .stButton button[kind="primary"] p,
        [data-testid="stSidebar"] .stButton button[kind="primary"] span { color: #ffffff !important; opacity: 1 !important; }
        [data-testid="stSidebar"] .stButton button:focus-visible { outline: 3px solid rgba(23,107,91,.25); outline-offset: 2px; }
        [data-testid="stSidebar"] [class*="st-key-retry_rolling_queue_"] button,
        [data-testid="stSidebar"] [class*="st-key-remove_rolling_queue_"] button,
        [data-testid="stSidebar"] [class*="st-key-cancel_rolling_queue_"] button {
            min-height:1.25rem !important; min-width:1.25rem !important; padding:0 !important; border:0 !important;
            background:transparent !important; box-shadow:none !important; color:var(--muted) !important; font-size:.82rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-retry_rolling_queue_"] button:hover { color:var(--green) !important; transform:none !important; box-shadow:none !important; }
        [data-testid="stSidebar"] [class*="st-key-remove_rolling_queue_"] button:hover { color:#a54843 !important; transform:none !important; box-shadow:none !important; }
        [data-testid="stSidebar"] [class*="st-key-cancel_rolling_queue_"] button:hover { color:#a54843 !important; transform:none !important; box-shadow:none !important; }
        /* Compact queue: green is live work, amber is waiting, and a lighter
           red is reserved for a failed task that needs attention. */
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_"] {
            position: relative; isolation: isolate; overflow: hidden; margin: .28rem 0;
            padding: .46rem .52rem .42rem; border: 1px solid var(--line); border-radius: 4px;
            background: rgba(251,250,246,.58);
        }
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_"] p { margin: 0 !important; }
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_"] [data-testid="stCaptionContainer"] { margin-top: .12rem; font-size: .69rem; line-height: 1.35; }
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_active_"] {
            color: var(--green); border-color: rgba(23,107,91,.42); background: rgba(23,107,91,.09);
        }
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_active_"] p,
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_active_"] [data-testid="stCaptionContainer"] { color: var(--green) !important; }
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_active_"]::before {
            content: ""; position: absolute; z-index: -1; inset: 0 auto 0 0; width: 48%;
            background: linear-gradient(90deg, transparent, rgba(23,107,91,.18), transparent);
            transform: translateX(-140%); animation: rollingQueuePulse 2.4s ease-in-out infinite;
            will-change: transform;
        }
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_waiting_"] {
            color: var(--gold); border-color: rgba(154,123,56,.44); background: rgba(154,123,56,.075);
        }
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_waiting_"] p,
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_waiting_"] [data-testid="stCaptionContainer"] { color: var(--gold) !important; }
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_failed_"] {
            color: #b86763; border-color: rgba(184,103,99,.42); background: rgba(184,103,99,.07);
        }
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_failed_"] p,
        [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_failed_"] [data-testid="stCaptionContainer"] { color: #a9605c !important; }
        .rolling-queue-heading { margin: .7rem 0 .25rem; font-size: .74rem; font-weight: 700; letter-spacing: .015em; }
        .rolling-queue-heading.is-active { color: var(--green); }
        .rolling-queue-heading.is-waiting { color: var(--gold); }
        .rolling-queue-heading.is-failed { color: #b86763; }
        @keyframes rollingQueuePulse { 0%, 100% { transform: translateX(-140%); opacity: .18; } 48% { opacity: .75; } 100% { transform: translateX(315%); opacity: .18; } }
        @media (prefers-reduced-motion: reduce) { [data-testid="stSidebar"] [class*="st-key-rolling_queue_task_active_"]::before { animation: none; opacity: .22; transform: translateX(60%); } }
        .strategy-mode-note { margin: .9rem 0 .15rem; padding: .78rem .85rem .82rem; border-left: 3px solid #8a9690; background: rgba(24,32,29,.045); color: var(--ink); }
        .strategy-mode-note strong { display: block; font-size: .82rem; letter-spacing: .015em; color: #52645c; margin-bottom: .22rem; }
        .strategy-mode-note span { display: block; color: #52645c; font-size: .72rem; line-height: 1.55; }
        .rolling-strategy-note { border-left-color: var(--green); background: rgba(23,107,91,.075); }
        .rolling-strategy-note strong { color: var(--green); }
        [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted); }
        [data-baseweb="slider"] [role="slider"] { background: var(--green); }

        .stButton > button {
            min-height: 2.65rem;
            color: var(--ink) !important;
            background: #fbfaf6 !important;
            border: 1px solid #b8beb9 !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease, background .2s ease !important;
        }
        .stButton > button:hover {
            color: var(--green) !important;
            border-color: var(--green) !important;
            background: #f7f8f3 !important;
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(23,107,91,.09);
        }
        .stButton > button:active { transform: translateY(1px) scale(.99); }
        .stButton > button[kind="primary"] {
            color: #ffffff !important;
            background: var(--green) !important;
            border-color: var(--green) !important;
        }
        .stButton > button[kind="tertiary"] {
            min-height: 1.75rem !important;
            padding: .05rem .15rem !important;
            color: var(--coral) !important;
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            font-size: 1.18rem !important;
            line-height: 1 !important;
        }
        .stButton > button[kind="tertiary"]:hover { color: #a94f3d !important; background: transparent !important; border: 0 !important; box-shadow: none !important; }
        [class*="st-key-favorite_"] button {
            min-width: 1.75rem !important;
            min-height: 1.75rem !important;
            padding: .12rem !important;
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
        }
        [class*="st-key-favorite_control_saved_"] [data-testid="stIconMaterial"] {
            font-variation-settings: "FILL" 1, "wght" 400, "GRAD" 0, "opsz" 22 !important;
        }
        [class*="st-key-favorite_"] button p {
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
            font-size: 0 !important;
        }
        [class*="st-key-favorite_"] button svg {
            width: 1.22rem !important;
            height: 1.22rem !important;
            color: var(--coral) !important;
        }
        [class*="st-key-favorite_"] button [data-testid="stIconMaterial"] { color: var(--coral) !important; }
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span { color: #ffffff !important; opacity: 1 !important; }
        .stButton > button[kind="primary"]:hover { background: #11594c !important; color: #ffffff !important; }
        .stButton > button:focus-visible { outline: 3px solid rgba(23,107,91,.24) !important; outline-offset: 2px; }
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="collapsedControl"] button {
            color: var(--green) !important;
            background: #fbfaf6 !important;
            border: 1px solid #aeb8b2 !important;
            box-shadow: 0 5px 16px rgba(24,32,29,.10) !important;
        }

        .app-header-integration {
            position: absolute;
            inset: 0 8.5rem 0 0;
            z-index: 5;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: minmax(15rem, 1fr) auto minmax(15rem, 1fr);
            align-items: center;
            gap: 2rem;
            pointer-events: none;
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: .72rem;
            min-width: 0;
            color: var(--ink) !important;
            text-decoration: none !important;
            border-bottom: 0 !important;
            transition: color .2s ease, transform .2s ease;
            pointer-events: auto;
        }
        .brand-lockup *,
        .brand-lockup:link,
        .brand-lockup:visited,
        .brand-lockup:hover,
        .brand-lockup:active { color: inherit; text-decoration: none !important; border-bottom: 0 !important; }
        .brand-lockup:hover { color: var(--coral) !important; transform: translateY(-1px); }
        .brand-lockup:focus-visible { outline: 3px solid rgba(187,101,79,.22); outline-offset: 5px; }
        .brand-mark { width: .72rem; height: .72rem; background: var(--green); border-radius: 2px; box-shadow: .34rem .34rem 0 var(--coral); flex: 0 0 auto; }
        .brand-copy { display: flex; flex-direction: column; justify-content: center; gap: .08rem; line-height: 1.05; }
        .brand-name { font-weight: 700; font-size: .96rem; }
        .brand-sub { color: var(--muted); font-size: .62rem; margin-left: 0; font-family: "IBM Plex Mono", monospace; }
        .top-navigation {
            display: flex;
            align-items: center;
            gap: 1.25rem;
            height: 2.25rem;
            pointer-events: auto;
        }
        .app-header-spacer { min-width: 15rem; }
        .top-nav-link,
        .top-nav-link:link,
        .top-nav-link:visited {
            color: var(--muted) !important;
            text-decoration: none !important;
            border-bottom: 2px solid transparent !important;
            padding: .45rem 0 .38rem;
            font-size: .84rem;
            font-weight: 600;
            transition: color .2s ease, border-color .2s ease;
        }
        .top-nav-link:hover { color: var(--green) !important; }
        .top-nav-link.active { color: var(--ink) !important; border-bottom-color: var(--coral) !important; }

        .hero-shell {
            display: grid;
            grid-template-columns: minmax(0, 8fr) minmax(16rem, 4fr);
            gap: clamp(2rem, 5vw, 6rem);
            align-items: end;
            padding: clamp(3rem, 7vw, 7rem) 0 clamp(2.8rem, 5vw, 5rem);
            animation: rise .55s .06s ease both;
        }
        .eyebrow { margin: 0 0 1rem; color: var(--green); font-size: .82rem; font-weight: 600; }
        .hero-title { max-width: 78rem; margin: 0; font-size: clamp(2.8rem, 5vw, 5.4rem); line-height: .98; letter-spacing: 0; font-weight: 650; text-wrap: balance; }
        .hero-copy { max-width: 34rem; margin: 1.4rem 0 0; color: var(--muted); font-size: 1rem; line-height: 1.7; text-wrap: pretty; }
        .hero-aside { border-left: 1px solid var(--line); padding-left: 1.5rem; }
        .aside-label { color: var(--muted); font-size: .78rem; margin-bottom: .55rem; }
        .aside-value { font-family: "IBM Plex Mono", "Microsoft YaHei", sans-serif; font-size: clamp(1.8rem, 3vw, 3.1rem); font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1; }
        .aside-note { color: var(--muted); font-size: .82rem; margin-top: .75rem; line-height: 1.5; }
        .st-key-historical_hero { padding: clamp(3rem, 7vw, 7rem) 0 clamp(2.8rem, 5vw, 5rem); animation: rise .55s .06s ease both; }
        .st-key-historical_hero [data-testid="stHorizontalBlock"] { align-items: end; }
        .historical-hero-main { padding-right: clamp(1rem, 3vw, 3rem); }
        .historical-hero-side { min-height: 9.5rem; border-left: 1px solid var(--line); padding-left: 1.35rem; display: flex; flex-direction: column; justify-content: flex-end; }
        .st-key-period_compact { min-height: 9.5rem; border-left: 1px solid var(--line); padding-left: 1.35rem; display: flex; flex-direction: column; justify-content: flex-end; }
        .st-key-period_compact [data-testid="stSelectbox"] { margin-bottom: .35rem; }
        .st-key-period_compact [data-testid="stWidgetLabel"] p { color: var(--muted); font-size: .76rem; }
        .period-current { display: flex; flex-direction: column; gap: .3rem; padding-bottom: .1rem; }
        .period-current strong { color: var(--ink); font-size: 1rem; font-weight: 600; }
        .period-current span { color: var(--muted); font-size: .72rem; }
        .factor-toggle-intro { color: var(--muted); font-size: .77rem; line-height: 1.55; margin: -.15rem 0 .65rem; }
        .period-half-shortcuts { margin-top: .42rem; }
        .period-half-title { color: var(--muted); font-size: .76rem; line-height: 1.4; margin: .15rem 0 .2rem; }
        .period-half-title.disabled { color: #a4aaa5; }
        .period-half-shortcuts .stButton button { min-height: 2.08rem; padding: .22rem .55rem; font-size: .76rem; font-weight: 500 !important; }

        .metric-grid { display: grid; grid-template-columns: repeat(12, 1fr); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); margin: 0 0 2.3rem; animation: rise .55s .12s ease both; }
        .metric-cell { grid-column: span 3; padding: 1.35rem 1.2rem 1.5rem 0; min-width: 0; }
        .metric-cell + .metric-cell { border-left: 1px solid var(--line); padding-left: 1.2rem; }
        .metric-cell:nth-child(4n + 1) { border-left: 0; padding-left: 0; }
        .metric-cell:nth-child(n + 5) { border-top: 1px solid var(--line); }
        .metric-label { color: var(--muted); font-size: .78rem; margin-bottom: .65rem; }
        .metric-value { font-family: "IBM Plex Mono", "Microsoft YaHei", sans-serif; font-size: clamp(1.55rem, 2.3vw, 2.35rem); line-height: 1; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
        .metric-detail { color: var(--muted); font-size: .76rem; margin-top: .65rem; }
        .metric-cell.compact-value .metric-value { font-size: clamp(1rem, 1.45vw, 1.45rem); line-height: 1.25; white-space: normal; }
        .positive { color: var(--coral); }
        .negative { color: var(--green); }

        .launch-grid { display: grid; grid-template-columns: minmax(0, 8fr) minmax(17rem, 4fr); gap: 1rem; margin-top: 1rem; }
        .launch-panel { background: rgba(251,250,246,.72); border: 1px solid var(--line); border-radius: 6px; padding: clamp(1.4rem, 3vw, 2.4rem); }
        .launch-panel h3 { margin: 0 0 1rem; font-size: 1.25rem; letter-spacing: 0; }
        .logic-flow { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); }
        .logic-step { background: #f8f7f2; padding: 1.2rem; min-height: 8.5rem; }
        .logic-step strong { display: block; margin-bottom: .65rem; font-size: .92rem; }
        .logic-step span { color: var(--muted); font-size: .82rem; line-height: 1.55; }
        .config-line { display: flex; justify-content: space-between; gap: 1rem; padding: .8rem 0; border-bottom: 1px solid var(--line); font-size: .84rem; }
        .config-line:last-child { border-bottom: 0; }
        .config-line span { color: var(--muted); }
        .config-line strong { font-family: "IBM Plex Mono", monospace; text-align: right; font-variant-numeric: tabular-nums; }

        .section-head { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin: 1.6rem 0 1rem; }
        .section-head h2 { margin: 0; font-size: clamp(1.35rem, 2vw, 1.85rem); letter-spacing: 0; }
        .section-head p { color: var(--muted); margin: 0; font-size: .82rem; }
        .research-type-head { margin-top: 1.1rem; padding: .9rem 0 .85rem; border-bottom: 1px solid var(--line); }
        .research-type-head h2 { margin: 0 0 .35rem; font-size: 1.38rem; font-weight: 720; }
        .provenance-overview { container-type: inline-size; padding: .6rem 0 1rem; letter-spacing: 0; }
        .provenance-context { display: flex; flex-wrap: wrap; gap: .5rem 1.8rem; margin: 0 0 1.8rem; color: var(--muted); font-size: .8rem; }
        .provenance-context strong { margin-left: .55rem; color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
        .provenance-chain { display: grid; grid-template-columns: 1fr; list-style: none; margin: 0 !important; padding: 0 !important; }
        .provenance-chain > li { position: relative; padding: 0 0 1.5rem 1.3rem; border-top: 0; border-left: 2px solid var(--line); }
        .provenance-chain > li::before { content: ''; position: absolute; top: .35rem; left: -5px; width: 8px; height: 8px; border-radius: 50%; background: #8a9690; }
        .provenance-chain > li:not(:last-child)::after { content: ''; position: absolute; bottom: .45rem; left: -4px; width: 6px; height: 6px; border-bottom: 1px solid #8a9690; border-right: 1px solid #8a9690; transform: rotate(45deg); }
        .provenance-chain > li:last-child { border-left-color: var(--ink); }
        .provenance-chain > li:last-child::before { background: var(--ink); }
        .provenance-role { display: block; color: var(--muted); font-size: .72rem; margin-bottom: .6rem; font-variant-numeric: tabular-nums; }
        .provenance-title { color: var(--ink); font-size: 1rem; font-weight: 650; line-height: 1.5; overflow-wrap: anywhere; }
        .provenance-chain > li:last-child .provenance-title { font-weight: 750; }
        .provenance-subtitle { color: var(--muted); font-size: .77rem; line-height: 1.65; margin-top: .35rem; overflow-wrap: anywhere; white-space: pre-line; }
        .provenance-reference-original, .provenance-reference-path { color: var(--muted); font-size: .76rem; line-height: 1.55; overflow-wrap: anywhere; }
        .provenance-strategy-link { color: var(--green) !important; text-decoration: none !important; font-weight: 700; }
        .provenance-strategy-link:hover { text-decoration: underline !important; }
        .provenance-chain.is-long { display: grid; grid-template-columns: 1fr; }
        @container (max-width: 720px) {
            .provenance-chain { display: grid; grid-template-columns: 1fr; }
            .provenance-role { margin-bottom: .2rem; }
        }
        .st-key-combined_search_band {
            margin: 1.2rem 0 1.8rem;
            padding: 1.1rem 0 1.2rem;
            border-top: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
        }
        .st-key-combined_search_band [data-testid="stHorizontalBlock"] { align-items: center; }
        .st-key-combined_search_band h3 { margin: 0 0 .35rem; font-size: 1.18rem; }
        .st-key-combined_search_band [data-testid="stCaptionContainer"] { color: var(--muted); }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.7rem; border-bottom: 1px solid var(--line); }
        [data-testid="stTabs"] button[role="tab"] { padding: .8rem 0; color: var(--muted); font-weight: 500; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: var(--ink); }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--green); }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }
        [data-testid="stDataEditor"] a[href*="bookmark_icon=bookmark"] {
            color: var(--coral) !important;
            text-decoration: none !important;
            font-family: "Material Symbols Rounded" !important;
            font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 22;
            font-size: 1.25rem;
            line-height: 1;
        }
        [data-testid="stDataEditor"] a[href*="bookmark_icon=bookmark"]:hover { color: #a94f3d !important; }
        iframe { background: transparent; border-radius: 5px; }
        .history-load-status-parent {
            margin: .35rem 0 .85rem;
            color: var(--muted);
            font: 500 .72rem/1.4 "IBM Plex Mono", "Microsoft YaHei", monospace;
            font-variant-numeric: tabular-nums;
            text-align: right;
        }
        .signal-asof-banner {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: .28rem .75rem;
            margin: .1rem 0 .75rem;
            padding: .5rem .7rem;
            border-left: 3px solid var(--green);
            background: rgba(143,184,173,.12);
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.55;
        }
        .signal-asof-banner strong { color: var(--ink); font-size: .83rem; }
        .stAlert { border-radius: 5px; }
        hr { border-color: var(--line); }

        .research-table-wrap {
            width: 100%;
            overflow: auto;
            border: 1px solid var(--line);
            border-radius: 5px;
            background: rgba(251,250,246,.82);
        }
        .research-table { width: 100%; border-collapse: collapse; color: var(--ink); font-size: .86rem; }
        .research-table th {
            padding: .78rem .9rem;
            color: #4f5b56;
            background: #e7ebe5;
            border-bottom: 1px solid #cdd4cf;
            font-size: .76rem;
            font-weight: 600;
            text-align: left;
            white-space: nowrap;
        }
        .research-table td {
            padding: .78rem .9rem;
            background: rgba(251,250,246,.64);
            border-bottom: 1px solid #dde1dc;
            vertical-align: middle;
        }
        .research-table tbody tr:last-child td { border-bottom: 0; }
        .research-table tbody tr:hover td { background: #f0f3ed; }
        .research-table a.table-link { position: relative; z-index: 1000002; color: var(--green) !important; font-weight: 650; text-decoration: none !important; border-bottom: 1px solid rgba(23,107,91,.28); }
        .research-table a.table-link:hover { color: var(--coral) !important; border-bottom-color: var(--coral); }
        .factor-detail-actions { display: flex; flex-wrap: wrap; gap: .4rem; min-width: 13rem; }
        .factor-detail-link { display: inline-flex; align-items: center; gap: .45rem; padding: .4rem .55rem; color: var(--ink) !important; background: #f8f8f4; border: 1px solid #cbd3cd; border-radius: 3px; font-size: .76rem; font-weight: 600; text-decoration: none !important; transition: color .16s ease, border-color .16s ease, background .16s ease, transform .16s ease; }
        .factor-detail-link::after { content: '查看'; color: var(--muted); font-size: .68rem; font-family: "IBM Plex Mono", monospace; }
        .factor-detail-link:hover { color: var(--green) !important; border-color: var(--green); background: #f1f5f0; transform: translateY(-1px); }
        .research-table tr.factor-inactive td { color: #8a918d; background: #f2f4f1; }
        .research-table tr.factor-inactive:hover td { background: #edf0ed; }
        .research-table tr.factor-inactive td:first-child { box-shadow: inset 3px 0 0 #c2c8c4; }
        .research-table tr.factor-inactive .qualitative { filter: grayscale(1); opacity: .55; }
        .research-table tr.factor-active td:first-child { box-shadow: inset 3px 0 0 #36443d; }
        .archive-breadcrumb { display: flex; align-items: center; gap: .55rem; margin: 2.2rem 0 -1.2rem; color: var(--muted); font-size: .78rem; }
        .archive-breadcrumb a { color: var(--green) !important; text-decoration: none !important; }
        .archive-breadcrumb strong { color: var(--ink); font-weight: 600; }
        .archive-breadcrumb .archive-rename, .archive-breadcrumb .archive-note { position: relative; z-index: 2; box-sizing: border-box; flex: 0 0 1.75rem; display: inline-flex; align-items: center; justify-content: center; width: 1.75rem; min-width: 1.75rem; height: 1.75rem; min-height: 1.75rem; margin-left: .05rem; padding: 0; color: #989898 !important; background: #f6f6f6; border: 1px solid #dedede; border-radius: 5px; cursor: pointer; pointer-events: auto !important; touch-action: manipulation; font: inherit; line-height: 1; transition: color .16s ease, border-color .16s ease, background .16s ease, transform .16s ease; }
        .archive-breadcrumb .archive-rename:hover, .archive-breadcrumb .archive-note:hover { color: var(--green) !important; background: #f4f7f4; border-color: rgba(23,107,91,.38); }
        .archive-breadcrumb .archive-note.has-note { color: var(--green) !important; background: #eef4ef; border-color: rgba(23,107,91,.28); }
        .archive-breadcrumb .archive-rename:active, .archive-breadcrumb .archive-note:active { transform: scale(.93); }
        .archive-breadcrumb .archive-rename:focus-visible, .archive-breadcrumb .archive-note:focus-visible { outline: 2px solid rgba(23,107,91,.4); outline-offset: 2px; }
        .archive-breadcrumb .archive-note[data-note]:not([data-note=""]):hover::after { content: attr(data-note); position: absolute; z-index: 4; top: calc(100% + .42rem); left: 50%; width: max-content; max-width: min(22rem, calc(100vw - 3rem)); padding: .48rem .58rem; color: var(--ink); background: #fbfaf6; border: 1px solid var(--line); border-radius: 4px; box-shadow: 0 8px 18px rgba(24,32,29,.12); font: 500 .72rem/1.45 Geist, "Microsoft YaHei", sans-serif; text-align: left; white-space: normal; overflow-wrap: anywhere; pointer-events: none; transform: translateX(-50%); }
        .archive-breadcrumb .material-symbols-rounded { pointer-events: none; font-size: 1rem; font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 20; }
        .search-result-link { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-top: .75rem; padding: .82rem .9rem; color: var(--ink) !important; background: rgba(251,250,246,.72); border: 1px solid var(--line); border-radius: 4px; text-decoration: none !important; transition: border-color .2s ease, transform .2s ease; }
        .search-result-link:hover { border-color: var(--green); transform: translateY(-1px); }
        .search-result-link strong { color: var(--green); font-size: .8rem; }
        .search-result-link span { color: var(--muted); font-size: .72rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .history-quick-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .6rem; margin: 1rem 0 1.5rem; }
        .history-quick-link { display: flex; align-items: center; justify-content: space-between; gap: 1rem; min-width: 0; padding: .85rem .95rem; color: var(--ink) !important; background: rgba(251,250,246,.72); border: 1px solid var(--line); border-radius: 4px; text-decoration: none !important; transition: border-color .2s ease, transform .2s ease; }
        .history-quick-link:hover { border-color: var(--green); transform: translateY(-1px); }
        .history-quick-link strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: .82rem; }
        .history-quick-link span { flex: 0 0 auto; color: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: .68rem; }
        .research-table td.numeric { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
        .research-table-wrap.scrollable { max-height: 38rem; }
        .research-table-wrap.scrollable .research-table th { position: sticky; top: 0; z-index: 1; }
        .research-table.wide { min-width: 1100px; }
        .qualitative {
            display: inline-block;
            min-width: 3.2rem;
            padding: .18rem .45rem;
            border-left: 3px solid #9aa49f;
            color: #5e6863;
            background: #eceeea;
            font-weight: 600;
        }
        .qualitative.bullish { color: #934634; border-color: var(--coral); background: #f2e5e1; }
        .qualitative.bearish { color: #11594c; border-color: var(--green); background: #e2eee9; }

        .featured-carousel { position: relative; min-height: 24rem; margin-bottom: 2.4rem; overflow: hidden; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
        .featured-slide { position: absolute; inset: 0; display: grid; grid-template-rows: auto 1fr; opacity: 0; pointer-events: none; animation: featuredCycle 15s infinite cubic-bezier(.22,.8,.25,1); will-change: opacity, transform, filter; }
        .featured-carousel.single .featured-slide { display: none; animation: none; }
        .featured-carousel.single .featured-slide:first-child { display: grid; opacity: 1; pointer-events: auto; }
        .featured-carousel:hover .featured-slide { animation-play-state: paused; }
        .featured-slide-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 1rem 0 .75rem; }
        .featured-strategy { color: var(--ink); font-size: 1rem; font-weight: 650; }
        .featured-strategy a { color: inherit !important; text-decoration: none !important; border-bottom: 1px solid rgba(23,107,91,.25); }
        .featured-strategy a:hover { color: var(--coral) !important; border-bottom-color: var(--coral); }
        .featured-badge { color: var(--coral); font-size: .74rem; font-weight: 650; }
        .featured-metrics { display: grid; grid-template-columns: repeat(4, 1fr); }
        .featured-metric { padding: 1.2rem 1.1rem 1.3rem 0; min-width: 0; }
        .featured-metric + .featured-metric { border-left: 1px solid var(--line); padding-left: 1.1rem; }
        .featured-metric:nth-child(4n + 1) { border-left: 0; padding-left: 0; }
        .featured-metric:nth-child(n + 5) { border-top: 1px solid var(--line); }
        .featured-metric.highlight { background: rgba(187,101,79,.075); box-shadow: inset 0 3px 0 var(--coral); padding-left: 1.1rem; }
        .featured-metric-label { color: var(--muted); font-size: .76rem; margin-bottom: .65rem; }
        .featured-metric-value { color: var(--ink); font-family: "IBM Plex Mono", "Microsoft YaHei", sans-serif; font-size: clamp(1.35rem, 2vw, 2.05rem); font-weight: 600; white-space: nowrap; }
        .featured-metric.compact-value .featured-metric-value { font-size: clamp(.95rem, 1.35vw, 1.3rem); line-height: 1.25; white-space: normal; }
        .featured-metric.highlight .featured-metric-value { color: var(--coral); }
        .featured-metric-benchmark { color: var(--muted); font-size: .72rem; margin-top: .65rem; }
        .period-comparison { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); margin: .8rem 0 1.6rem; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
        .period-snapshot { padding: 1rem 1rem 1.15rem 0; min-width: 0; }
        .period-snapshot + .period-snapshot { border-left: 1px solid var(--line); padding-left: 1rem; }
        .period-snapshot.active { background: rgba(23,107,91,.055); box-shadow: inset 0 3px 0 var(--green); }
        .period-snapshot h4 { margin: 0 0 .7rem; font-size: .82rem; color: var(--ink); }
        .period-snapshot .period-range { color: var(--muted); font-size: .66rem; margin-bottom: .75rem; }
        .period-snapshot dl { display: grid; grid-template-columns: 1fr auto; gap: .42rem .7rem; margin: 0; font-size: .72rem; }
        .period-snapshot dt { color: var(--muted); }
        .period-snapshot dd { margin: 0; font-family: "IBM Plex Mono", "Microsoft YaHei", sans-serif; font-variant-numeric: tabular-nums; }

        @keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pageReveal { from { opacity: 0; } to { opacity: 1; } }
        @keyframes cascadeReveal { from { opacity: 0; transform: translateY(9px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes featuredCycle { 0%, 29% { opacity: 1; pointer-events: auto; transform: translateY(0); } 33%, 96% { opacity: 0; pointer-events: none; transform: translateY(8px); } 100% { opacity: 1; pointer-events: auto; transform: translateY(0); } }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] { padding-top: 2.1rem; }
            .app-header-integration { right: 6.5rem; padding: 0 1rem; grid-template-columns: minmax(10rem, 1fr) auto; gap: 1rem; }
            .top-navigation { justify-content: flex-end; gap: .8rem; }
            .app-header-spacer { display: none; }
            .hero-shell, .launch-grid { grid-template-columns: 1fr; }
            .st-key-historical_hero [data-testid="stHorizontalBlock"] { gap: 1.4rem !important; }
            .historical-hero-side, .st-key-period_compact { border-left: 0; border-top: 1px solid var(--line); padding: 1.1rem 0 0; }
            .hero-shell { gap: 2rem; padding-top: 2.6rem; }
            .hero-title { font-size: 2.55rem; }
            .hero-aside { border-left: 0; border-top: 1px solid var(--line); padding: 1.3rem 0 0; }
            .metric-cell { grid-column: span 6; }
            .metric-cell:nth-child(odd) { border-left: 0; padding-left: 0; }
            .metric-cell:nth-child(n + 3) { border-top: 1px solid var(--line); }
            .logic-flow { grid-template-columns: 1fr; }
            .featured-carousel { min-height: 42rem; }
            .featured-metrics { grid-template-columns: 1fr 1fr; }
            .period-comparison { grid-template-columns: 1fr 1fr; }
            .featured-metric:nth-child(odd) { border-left: 0; padding-left: 0; }
            .featured-metric:nth-child(n + 3) { border-top: 1px solid var(--line); }
            .history-quick-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 560px) {
            [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
            .brand-sub { display: none; }
            .brand-name { white-space: nowrap; }
            .app-header-integration { right: 5.5rem; padding: 0 .65rem; grid-template-columns: auto 1fr; gap: .7rem; }
            .brand-name { font-size: .8rem; }
            .top-navigation { gap: .55rem; }
            .top-nav-link, .top-nav-link:link, .top-nav-link:visited { font-size: .72rem; }
            .hero-title { font-size: 2.55rem; }
            .metric-cell { grid-column: span 12; border-left: 0 !important; padding-left: 0 !important; border-bottom: 1px solid var(--line); }
            .metric-cell:last-child { border-bottom: 0; }
            .featured-carousel { min-height: 72rem; }
            .featured-slide-header { align-items: flex-start; flex-direction: column; }
            .featured-metrics { grid-template-columns: 1fr; }
            .featured-metric { border-left: 0 !important; border-top: 1px solid var(--line); padding-left: 0 !important; }
            .featured-metric.highlight { padding-left: .8rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _select_config() -> Path:
    st.sidebar.markdown("## 策略控制台")
    st.sidebar.caption("参数修改只在本次会话生效；保存后才会写入新的 JSON 配置。")
    config_dir = ROOT / CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = config_dir / "baselines"
    default = baseline_dir / "基线07_v2上限50_重胜率_周频_推荐.json"
    if not default.exists():
        default = config_dir / "策略02_最佳权重_中性0.5.json"
    if not default.exists():
        default = config_dir / "dashboard_signal_v1_default.json"
    return _strategy_config_picker(
        "基线配置",
        "home_baseline_config",
        default=default,
        surface=st.sidebar,
        help_text="选择基础配置、保留基线、搜索结果或历史实验配置作为本次调参起点。",
        include_factor_archives=True,
    )


@st.cache_data(show_spinner=False)
def _benchmark_date_bounds() -> tuple[object, object]:
    return market_date_bounds(ROOT, GOV_10Y)


@st.cache_data(show_spinner=False)
def _full_strategy_date_bounds(signal_frequency: str = "weekly") -> tuple[str, str]:
    signal_path = ROOT / signal_file_for_frequency(signal_frequency)
    benchmark_dates = load_market_data(ROOT, GOV_10Y)["date"]
    signal_dates = pd.to_datetime(
        pd.read_csv(signal_path, encoding="utf-8-sig", usecols=["信号日期"])["信号日期"], errors="coerce"
    ).dropna()
    if benchmark_dates.empty or signal_dates.empty:
        raise ValueError("基准或信号文件没有可用日期")
    usable = benchmark_dates.loc[benchmark_dates >= signal_dates.min()]
    if usable.empty:
        raise ValueError("基准与信号数据没有重叠日期")
    return usable.iloc[0].strftime("%Y-%m-%d"), benchmark_dates.iloc[-1].strftime("%Y-%m-%d")


def _sidebar_config(
    base: DashboardStrategyConfig,
    key_prefix: str,
    default_start: object | None = None,
    default_end: object | None = None,
) -> tuple[DashboardStrategyConfig, bool, bool]:
    widget_key = lambda name: f"{key_prefix}_{name}"
    with st.sidebar.container(border=True):
        name = st.text_input(
            "另存为策略名称",
            value=base.name,
            help="保存时将使用该名称生成 JSON；同名文件会自动增加 _02、_03，不会覆盖。",
            key=widget_key("strategy_name"),
        )
        run_col, save_col = st.columns(2)
        run_clicked = run_col.button("运行回测", type="primary", use_container_width=True, key=widget_key("run_backtest"))
        save_clicked = save_col.button("保存配置", type="primary", use_container_width=True, key=widget_key("save_config"))
        provenance = base.research_provenance or {}
        is_factor_archive = isinstance(provenance, dict) and provenance.get("研究类型") == "因子增加单一策略归档"
        if _is_rolling_provenance(provenance):
            _render_rolling_strategy_notice()
        elif is_factor_archive:
            _render_factor_strategy_notice()
        else:
            _render_fixed_strategy_notice()
        st.caption("先命名，再保存；运行使用当前页面参数。")

    st.sidebar.caption(f"条件基准：{CONDITIONAL_BENCHMARK_NAME}")
    frequency_label = st.sidebar.segmented_control(
        "信号频率",
        ["周频", "日频"],
        default="日频" if base.signal_frequency == "daily" else "周频",
        key=widget_key("signal_frequency"),
        help="日频按交易日更新信号和目标仓位；周频沿用每周看板。",
    )
    selected_frequency = "daily" if frequency_label == "日频" else "weekly"
    available_start, available_end = _benchmark_date_bounds()
    if st.sidebar.button("使用最新行情结束日", use_container_width=True, key=widget_key("use_latest_end")):
        st.session_state[widget_key("end_date")] = available_end
        st.rerun()
    initial_start = pd.Timestamp(base.backtest_start or default_start or available_start).date()
    initial_end = pd.Timestamp(base.backtest_end or default_end or available_end).date()
    initial_start = max(available_start, min(initial_start, available_end))
    initial_end = max(available_start, min(initial_end, available_end))
    with st.sidebar.expander("回测区间", expanded=True):
        date_col1, date_col2 = st.columns(2)
        selected_start = date_col1.date_input(
            "起始日期", value=initial_start, min_value=available_start, max_value=available_end, key=widget_key("start_date")
        )
        selected_end = date_col2.date_input(
            "结束日期", value=initial_end, min_value=available_start, max_value=available_end, key=widget_key("end_date")
        )
        if selected_start > selected_end:
            st.error("起始日期不能晚于结束日期。")
            run_clicked = False
        st.caption(f"可用收益率曲线：{available_start} 至 {available_end}；实际起点还取决于首个可用信号。")

    with st.sidebar.expander("因子权重", expanded=True):
        st.caption("最小调整步长为 5 分。")
        weight_values = {}
        for key, label in WEIGHT_LABELS.items():
            default = getattr(base.weights, key)
            if key == "fly_penalty":
                st.number_input(label, value=0.0, disabled=True, key=f"{key_prefix}_weight_{key}_disabled")
                weight_values[key] = 0.0
                continue
            min_value = -50.0 if key == "fly_penalty" else 0.0
            max_value = 0.0 if key == "fly_penalty" else 50.0
            weight_values[key] = st.slider(label, min_value, max_value, float(default), 5.0, key=f"{key_prefix}_weight_{key}")

    with st.sidebar.expander("定性阈值"):
        threshold_values = {
            "supply_low": st.slider("供给低分位", 0.0, 50.0, float(base.thresholds.supply_low), 5.0, key=widget_key("threshold_supply_low")),
            "supply_high": st.slider("供给高分位", 50.0, 100.0, float(base.thresholds.supply_high), 5.0, key=widget_key("threshold_supply_high")),
            "demand_low": st.slider("需求低分位", 0.0, 50.0, float(base.thresholds.demand_low), 5.0, key=widget_key("threshold_demand_low")),
            "demand_high": st.slider("需求高分位", 50.0, 100.0, float(base.thresholds.demand_high), 5.0, key=widget_key("threshold_demand_high")),
            "spread_low": st.slider("地方债利差低分位", 0.0, 50.0, float(base.thresholds.spread_low), 5.0, key=widget_key("threshold_spread_low")),
            "spread_high": st.slider("地方债利差高分位", 50.0, 100.0, float(base.thresholds.spread_high), 5.0, key=widget_key("threshold_spread_high")),
            "ncd_low": st.slider("NCD利差低分位", 0.0, 50.0, float(base.thresholds.ncd_low), 5.0, key=widget_key("threshold_ncd_low")),
            "ncd_high": st.slider("NCD利差高分位", 50.0, 100.0, float(base.thresholds.ncd_high), 5.0, key=widget_key("threshold_ncd_high")),
            "spread_change_bp": st.slider("利差5日变化阈值（BP）" if selected_frequency == "daily" else "利差周变化阈值（BP）", 0.5, 10.0, float(base.thresholds.spread_change_bp), 0.5, key=widget_key("threshold_spread_change")),
        }

    with st.sidebar.expander("仓位规则", expanded=True):
        position_values = {
            "bullish_threshold": st.slider("看多分数阈值", 50.0, 100.0, float(base.positions.bullish_threshold), 5.0, key=widget_key("position_bullish_threshold")),
            "bearish_threshold": st.slider("看空分数阈值", 0.0, 50.0, float(base.positions.bearish_threshold), 5.0, key=widget_key("position_bearish_threshold")),
            "bullish_position": st.slider("看多仓位", -1.0, 1.5, float(base.positions.bullish_position), 0.1, key=widget_key("position_bullish")),
            "neutral_position": st.slider("中性仓位", -1.0, 1.5, float(base.positions.neutral_position), 0.1, key=widget_key("position_neutral")),
            "bearish_position": st.slider("看空仓位", -1.5, 1.0, float(base.positions.bearish_position), 0.1, key=widget_key("position_bearish")),
            "bearish_min_core_factors": st.select_slider(
                "看空所需核心利空模块数",
                options=[0, 1, 2, 3, 4],
                value=int(base.positions.bearish_min_core_factors),
                help="0表示只看总分；2表示至少两个模块同时利空。",
                key=widget_key("position_min_core"),
            ),
            "bearish_require_supply_or_demand": int(
                st.toggle(
                    "看空必须包含供给或需求利空",
                    value=bool(base.positions.bearish_require_supply_or_demand),
                    key=widget_key("position_require_supply_demand"),
                )
            ),
            "bearish_confirmation_periods": st.select_slider(
                "看空连续确认天数" if selected_frequency == "daily" else "看空连续确认周数",
                options=[1, 2, 3],
                value=int(base.positions.bearish_confirmation_periods),
                key=widget_key("position_confirmation"),
            ),
        }
        stop_col1, stop_col2 = st.columns(2)
        take_profit_enabled = stop_col1.toggle(
            "启用止盈",
            value=float(base.positions.take_profit_bp) > 0.0,
            key=widget_key("take_profit_enabled"),
        )
        stop_loss_enabled = stop_col2.toggle(
            "启用止损",
            value=float(base.positions.stop_loss_bp) > 0.0,
            key=widget_key("stop_loss_enabled"),
        )
        position_values["take_profit_bp"] = stop_col1.number_input(
            "止盈阈值（BP）",
            min_value=0.5,
            max_value=100.0,
            value=max(float(base.positions.take_profit_bp), 3.0),
            step=0.5,
            disabled=not take_profit_enabled,
            key=widget_key("take_profit_bp"),
        ) if take_profit_enabled else 0.0
        position_values["stop_loss_bp"] = stop_col2.number_input(
            "止损阈值（BP）",
            min_value=0.5,
            max_value=100.0,
            value=max(float(base.positions.stop_loss_bp), 3.0),
            step=0.5,
            disabled=not stop_loss_enabled,
            key=widget_key("stop_loss_bp"),
        ) if stop_loss_enabled else 0.0
        st.caption("触发日收益计入该笔交易；随后清仓。同方向信号持续禁开，直到信号先转为中性或反向。")

    with st.sidebar.expander("搜索目标函数（只读）"):
        st.caption("单次回测不使用目标函数。这里仅展示该配置保存时的搜索口径；请在顶部“搜索研究”页调整。")
        objective_values = base.objective.as_dict()
        objective_display = pd.DataFrame(
            [
                {"项目": "累计收益率资本利得BP", "权重": objective_values["capital_gain_bp_weight"]},
                {"项目": "收益率资本利得超额BP", "权重": objective_values["capital_gain_excess_bp_weight"]},
                {"项目": "已平仓交易胜率", "权重": objective_values["capital_trade_win_rate_weight"]},
                {"项目": "平均每笔盈利BP", "权重": objective_values["capital_gain_avg_win_bp_weight"]},
                {"项目": "资本利得回撤BP", "权重": objective_values["capital_gain_drawdown_bp_penalty"]},
                {"项目": "累计收益", "权重": objective_values["total_return_weight"]},
                {"项目": "超额收益", "权重": objective_values["excess_return_weight"]},
                {"项目": "夏普", "权重": objective_values["sharpe_weight"]},
                {"项目": "调仓周期胜率", "权重": objective_values["signal_win_rate_weight"]},
            ]
        )
        st.dataframe(objective_display, hide_index=True, use_container_width=True)

    effective_name = name.strip() or base.name
    parameter_groups = {
        "weights": weight_values,
        "thresholds": threshold_values,
        "positions": position_values,
        "objective": objective_values,
    }
    changed_groups = [
        group
        for group, values in parameter_groups.items()
        if values != getattr(base, group).as_dict()
    ]
    selected_window = {"start_date": selected_start.isoformat(), "end_date": selected_end.isoformat()}
    base_window = {
        "start_date": base.backtest_start or initial_start.isoformat(),
        "end_date": base.backtest_end or initial_end.isoformat(),
    }
    if selected_window != base_window:
        changed_groups.append("backtest")
    if selected_frequency != base.signal_frequency:
        changed_groups.append("frequency")
    if effective_name == base.name and changed_groups:
        effective_name = _suggest_variant_name(
            base,
            weight_values,
            threshold_values,
            position_values,
            changed_groups,
            selected_window,
            selected_frequency,
        )
        st.sidebar.info(f"保存时自动更名：{effective_name}")

    config = DashboardStrategyConfig(
        name=effective_name,
        weights=DashboardWeights(**weight_values),
        thresholds=DashboardThresholds(**threshold_values),
        positions=DashboardPositionPolicy(**position_values),
        objective=ObjectiveConfig(**objective_values),
        backtest_start=selected_window["start_date"],
        backtest_end=selected_window["end_date"],
        benchmark_id=GOV_10Y,
        signal_frequency=selected_frequency,
        research_provenance=base.research_provenance,
        source_config_path=base.source_config_path,
    )
    config = record_manual_changes(base, config, ROOT)
    return config, run_clicked, save_clicked


def _suggest_variant_name(
    base: DashboardStrategyConfig,
    weights: dict[str, float],
    thresholds: dict[str, float],
    positions: dict[str, float | int],
    changed_groups: list[str],
    backtest: dict[str, str],
    signal_frequency: str,
) -> str:
    tags: list[str] = []
    if "weights" in changed_groups:
        tags.append("权重调整")
    if "thresholds" in changed_groups:
        tags.append("阈值调整")
    if "positions" in changed_groups:
        tags.append(
            "仓位多{}_中{}_空{}".format(
                _name_number(positions["bullish_position"]),
                _name_number(positions["neutral_position"]),
                _name_number(positions["bearish_position"]),
            )
        )
        if float(positions["bearish_threshold"]) != base.positions.bearish_threshold:
            tags.append(f"看空{_name_number(positions['bearish_threshold'])}")
        if float(positions.get("take_profit_bp", 0.0)) > 0.0:
            tags.append(f"止盈{_name_number(positions['take_profit_bp'])}BP")
        if float(positions.get("stop_loss_bp", 0.0)) > 0.0:
            tags.append(f"止损{_name_number(positions['stop_loss_bp'])}BP")
    if "frequency" in changed_groups:
        tags.append("日频" if signal_frequency == "daily" else "周频")
        if int(positions["bearish_min_core_factors"]) > 0:
            tags.append(f"确认{int(positions['bearish_min_core_factors'])}模块")
        if int(positions["bearish_require_supply_or_demand"]):
            tags.append("含供需")
        if int(positions["bearish_confirmation_periods"]) > 1:
            period_unit = "天" if signal_frequency == "daily" else "周"
            tags.append(f"连续{int(positions['bearish_confirmation_periods'])}{period_unit}")
    if "objective" in changed_groups:
        tags.append("搜索目标调整")
    if "backtest" in changed_groups:
        tags.append(f"区间{backtest['start_date'].replace('-', '')}-{backtest['end_date'].replace('-', '')}")
    return f"{base.name}__改_{'_'.join(tags)}"


def _name_number(value: object) -> str:
    return f"{float(value):g}"


def _save_config(config: DashboardStrategyConfig) -> None:
    name = config.name.strip() or f"dashboard_config_{datetime.now():%Y%m%d_%H%M%S}"
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    save_path = _next_available_config_path(ROOT / CONFIG_DIR / f"{safe_name}.json")
    save_strategy_config(config, save_path)
    st.sidebar.success(f"已保存：{save_path.name}")
    st.toast(f"配置已保存：{save_path.name}")


def _next_available_config_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("同名配置版本过多，请更换策略配置名称。")


def _run_backtest(config: DashboardStrategyConfig) -> Path | None:
    try:
        with st.spinner("正在计算收益、归因与调仓周期诊断..."):
            provenance = config.research_provenance or {}
            is_factor_archive = isinstance(provenance, dict) and provenance.get("研究类型") == "因子增加单一策略归档"
            source_archive = _rolling_archive_dir_from_config(config)
            is_rolling_archive = _is_rolling_provenance(provenance) or (
                source_archive is not None and _is_rolling_archive(source_archive, provenance)
            )
            if is_factor_archive:
                daily, signals, strategy_metrics, benchmark_metrics, experiment_dir = _rerun_factor_archive(config)
            elif is_rolling_archive:
                daily, signals, strategy_metrics, benchmark_metrics, experiment_dir = _rerun_rolling_archive(config)
            else:
                daily, signals, strategy_metrics, benchmark_metrics = run_dashboard_config(ROOT, config)
                experiment_dir = archive_dashboard_experiment(
                    ROOT,
                    config,
                    daily,
                    signals,
                    strategy_metrics,
                    benchmark_metrics,
                    source="网页",
                )
            diagnostics = _build_period_diagnostics(daily, signals)
        st.session_state[RESULT_STATE_KEY] = (
            daily,
            signals,
            strategy_metrics,
            benchmark_metrics,
            diagnostics,
            config.name,
            experiment_dir,
            load_strategy_config(experiment_dir / "config.json"),
        )
        st.toast(f"实验已归档：{experiment_dir.name}")
        return experiment_dir
    except Exception as exc:
        st.error(f"回测运行失败：{exc}")
        return None


def _rolling_provenance_step(provenance: dict[str, object]) -> dict[str, object] | None:
    steps = provenance.get("steps", []) if isinstance(provenance, dict) else []
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, dict) and str(step.get("operation", "")) == "滚动定参":
            return step
    return None


def _is_rolling_provenance(provenance: object) -> bool:
    return isinstance(provenance, dict) and _rolling_provenance_step(provenance) is not None


def _archive_run_manifest(experiment_dir: Path) -> dict[str, object]:
    """Read an immutable archive manifest without creating any new IDs or files."""
    try:
        payload = json.loads((experiment_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _rolling_output_dir_from_archive(experiment_dir: Path) -> Path | None:
    """Resolve the recorded rolling-result directory for both new and legacy archives."""
    manifest = _archive_run_manifest(experiment_dir)
    research = manifest.get("研究区间", {})
    raw_path = research.get("滚动结果目录") if isinstance(research, dict) else None
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    candidate = (ROOT / Path(raw_path)).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_dir() else None


def _is_rolling_archive(experiment_dir: Path, provenance: object) -> bool:
    """Recognize old rolling archives whose config predates provenance recording."""
    if _is_rolling_provenance(provenance):
        return True
    manifest = _archive_run_manifest(experiment_dir)
    research = manifest.get("研究区间", {})
    return isinstance(research, dict) and str(research.get("研究类型", "")).strip() == "滚动定参"


def _rolling_archive_dir_from_config(config: DashboardStrategyConfig) -> Path | None:
    source = config.source_config_path
    if not source:
        return None
    path = Path(source)
    if path.name != "config.json" or not (path.parent / "run_manifest.json").is_file():
        return None
    return path.parent


def _latest_rolling_period_config(experiment_dir: Path) -> DashboardStrategyConfig | None:
    """Load the last selected rolling parameters; never trigger a new search here."""
    output_dir = _rolling_output_dir_from_archive(experiment_dir)
    if output_dir is None:
        return None
    config_file: Path | None = None
    periods_file = output_dir / "逐期定参与样本外表现.csv"
    try:
        periods = pd.read_csv(periods_file, encoding="utf-8-sig")
        if "参数配置文件" in periods:
            recorded = periods["参数配置文件"].dropna().astype(str).str.strip()
            if not recorded.empty:
                candidate = Path(recorded.iloc[-1])
                if candidate.is_file():
                    config_file = candidate
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        pass
    if config_file is None:
        candidates = sorted((output_dir / "periods").glob("*/参数配置.json"))
        if candidates:
            config_file = candidates[-1]
    if config_file is None:
        return None
    try:
        return load_strategy_config(config_file)
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return None


def _current_rolling_signals(
    experiment_dir: Path,
    archived_signals: pd.DataFrame,
) -> tuple[pd.DataFrame | None, DashboardStrategyConfig | None]:
    """Append display-only current signals using the final saved rolling parameters.

    Performance remains the immutable archived path.  Only the final rolling
    parameters are applied to later signal-grid rows, so showing a new signal
    neither changes historical scores nor initiates a rolling search.
    """
    selected = _latest_rolling_period_config(experiment_dir)
    if selected is None or archived_signals.empty or "signal_date" not in archived_signals:
        return None, selected
    archive_latest = pd.to_datetime(archived_signals["signal_date"], errors="coerce").max()
    if pd.isna(archive_latest):
        return None, selected
    try:
        current = build_dashboard_signal(
            ROOT,
            weights=selected.weights,
            thresholds=selected.thresholds,
            position_policy=selected.positions,
            factor_windows=selected.factor_windows,
            signal_frequency=selected.signal_frequency,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None, selected
    current = current.loc[pd.to_datetime(current["signal_date"], errors="coerce") >= archive_latest].copy()
    if current.empty:
        return archived_signals, selected
    combined = pd.concat([archived_signals, current], ignore_index=True, sort=False)
    combined["signal_date"] = pd.to_datetime(combined["signal_date"], errors="coerce")
    combined = (
        combined.dropna(subset=["signal_date"])
        .drop_duplicates("signal_date", keep="last")
        .sort_values("signal_date")
        .reset_index(drop=True)
    )
    return combined, selected


def _merge_current_signal_tail(archived: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame | None:
    """Keep frozen history and replace only its last signal onward for display."""
    if archived.empty or "signal_date" not in archived or "signal_date" not in current:
        return None
    archive_latest = pd.to_datetime(archived["signal_date"], errors="coerce").max()
    if pd.isna(archive_latest):
        return None
    tail = current.loc[pd.to_datetime(current["signal_date"], errors="coerce") >= archive_latest].copy()
    if tail.empty:
        return archived
    merged = pd.concat([archived, tail], ignore_index=True, sort=False)
    merged["signal_date"] = pd.to_datetime(merged["signal_date"], errors="coerce")
    return (
        merged.dropna(subset=["signal_date"])
        .drop_duplicates("signal_date", keep="last")
        .sort_values("signal_date")
        .reset_index(drop=True)
    )


def _is_factor_research_strategy(config: DashboardStrategyConfig) -> bool:
    provenance = config.research_provenance or {}
    return isinstance(provenance, dict) and bool(
        provenance.get("因子研究权重")
        or provenance.get("因子增加研究路线")
        or str(provenance.get("研究类型", "")).startswith("因子")
    )


def _factor_display_signal_config(config: DashboardStrategyConfig) -> ExpansionResearchConfig | None:
    """Rebuild factor rows from saved parameters when an old archive omitted them."""
    provenance = config.research_provenance or {}
    if not isinstance(provenance, dict):
        return None
    weights = _display_factor_weights(config)
    if not weights:
        return None
    route = provenance.get("因子增加研究路线", {})
    route = route if isinstance(route, dict) else {}
    factor_version = str(route.get("因子版本") or provenance.get("因子版本") or "").strip()
    if factor_version not in {"原始因子", "扩展因子", "自选因子"}:
        factor_version = "扩展因子" if any(key in EXPANDED_FACTOR_COLUMNS for key in weights) else "原始因子"
    recorded_enabled = route.get("事前启用因子") or provenance.get("因子集合") or tuple(weights)
    enabled = tuple(str(value) for value in recorded_enabled if str(value) in ALL_FACTOR_COLUMNS)
    universe = BASE_FACTOR_COLUMNS if factor_version == "原始因子" else ALL_FACTOR_COLUMNS
    enabled = tuple(column for column in universe if column in enabled)
    if not enabled:
        return None
    objective_name = str(route.get("目标", "收益"))
    if objective_name not in OBJECTIVES:
        objective_name = "收益"
    return ExpansionResearchConfig(
        factor_version=factor_version,
        objective_name=objective_name,
        signal_frequency=config.signal_frequency,
        weights={column: float(weights.get(column, 0.0) or 0.0) for column in enabled},
        thresholds=config.thresholds,
        positions=config.positions,
        factor_windows=config.factor_windows,
        objective_config=config.objective,
        enabled_factor_columns=enabled,
    )


def _current_factor_strategy_signals(
    config: DashboardStrategyConfig,
    archived_signals: pd.DataFrame,
) -> pd.DataFrame | None:
    factor_config = _factor_display_signal_config(config)
    if factor_config is None:
        return None
    try:
        current = build_expansion_signals(ROOT, factor_config)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return _merge_current_signal_tail(archived_signals, current)


def _current_fixed_strategy_signals(
    config: DashboardStrategyConfig,
    archived_signals: pd.DataFrame,
) -> pd.DataFrame | None:
    """Fill an old fixed archive's latest factor view from its frozen parameters."""
    try:
        current = build_dashboard_signal(
            ROOT,
            weights=config.weights,
            thresholds=config.thresholds,
            position_policy=config.positions,
            factor_windows=config.factor_windows,
            signal_frequency=config.signal_frequency,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return _merge_current_signal_tail(archived_signals, current)


def _render_rolling_strategy_notice() -> None:
    """Render the same understated rolling-status cue wherever a strategy can be run."""
    st.markdown(
        "<aside class='strategy-mode-note rolling-strategy-note'><strong>滚动定参策略</strong>"
        "<span>本次会复用已保存的逐期参数并重算表现；只有行情跨过下一定参点时，才补搜新增期间。"
        "若修改策略参数，则按新的设置重新定参。</span></aside>",
        unsafe_allow_html=True,
    )


def _render_fixed_strategy_notice() -> None:
    st.markdown(
        "<aside class='strategy-mode-note'><strong>固定参数策略</strong>"
        "<span>本次按当前保存参数完整重算，不会自动执行权重或阈值搜索；只有手动修改参数后，"
        "才会以修改后的固定参数生成新结果。</span></aside>",
        unsafe_allow_html=True,
    )


def _render_factor_strategy_notice() -> None:
    st.markdown(
        "<aside class='strategy-mode-note'><strong>因子研究策略</strong>"
        "<span>本次会按归档记录的已启用因子集合和搜索规则重新生成信号；不会降级为旧8因子回测。"
        "左侧旧版权重仅是兼容展示，实际所有因子权重由该策略的研究引擎处理。</span></aside>",
        unsafe_allow_html=True,
    )


def _rerun_rolling_archive(
    config: DashboardStrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object], Path]:
    """Re-run a rolling archive using its recorded window/search contract."""
    provenance = config.research_provenance or {}
    step = _rolling_provenance_step(provenance)
    output = step.get("output", {}) if isinstance(step, dict) and isinstance(step.get("output"), dict) else {}
    output_path = str(output.get("path", ""))
    manifest_path = ROOT / output_path / "滚动配置.json" if output_path else None
    if manifest_path is None or not manifest_path.exists():
        archive_dir = _rolling_archive_dir_from_config(config)
        output_dir = _rolling_output_dir_from_archive(archive_dir) if archive_dir is not None else None
        manifest_path = output_dir / "滚动配置.json" if output_dir is not None else None
    if manifest_path is None or not manifest_path.exists():
        raise ValueError("滚动定参原始清单不存在，无法按原窗口复现")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    search_mode = str(manifest.get("搜索模式", "combined"))
    training_mode = str(manifest.get("训练方式", "expanding"))
    interval_text = str(manifest.get("定参频率", "3 months")).split()
    interval_count = int(float(interval_text[0])) if interval_text else 3
    interval_unit = interval_text[1] if len(interval_text) > 1 else "months"
    requested_first_end = None if manifest.get("启动方式") == "自动最早启动" else manifest.get("请求首次训练截止日")
    rolling_config = RollingResearchConfig(
        base_config=replace(config, backtest_start=None, backtest_end=None),
        search_mode=search_mode,
        training_mode=training_mode,
        first_training_end=requested_first_end,
        recalibration_interval=interval_count,
        recalibration_unit=interval_unit,
        rolling_window_months=manifest.get("固定训练窗口月数"),
        min_objective_improvement=float(manifest.get("最小目标函数改善阈值", 0.0) or 0.0),
        weight_search_version=str(manifest.get("权重搜索版本", "v2")),
        minimum_training_months=manifest.get("最低训练长度月数"),
        resume_from=manifest_path.parent,
    )
    result = run_rolling_research(ROOT, rolling_config, progress=st.write)
    experiment_dir = Path(str(result["experiment_dir"]))
    daily, signals, strategy_metrics, benchmark_metrics, _ = load_experiment_result(experiment_dir)
    return daily, signals, strategy_metrics, benchmark_metrics, experiment_dir


def _rerun_factor_archive(
    config: DashboardStrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object], Path]:
    provenance = config.research_provenance or {}
    metadata = provenance.get("因子增加研究路线", {}) if isinstance(provenance.get("因子增加研究路线"), dict) else {}
    objective = str(metadata.get("目标", "收益"))
    factor_version = str(metadata.get("因子版本", "扩展因子"))
    if factor_version not in {"原始因子", "扩展因子", "自选因子"}:
        factor_version = "扩展因子"
    frequency = str(metadata.get("频率", config.signal_frequency))
    if frequency in {"日频", "daily"}:
        frequency = "daily"
    elif frequency in {"周频", "weekly"}:
        frequency = "weekly"
    else:
        raise ValueError(f"因子策略溯源中的信号频率无法识别：{frequency}")
    training_end = None
    for step in provenance.get("steps", []) if isinstance(provenance.get("steps"), list) else []:
        if isinstance(step, dict) and step.get("training_end"):
            training_end = str(step["training_end"])
            break
    training_end = training_end or DEFAULT_TRAINING_END
    enabled_factor_columns = metadata.get("事前启用因子", ()) if isinstance(metadata, dict) else ()
    if not enabled_factor_columns and isinstance(provenance, dict):
        disabled = set(provenance.get("事前停用因子", []) or [])
        enabled_factor_columns = tuple(column for column in ALL_FACTOR_COLUMNS if column not in disabled)
    study_dir = run_factor_expansion_static_research(
        ROOT,
        training_end=training_end,
        objectives=(objective,),
        frequencies=(frequency,),
        research_scope="单一策略对测试",
        base_config=config,
        enabled_factor_columns=tuple(enabled_factor_columns) or None,
    )
    experiment_dir = _archive_single_factor_expansion_result(
        study_dir, objective, frequency, training_end, config, factor_version
    )
    daily, signals, strategy_metrics, benchmark_metrics, _ = load_experiment_result(experiment_dir)
    return daily, signals, strategy_metrics, benchmark_metrics, experiment_dir


def _render_launch_state(config: DashboardStrategyConfig) -> None:
    frequency_copy = "日度看板信号按交易日" if config.signal_frequency == "daily" else "周度看板信号按周"
    st.markdown(
        f"""
        <section class="hero-shell">
            <div>
                <p class="eyebrow">10Y 地方债方向与仓位研究</p>
                <h1 class="hero-title">把看板判断，转成可复现的持仓路径。</h1>
                <p class="hero-copy">以{frequency_copy}决定交易仓位，主要评价策略捕获的10Y地方债收益率方向变动BP、逐笔胜率和亏损控制；久期折算价格收益、carry与传统净值作为辅助。</p>
            </div>
            <aside class="hero-aside">
                <div class="aside-label">当前中性仓位</div>
                <div class="aside-value">{config.positions.neutral_position:.1f}</div>
                <div class="aside-note">看多 {config.positions.bullish_position:.1f} / 看空 {config.positions.bearish_position:.1f}<br>阈值 {config.positions.bullish_threshold:.0f} / {config.positions.bearish_threshold:.0f}</div>
            </aside>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_launch_details(config: DashboardStrategyConfig) -> None:
    update_copy = "按交易日" if config.signal_frequency == "daily" else "按周"
    st.markdown(
        f"""
        <section class="launch-grid">
            <article class="launch-panel">
                <h3>策略计算路径</h3>
                <div class="logic-flow">
                    <div class="logic-step"><strong>看板因子</strong><span>供给、银行需求、估值利差与非银情绪{update_copy}更新定性结论。</span></div>
                    <div class="logic-step"><strong>分数与仓位</strong><span>因子权重合成总分，再映射到看多、中性和看空三档仓位。</span></div>
                    <div class="logic-step"><strong>交易与诊断</strong><span>非零仓位开仓、归零平仓，反向时先平后开；统计逐笔资本利得BP、胜率与盈亏比。</span></div>
                </div>
            </article>
            <article class="launch-panel">
                <h3>本次参数快照</h3>
                <div class="config-line"><span>看多触发</span><strong>总分 ≥ {config.positions.bullish_threshold:.0f}</strong></div>
                <div class="config-line"><span>看空触发</span><strong>总分 &lt; {config.positions.bearish_threshold:.0f}</strong></div>
                <div class="config-line"><span>仓位档位</span><strong>{config.positions.bullish_position:.1f} / {config.positions.neutral_position:.1f} / {config.positions.bearish_position:.1f}</strong></div>
                <div class="config-line"><span>信号频率</span><strong>{'日频' if config.signal_frequency == 'daily' else '周频'}</strong></div>
                <div class="config-line"><span>条件基准</span><strong>{escape(CONDITIONAL_BENCHMARK_NAME)}</strong></div>
                <div class="config-line"><span>权重总和</span><strong>{sum(config.weights.as_dict().values()):.0f}</strong></div>
            </article>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info("在左侧调整参数，然后运行回测。当前默认配置为已保存的 search 最优策略。")


def _render_summary(strategy_metrics: dict[str, object], benchmark_metrics: dict[str, object], signals: pd.DataFrame) -> None:
    latest = signals.iloc[-1]
    capital_bp = float(strategy_metrics["capital_gain_total_bp"])
    benchmark_capital_bp = float(benchmark_metrics["capital_gain_total_bp"])
    capital_excess_bp = capital_bp - benchmark_capital_bp
    benchmark_name = str(benchmark_metrics.get("benchmark_name", "10Y地方政府债"))
    capital_win_rate = strategy_metrics.get("capital_gain_trade_win_rate")
    average_win_bp = strategy_metrics.get("capital_gain_avg_win_bp")
    capital_drawdown_bp = strategy_metrics.get("capital_gain_max_drawdown_bp")
    annualized_capital_bp = strategy_metrics.get("capital_gain_annualized_bp")
    benchmark_annualized_capital_bp = benchmark_metrics.get("capital_gain_annualized_bp")
    average_loss_bp = strategy_metrics.get("capital_gain_avg_loss_bp")
    worst_trade_bp = strategy_metrics.get("capital_gain_worst_trade_bp")
    drawdown_start = strategy_metrics.get("capital_gain_max_drawdown_start")
    drawdown_end = strategy_metrics.get("capital_gain_max_drawdown_end")
    benchmark_drawdown_start = benchmark_metrics.get("capital_gain_max_drawdown_start")
    benchmark_drawdown_end = benchmark_metrics.get("capital_gain_max_drawdown_end")
    average_holding_days = strategy_metrics.get("capital_gain_avg_holding_days")
    max_holding_days = strategy_metrics.get("capital_gain_max_holding_days")
    win_rate_text = "暂无已平仓" if capital_win_rate is None else _pct(capital_win_rate)
    average_win_text = "暂无盈利交易" if average_win_bp is None else f"{float(average_win_bp):.2f} BP"
    capital_drawdown_text = "暂无" if capital_drawdown_bp is None else f"{float(capital_drawdown_bp):.2f} BP"
    payoff = strategy_metrics.get("capital_gain_profit_loss_ratio")
    sharpe = float(strategy_metrics["sharpe"])
    signal_date = _latest_signal_date(latest)
    st.markdown(
        f"""
        <section class="hero-shell">
            <div>
                <p class="eyebrow">回测结果 / {escape(signal_date)}</p>
                <h1 class="hero-title">当前结论：{escape(str(latest.get('结论', '未识别')))}</h1>
                <p class="hero-copy">总分 {float(latest.get('总分', 0)):.1f}，目标仓位 {float(latest['仓位']):.1f}。资本利得BP按 -仓位 × YTM变化BP 计算，不乘久期。</p>
            </div>
            <aside class="hero-aside">
                <div class="aside-label">最新仓位</div>
                <div class="aside-value">{float(latest['仓位']):.1f}</div>
                <div class="aside-note">信号日期 {escape(signal_date)}<br>策略夏普 {sharpe:.2f}</div>
            </aside>
        </section>
        <section class="metric-grid">
            {_metric_cell('累计资本利得', f'{capital_bp:.2f} BP', f'{benchmark_name}累计 {benchmark_capital_bp:.2f} BP', capital_bp)}
            {_metric_cell('已平仓交易胜率', win_rate_text, f"盈利 {strategy_metrics['capital_gain_winning_trades']:.0f} / 已平仓 {strategy_metrics['capital_gain_closed_trade_count']:.0f} 笔", 0.0 if capital_win_rate is None else float(capital_win_rate) - 0.5)}
            {_metric_cell('平均每笔盈利', average_win_text, _payoff_detail(payoff), 0.0 if average_win_bp is None else float(average_win_bp))}
            {_metric_cell('资本利得最大回撤', capital_drawdown_text, _drawdown_period_detail(drawdown_start, drawdown_end), 0.0 if capital_drawdown_bp is None else float(capital_drawdown_bp))}
            {_metric_cell('超额资本利得', f'{capital_excess_bp:.2f} BP', _relative_excess_detail(benchmark_name, capital_excess_bp, benchmark_capital_bp), capital_excess_bp)}
            {_metric_cell('年化资本利得', '暂无' if annualized_capital_bp is None else f'{float(annualized_capital_bp):.2f} BP', f"{benchmark_name} 暂无" if benchmark_annualized_capital_bp is None else f'{benchmark_name} {float(benchmark_annualized_capital_bp):.2f} BP', 0.0 if annualized_capital_bp is None else float(annualized_capital_bp))}
            {_metric_cell('平均单笔亏损', '暂无亏损' if average_loss_bp is None else f'{float(average_loss_bp):.2f} BP', '区间内无亏损交易' if average_loss_bp is None else ('最大亏损暂无' if worst_trade_bp is None or float(worst_trade_bp) >= 0 else f'最大亏损 {float(worst_trade_bp):.2f} BP'), 0.0 if average_loss_bp is None else float(average_loss_bp))}
            {_metric_cell('平均每笔持有时间', '暂无交易' if average_holding_days is None else f'{float(average_holding_days):.1f} 交易日', '最长持有暂无' if max_holding_days is None else f'最长 {int(max_holding_days)} 交易日', 0.0)}
        </section>
        """,
        unsafe_allow_html=True,
    )


def _metric_cell(label: str, value: str, detail: str, direction: float, compact: bool = False) -> str:
    color_class = "positive" if direction > 0 else "negative" if direction < 0 else ""
    compact_class = " compact-value" if compact else ""
    return f"<div class='metric-cell{compact_class}'><div class='metric-label'>{escape(label)}</div><div class='metric-value {color_class}'>{escape(value)}</div><div class='metric-detail'>{escape(detail)}</div></div>"


def _payoff_detail(payoff: object) -> str:
    return "盈亏比暂无" if payoff is None or pd.isna(payoff) else f"盈亏比 {float(payoff):.2f}"


def _drawdown_duration_text(start: object, end: object) -> str:
    if not start or not end:
        return "暂无持续时间"
    days = max((pd.Timestamp(end) - pd.Timestamp(start)).days, 0)
    return f"持续 {days} 天 / {days / 7.0:.1f} 周"


def _drawdown_period_detail(start: object, end: object) -> str:
    if not start or not end:
        return "暂无回撤区间"
    return f"{start} 至 {end} · {_drawdown_duration_text(start, end)}"


def _relative_excess_detail(benchmark_name: str, excess_bp: object, benchmark_bp: object) -> str:
    if excess_bp is None or benchmark_bp is None or pd.isna(excess_bp) or pd.isna(benchmark_bp) or abs(float(benchmark_bp)) < 1e-12:
        return f"较{benchmark_name}提升暂无"
    return f"较{benchmark_name} {float(excess_bp) / abs(float(benchmark_bp)):+.2%}"


def _run_display_name(experiment_dir: Path | None, strategy_name: object) -> str:
    """Show the immutable archive run id without changing stored names or paths."""
    name = _normalise_display_name(strategy_name)
    if experiment_dir is None:
        return name
    archive_name = Path(experiment_dir).name
    archive_id = _archive_run_id(Path(experiment_dir))
    if archive_id is None:
        return name
    display_name = _experiment_display_names().get(archive_name, name)
    return f"[{archive_id}] {_normalise_display_name(display_name)}"


def _provenance_reference_experiment(reference: object) -> Path | None:
    """Resolve a recorded strategy reference to its immutable result archive.

    Older provenance records often point at the source configuration under
    ``configs/`` rather than at the final archived ``config.json``.  The
    source configuration is not independently viewable and does not have a
    run ID, so resolve it by its saved configuration identity.  This makes a
    provenance reference consistently mean an identified, clickable strategy
    result without ever allocating or changing an archive ID at display time.
    """
    if not isinstance(reference, dict):
        return None
    raw_path = str(reference.get("path", "")).strip()
    if not raw_path:
        # Some early factor-rolling records retained only the exact parent
        # archive name.  Resolve that read-only reference so it still carries
        # the immutable run ID and result-page link.
        label = _normalise_display_name(reference.get("label", ""))
        if not label:
            return None
        matches: list[Path] = []
        for candidate_path in sorted((ROOT / "backtest_outputs" / "experiments").glob("*/config.json")):
            try:
                candidate = load_strategy_config(candidate_path)
            except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
                continue
            if _normalise_display_name(candidate.name) == label:
                matches.append(candidate_path.parent)
        return matches[0] if len(matches) == 1 else None
    path = Path(raw_path)
    path = path if path.is_absolute() else ROOT / path
    if path.name == "config.json" and (path.parent / "run_manifest.json").exists():
        return path.parent
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        source = load_strategy_config(path)
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return None

    def signature(config: DashboardStrategyConfig) -> str:
        values = config_parameters(config)
        # The archived strategy name may have been renamed by the researcher;
        # identity is its executable parameter set, not its presentation name.
        values.pop("name", None)
        return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    source_signature = signature(source)
    source_name = _normalise_display_name(source.name)
    exact_name_matches: list[Path] = []
    parameter_matches: list[Path] = []
    for candidate_path in sorted((ROOT / "backtest_outputs" / "experiments").glob("*/config.json")):
        try:
            candidate = load_strategy_config(candidate_path)
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        if signature(candidate) != source_signature:
            continue
        parameter_matches.append(candidate_path.parent)
        if _normalise_display_name(candidate.name) == source_name:
            exact_name_matches.append(candidate_path.parent)
    # Deterministic earliest archive: it is the original reproducible result,
    # rather than a later rerun with identical parameters.
    return (exact_name_matches or parameter_matches or [None])[0]
    return None


def _provenance_reference_text(reference: object) -> str:
    """Use the global archive identity wherever a provenance reference is a strategy."""
    if not isinstance(reference, dict):
        return "未登记"
    label = _normalise_display_name(reference.get("label", "未登记"))
    experiment_dir = _provenance_reference_experiment(reference)
    if experiment_dir is not None:
        return _run_display_name(experiment_dir, label)
    return label


def _provenance_reference_html(reference: object) -> str:
    """Render a concise identified reference, retaining an alias's original title."""
    if not isinstance(reference, dict):
        return "未登记"
    label = _normalise_display_name(reference.get("label", "未登记"))
    path = str(reference.get("path", "")).strip()
    experiment_dir = _provenance_reference_experiment(reference)
    if experiment_dir is None:
        text = escape(label)
    else:
        archive_name = experiment_dir.name
        original = label
        try:
            original = _normalise_display_name(load_strategy_config(experiment_dir / "config.json").name)
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            pass
        alias = _experiment_display_names().get(archive_name)
        display_name = _run_display_name(experiment_dir, original)
        target = quote(experiment_dir.name)
        text = (
            f'<a class="provenance-strategy-link" href="/history?experiment={target}" target="_self">'
            f"{escape(display_name)}</a>"
        )
        if alias and _normalise_display_name(alias) != original:
            text += f"<br><span class='provenance-reference-original'>原名称：{escape(original)}</span>"
    if path:
        text += f"<br><span class='provenance-reference-path'>{escape(path)}</span>"
    return text


def _provenance_reference_link_html(reference: object) -> str:
    """Return a compact result-page link only for a recorded experiment archive."""
    if not isinstance(reference, dict):
        return ""
    experiment_dir = _provenance_reference_experiment(reference)
    if experiment_dir is None:
        return ""
    try:
        original = load_strategy_config(experiment_dir / "config.json").name
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        original = reference.get("label", "历史策略")
    return (
        f'<a class="provenance-strategy-link" href="/history?experiment={quote(experiment_dir.name)}" target="_self">'
        f"{escape(_run_display_name(experiment_dir, original))}</a>"
    )


def _render_workspace(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
    diagnostics: pd.DataFrame,
    result_name: str,
    config: DashboardStrategyConfig,
    experiment_dir: Path,
    current_signals: pd.DataFrame | None = None,
    latest_signal_config: DashboardStrategyConfig | None = None,
) -> None:
    display_result_name = _run_display_name(experiment_dir, result_name)
    st.markdown(
        f"<div class='section-head'><h2>研究工作区</h2><p>结果参数：{escape(display_result_name)} · 图表支持框选与滚轮缩放</p></div>",
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["交易表现", "传统净值", "收益归因", "错判诊断", "最新信号", "研究溯源", "参数快照"])
    with tabs[0]:
        _render_trading_tab(daily, strategy_metrics, benchmark_metrics)
    with tabs[1]:
        _render_nav_tab(daily, strategy_metrics, benchmark_metrics)
    with tabs[2]:
        _render_attribution_tab(daily)
    with tabs[3]:
        _render_diagnostics_tab(diagnostics, config.signal_frequency)
    with tabs[4]:
        latest_signals = current_signals if current_signals is not None and not current_signals.empty else signals
        display_signal_config = latest_signal_config or config
        refresh_col, _ = st.columns([1.45, 8.55])
        if refresh_col.button(
            "刷新当前信号",
            key=f"refresh_current_signal_{experiment_dir.name}",
            help="重新读取当前市场与供给数据，并使用该策略最后一期已保存参数生成信号；不会重新搜索或修改历史回测。",
        ):
            st.toast("已按当前数据和最后一期保存参数刷新执行信号。")
        if not latest_signals.empty:
            performance_latest = pd.to_datetime(signals.get("signal_date"), errors="coerce").max()
            current_latest = pd.to_datetime(latest_signals.get("signal_date"), errors="coerce").max()
            if pd.notna(performance_latest) and pd.notna(current_latest) and current_latest > performance_latest:
                st.caption(
                    f"当前可执行信号已更新至 {current_latest:%Y-%m-%d}；"
                    f"本页所选绩效区间截至 {performance_latest:%Y-%m-%d}。"
                )
        _render_latest_signal(
            latest_signals,
            factor_weights=_display_factor_weights(display_signal_config),
            factor_thresholds=display_signal_config.thresholds.as_dict(),
            signal_frequency=display_signal_config.signal_frequency,
        )
    with tabs[5]:
        _render_research_provenance(config, experiment_dir)
    with tabs[6]:
        _render_config_snapshot(config)
        st.caption(f"本次实验目录：{experiment_dir}")


def _display_factor_weights(config: DashboardStrategyConfig) -> dict[str, object]:
    provenance = config.research_provenance or {}
    research_weights = provenance.get("因子研究权重") if isinstance(provenance, dict) else None
    if isinstance(research_weights, dict):
        return research_weights
    expanded = provenance.get("扩展因子权重") if isinstance(provenance, dict) else None
    if isinstance(expanded, dict):
        return expanded
    # Early self-selected factor archives only retained their legacy eight
    # weights in config.json.  Their complete weights remain in the immutable
    # study snapshot, so read that exact route for display instead of showing a
    # valid factor score beside a fabricated zero weight.
    route = provenance.get("因子增加研究路线", {}) if isinstance(provenance, dict) else {}
    study_name = str(provenance.get("因子增加研究批次", "")).strip() if isinstance(provenance, dict) else ""
    if study_name and isinstance(route, dict):
        objective = str(route.get("目标", "")).strip()
        frequency = str(route.get("频率", "")).strip()
        factor_version = str(route.get("因子版本", "")).strip()
        if objective and frequency and factor_version:
            try:
                study_dir = ROOT / FACTOR_RESEARCH_DIR / study_name
                payload = json.loads(
                    (study_dir / f"{_factor_stem(factor_version, objective, frequency)}_最终配置.json").read_text(encoding="utf-8")
                )
                weights = payload.get("权重")
                if isinstance(weights, dict):
                    return weights
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                pass
    return config.weights.as_dict()


_PROVENANCE_GROUP_LABELS = {
    "weights": "权重",
    "thresholds": "阈值",
    "positions": "仓位 / 执行规则",
    "factor_windows": "因子窗口",
    "objective": "搜索目标",
    "backtest": "回测区间",
    "signal_frequency": "信号频率",
    "benchmark": "比较基准",
}
_PROVENANCE_FIELD_LABELS = {
    **WEIGHT_LABELS,
    "supply_low": "供给低分位", "supply_high": "供给高分位",
    "demand_low": "需求低分位", "demand_high": "需求高分位",
    "spread_low": "地方债利差低分位", "spread_high": "地方债利差高分位",
    "ncd_low": "NCD利差低分位", "ncd_high": "NCD利差高分位",
    "spread_change_bp": "利差变化阈值（BP）",
    "bullish_threshold": "看多分数阈值", "bearish_threshold": "看空分数阈值",
    "bullish_position": "看多仓位", "neutral_position": "中性仓位", "bearish_position": "看空仓位",
    "bearish_min_core_factors": "看空所需核心模块数",
    "bearish_require_supply_or_demand": "看空必须含供给或需求",
    "bearish_confirmation_periods": "看空连续确认期数",
    "take_profit_bp": "止盈阈值（BP）", "stop_loss_bp": "止损阈值（BP）",
    "start_date": "起始日期", "end_date": "结束日期",
}


def _provenance_value(value: object) -> str:
    if value is None:
        return "未设置"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _provenance_change_rows(changes: object) -> list[dict[str, str]]:
    """Flatten recorded before/after values into a compact, human-readable table."""
    if not isinstance(changes, dict):
        return []
    rows: list[dict[str, str]] = []
    for group, delta in changes.items():
        group_label = _PROVENANCE_GROUP_LABELS.get(str(group), str(group))
        if not isinstance(delta, dict) or "before" not in delta or "after" not in delta:
            rows.append({"字段": group_label, "原值": "未登记", "新值": _provenance_value(delta)})
            continue
        before, after = delta.get("before"), delta.get("after")
        if isinstance(before, dict) and isinstance(after, dict):
            keys = list(dict.fromkeys([*before.keys(), *after.keys()]))
            for key in keys:
                old, new = before.get(key), after.get(key)
                if old == new:
                    continue
                label = _PROVENANCE_FIELD_LABELS.get(str(key), str(key))
                rows.append({"字段": f"{group_label} · {label}", "原值": _provenance_value(old), "新值": _provenance_value(new)})
        elif before != after:
            rows.append({"字段": group_label, "原值": _provenance_value(before), "新值": _provenance_value(after)})
    return rows


def _provenance_steps_for_display(provenance: dict[str, object]) -> list[dict[str, object]]:
    """Collapse the two internal stages of an explicitly recorded joint search."""
    raw_steps = [dict(step) for step in provenance.get("steps", []) if isinstance(step, dict)]
    compact: list[dict[str, object]] = []
    index = 0
    while index < len(raw_steps):
        first = raw_steps[index]
        second = raw_steps[index + 1] if index + 1 < len(raw_steps) else None
        first_operation = str(first.get("operation", ""))
        second_operation = str(second.get("operation", "")) if second else ""
        second_note = str(second.get("note", "")) if second else ""
        is_joint = (
            second is not None
            and "权重搜索" in first_operation
            and second_operation == "阈值搜索"
            and ("联合搜索" in second_note or "联合搜索" in str(first.get("note", "")))
        )
        if is_joint:
            merged = dict(second)
            merged["operation"] = "联合搜索"
            merged["input"] = first.get("input", {})
            merged["output"] = second.get("output", {})
            merged["training_start"] = second.get("training_start") or first.get("training_start")
            merged["training_end"] = second.get("training_end") or first.get("training_end")
            merged["signal_frequency"] = second.get("signal_frequency") or first.get("signal_frequency")
            version = str(first.get("search_version", "")).strip()
            normalized_version = re.sub(r"^v(?=\d)", "V", version, flags=re.IGNORECASE)
            weight_version = normalized_version if "权重搜索" in normalized_version else f"{normalized_version} 权重搜索"
            objective_config = first.get("arguments", {}).get("objective_config") if isinstance(first.get("arguments"), dict) else None
            objective_name = None
            if isinstance(objective_config, dict):
                try:
                    objective = ObjectiveConfig(**objective_config)
                    objective_name = next((name for name, value in OBJECTIVES.items() if value == objective), None)
                except (TypeError, ValueError):
                    pass
            objective_suffix = f"（{objective_name}优先）" if objective_name else ""
            merged["search_version"] = f"{weight_version} + 阈值搜索{objective_suffix}" if version and version != "不适用" else f"V2 权重搜索 + 阈值搜索{objective_suffix}"
            merged["entrypoint"] = "common.combined_research.run_combined_search"
            merged["arguments"] = {"权重阶段": first.get("arguments", {}), "阈值阶段": second.get("arguments", {})}
            merged["changes"] = {**(first.get("changes") or {}), **(second.get("changes") or {})}
            merged["note"] = "联合搜索：先执行权重搜索，再执行阈值搜索；" + (second_note or str(first.get("note", "")))
            compact.append(merged)
            index += 2
            continue
        compact.append(first)
        index += 1
    return compact


def _render_research_provenance(config: DashboardStrategyConfig, experiment_dir: Path | None = None) -> None:
    provenance = display_provenance(config, ROOT, experiment_dir)
    if not provenance:
        st.info(MISSING_PROVENANCE)
        return
    st.markdown(_provenance_overview_html(provenance, config.name, experiment_dir), unsafe_allow_html=True)
    if provenance.get("earlier_history") == "未登记":
        st.caption(UNKNOWN_HISTORY)
    steps = _provenance_steps_for_display(provenance)
    with st.expander("步骤明细与复现依据", expanded=False):
        origin = provenance["origin"]
        st.markdown("**可复现起点**")
        st.markdown(_provenance_reference_html(origin), unsafe_allow_html=True)
        for index, step in enumerate(steps, 1):
            st.markdown(f"##### {index}. {escape(str(step['operation']))}", unsafe_allow_html=True)
            for label, reference in (("输入", step.get("input", {})), ("输出", step.get("output", {}))):
                st.markdown(
                    f"<div class='provenance-reference'><strong>{escape(label)}：</strong>{_provenance_reference_html(reference)}</div>",
                    unsafe_allow_html=True,
                )
            frequency = {"daily": "日频", "weekly": "周频"}.get(step.get("signal_frequency"), "未登记")
            if step.get("training_end"):
                st.caption(f"训练：{step.get('training_start') or '首个可用观察'} 至 {step['training_end']} · {frequency}")
            version = step.get("search_version", "未登记")
            if version != "不适用":
                st.text(f"版本：{version}")
            if step.get("note"):
                st.write(step["note"])
            change_rows = _provenance_change_rows(step.get("changes"))
            if change_rows:
                st.markdown("**具体调整（原值 → 新值）**")
                _render_theme_table(pd.DataFrame(change_rows), scrollable=True)
            st.text(f"执行入口：{step.get('entrypoint') or '未登记'}")
        st.markdown("##### 复现边界")
        for note in provenance.get("notes", []):
            st.write(note)
        st.markdown("##### 证据文件")
        for evidence in provenance.get("evidence", []):
            st.text(evidence)
        st.download_button(
            "下载完整溯源", json.dumps(provenance, ensure_ascii=False, indent=2),
            file_name=f"{config.name}_研究溯源.json", mime="application/json",
            icon=":material/download:",
        )


def _factor_lineage_summary(provenance: dict[str, object], step: dict[str, object]) -> str:
    """Describe the frozen factor universe in an auditable, compact form."""
    arguments = step.get("arguments", {}) if isinstance(step.get("arguments"), dict) else {}
    route = provenance.get("因子增加研究路线", {}) if isinstance(provenance.get("因子增加研究路线"), dict) else {}
    raw_enabled = (
        route.get("事前启用因子")
        or provenance.get("因子集合")
        or arguments.get("因子列")
        or []
    )
    if isinstance(raw_enabled, str):
        # Direct-to-queue records store human labels; the frozen config still
        # states the selected count, which is safer than reverse-guessing keys.
        enabled = []
        enabled_text = raw_enabled
    else:
        enabled = [str(value) for value in raw_enabled if str(value) in ALL_FACTOR_COLUMNS]
        enabled_text = ""
    enabled_set = set(enabled)
    unused = [column for column in ALL_FACTOR_COLUMNS if column not in enabled_set]
    count = len(enabled) if enabled else (len([part for part in enabled_text.split("、") if part.strip()]) or "未登记")
    parts = [f"已启用：{count}项"]
    if enabled:
        parts.append("使用：" + "、".join(FACTOR_LABELS[column] for column in enabled))
    if unused:
        parts.append("未使用：" + "、".join(FACTOR_LABELS[column] for column in unused))
    elif enabled_text:
        parts.append("因子集合：" + enabled_text)
    return "\n".join(parts)


def _provenance_overview_html(
    provenance: dict[str, object], current_name: str, experiment_dir: Path | None = None
) -> str:
    steps = _provenance_steps_for_display(provenance)
    origin = provenance["origin"]
    # Registration is the endpoint in the overview; the full ordered record stays in details.
    if steps and str(steps[-1].get("operation", "")).startswith("登记为基线"):
        current_name = steps[-1].get("output", {}).get("label", current_name)
        steps = steps[:-1]

    def config_label(label: str) -> tuple[str, str]:
        match = re.match(r"基线\d+", label)
        if not match:
            return label.replace("_", " · "), ""
        traits = [value for value in ("日频", "周频", "重收益", "重胜率", "激进看多") if value in label]
        return match.group(), " · ".join(traits)

    nodes: list[dict[str, object]] = [{
        "role": "可复现起点",
        "title": config_label(_provenance_reference_text(origin))[0],
        "subtitle": config_label(_provenance_reference_text(origin))[1],
        "reference": origin,
    }]
    windows = {(s.get("training_start"), s["training_end"]) for s in steps if s.get("training_end")}
    frequencies = {s.get("signal_frequency") for s in steps if s.get("signal_frequency")}
    search_count = sum("搜索" in str(s.get("operation", "")) for s in steps)
    node_number = 0

    def append_operation(title: str, subtitle: str = "") -> None:
        nonlocal node_number
        node_number += 1
        nodes.append({"role": f"{node_number:02d}", "title": title, "subtitle": subtitle, "reference": None})

    def append_final_backtest(reference: object, subtitle: str) -> None:
        nonlocal node_number
        node_number += 1
        has_linked_strategy = _provenance_reference_experiment(reference) is not None
        nodes.append({
            "role": f"{node_number:02d} · 最终回测",
            "title": _provenance_reference_text(reference) if has_linked_strategy else "最终回测",
            "subtitle": subtitle,
            "reference": reference if has_linked_strategy else None,
        })

    for step in steps:
        operation = str(step.get("operation", "研究操作"))
        subtitle = ""
        arguments = step.get("arguments", {}) if isinstance(step.get("arguments"), dict) else {}
        if "权重搜索" in operation:
            cap = re.search(r"上限\s*(\d+)", f"{step.get('search_version', '')} {step.get('note', '')}")
            subtitle = f"模块上限 {cap.group(1)}" if cap else "优化因子权重"
            title = "搜索"
        elif operation == "联合搜索":
            title, subtitle = "搜索", f"联合搜索 · {step.get('search_version') or 'V2 权重搜索 + 阈值搜索'}"
        elif operation == "阈值搜索":
            title, subtitle = "搜索", "阈值搜索 · 固定权重 · 优化阈值"
        elif operation == "滚动定参":
            title = "滚动定参"
            rolling_parts = [
                str(step.get("search_version", "")).strip(),
                str(arguments.get("训练窗口", "")).strip(),
                str(arguments.get("最低训练长度", "")).strip(),
                str(arguments.get("定参频率", "")).strip(),
            ]
            subtitle = " · ".join(part for part in rolling_parts if part and part != "未登记")
        elif operation == "最终回测":
            title, subtitle = operation, "使用前面已经选出的参数回测；这一阶段不再搜索"
        else:
            title = operation
        if len(windows) > 1 and step.get("training_end"):
            subtitle += f" · {step.get('training_start') or '起始'} 至 {step['training_end']}"
        if len(frequencies) > 1:
            subtitle += f" · { {'daily': '日频', 'weekly': '周频'}.get(step.get('signal_frequency'), '频率未登记')}"
        # A strategy is shown exactly where it becomes a frozen result: the
        # final-backtest node.  Search and rolling nodes are operations, not
        # separately runnable strategy results, so repeating the preceding
        # strategy there made the lineage misleading.
        if operation == "最终回测":
            append_final_backtest(step.get("output", {}), subtitle)
            continue

        is_factor_operation = "因子" in operation or bool(arguments.get("因子集合"))
        if is_factor_operation:
            append_operation("因子研究", _factor_lineage_summary(provenance, step))
            if operation == "滚动定参":
                append_operation(title, subtitle)
            continue
        append_operation(title, subtitle)
    current_display = _run_display_name(experiment_dir, current_name) if experiment_dir else current_name
    nodes.append({
        "role": "当前策略",
        "title": config_label(current_display)[0],
        "subtitle": config_label(current_display)[1],
        "reference": {"label": current_name, "path": relative_path(ROOT, experiment_dir / "config.json")} if experiment_dir else None,
    })
    context = []
    if len(windows) == 1:
        start, end = next(iter(windows))
        context.append(("训练区间", f"{start or '首个可用观察'} → {end}"))
    if len(frequencies) == 1:
        context.append(("信号频率", {"daily": "日频", "weekly": "周频"}.get(next(iter(frequencies)), "未登记")))
    if search_count:
        context.append(("研究过程", f"{search_count} 次搜索"))
    summary = "".join(f"<span>{escape(label)}<strong>{escape(value)}</strong></span>" for label, value in context)
    items = ""
    for node in nodes:
        role, title, subtitle = (str(node["role"]), str(node["title"]), str(node["subtitle"]))
        reference_link = _provenance_reference_link_html(node.get("reference"))
        # Render the strategy itself as the prominent clickable item.  The old
        # layout put the link below the generic operation title, which hid the
        # strategy identity and made several nodes appear to have no ID.
        title_html = reference_link or escape(title)
        items += (
            f"<li><span class='provenance-role'>{escape(role)}</span>"
            f"<div class='provenance-title'>{title_html}</div>"
            f"<div class='provenance-subtitle'>{escape(subtitle)}</div></li>"
        )
    return (f"<section class='provenance-overview' aria-label='研究溯源流程'>"
            f"<div class='provenance-context'>{summary}</div>"
            f"<ol class='provenance-chain is-long'>{items}</ol></section>")


def _render_trading_tab(
    daily: pd.DataFrame,
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
) -> None:
    st.markdown("#### 收益率与实际持仓计价区间")
    st.caption("线段展示每笔仓位真正计入资本利得的收益率区间：左端为上一交易日收盘估值，右端为仓位生效日收盘估值。红色为盈利、绿色为亏损；不再把下一次信号日误画成平仓日。")
    _render_interactive_chart(_yield_trade_signal_chart(daily), key="yield_trade_signals")
    st.markdown("#### 资本利得交易曲线")
    st.caption("交易从非零仓位开始，归零或反向时结束；同方向加减仓仍属于同一笔。BP按 -仓位 × YTM变化BP 统计，不乘久期、不包含carry。")
    _render_interactive_chart(_capital_bp_chart(daily), key="capital_gain_bp")
    st.markdown("#### 逐笔资本利得")
    _render_interactive_chart(_capital_trade_chart(daily), key="capital_gain_trades")
    rows = []
    metric_specs = [
        ("累计资本利得", "capital_gain_total_bp", "BP"),
        ("交易笔数", "capital_gain_trade_count", "笔"),
        ("已平仓交易", "capital_gain_closed_trade_count", "笔"),
        ("盈利笔数", "capital_gain_winning_trades", "笔"),
        ("亏损笔数", "capital_gain_losing_trades", "笔"),
        ("逐笔胜率", "capital_gain_trade_win_rate", "%"),
        ("平均单笔", "capital_gain_avg_trade_bp", "BP"),
        ("平均盈利", "capital_gain_avg_win_bp", "BP"),
        ("平均亏损", "capital_gain_avg_loss_bp", "BP"),
        ("盈亏比", "capital_gain_profit_loss_ratio", "x"),
        ("最佳交易", "capital_gain_best_trade_bp", "BP"),
        ("最差交易", "capital_gain_worst_trade_bp", "BP"),
        ("平均每笔持有时间", "capital_gain_avg_holding_days", "交易日"),
        ("最长单笔持有时间", "capital_gain_max_holding_days", "交易日"),
        ("最大回撤", "capital_gain_max_drawdown_bp", "BP"),
        ("最长连续亏损", "capital_gain_longest_losing_streak", "笔"),
        ("当前未平仓交易", "capital_gain_open_trade_count", "笔"),
        ("未平仓浮动资本利得", "capital_gain_open_trade_bp", "BP"),
    ]
    for label, key, unit in metric_specs:
        strategy_value = strategy_metrics.get(key)
        benchmark_value = benchmark_metrics.get(key)
        if unit == "%":
            strategy_text, benchmark_text = _pct(strategy_value), _pct(benchmark_value)
        elif key == "capital_gain_avg_holding_days":
            strategy_text = "" if strategy_value is None else f"{float(strategy_value):.1f} {unit}"
            benchmark_text = "" if benchmark_value is None else f"{float(benchmark_value):.1f} {unit}"
        elif unit in {"笔", "交易日"}:
            strategy_text = "" if strategy_value is None else f"{int(strategy_value)} {unit}"
            benchmark_text = "" if benchmark_value is None else f"{int(benchmark_value)} {unit}"
        elif unit == "x":
            strategy_text = "" if strategy_value is None else f"{float(strategy_value):.2f}"
            benchmark_text = "" if benchmark_value is None else f"{float(benchmark_value):.2f}"
        else:
            strategy_text = "" if strategy_value is None else f"{float(strategy_value):.2f} {unit}"
            benchmark_text = "" if benchmark_value is None else f"{float(benchmark_value):.2f} {unit}"
        rows.append({"交易指标": label, "策略": strategy_text, "条件基准": benchmark_text})
    _render_theme_table(pd.DataFrame(rows), numeric_columns={"策略", "条件基准"})
    trades = capital_gain_trade_table(daily)
    if not trades.empty:
        trade_display = trades.rename(
            columns={
                "trade_id": "交易编号", "entry_date": "开仓日期", "exit_date": "平仓日期", "mark_date": "估值日期",
                "direction": "方向", "entry_position": "开仓仓位", "average_abs_position": "平均绝对仓位",
                "holding_days": "持有交易日", "strategy_capital_bp": "策略资本利得_BP",
                "benchmark_capital_bp": "同期条件基准资本利得_BP", "capital_excess_bp": "资本利得超额_BP", "status": "状态",
            }
        )
        trade_display = trade_display[["交易编号", "开仓日期", "平仓日期", "估值日期", "方向", "开仓仓位", "平均绝对仓位", "持有交易日", "策略资本利得_BP", "同期条件基准资本利得_BP", "资本利得超额_BP", "状态"]]
        for column in ["开仓日期", "平仓日期", "估值日期"]:
            trade_display[column] = pd.to_datetime(trade_display[column]).dt.strftime("%Y-%m-%d").fillna("")
        for column in ["策略资本利得_BP", "同期条件基准资本利得_BP", "资本利得超额_BP"]:
            trade_display[column] = pd.to_numeric(trade_display[column], errors="coerce").map(lambda value: f"{value:.2f} BP")
        st.markdown("#### 开平仓交易明细")
        _render_theme_table(trade_display, numeric_columns={"开仓仓位", "平均绝对仓位", "持有交易日", "策略资本利得_BP", "同期条件基准资本利得_BP", "资本利得超额_BP"}, scrollable=True, wide=True)


def _render_history_page() -> None:
    st.markdown(
        "<section class='hero-shell'><div><p class='eyebrow'>归档结果 / 不重新计算</p><h1 class='hero-title'>历史实验</h1><p class='hero-copy'>每条记录保留当时的配置、数据指纹、资本利得交易指标和完整HTML。查看归档不会使用当前数据重算。</p></div></section>",
        unsafe_allow_html=True,
    )
    _render_experiment_history(show_report=True)


def _factor_studies() -> list[Path]:
    root = ROOT / FACTOR_RESEARCH_DIR
    if not root.exists():
        return []
    studies = sorted(
        [path for path in root.iterdir() if path.is_dir() and (path / "研究清单.json").exists() and (path / "静态研究汇总.csv").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return studies


def _factor_study_label(study_dir: Path) -> str:
    manifest = _load_factor_manifest(study_dir)
    scope = str(manifest.get("研究范围", "因子对照"))
    training_end = str(manifest.get("训练截止日", "未登记"))
    return f"研究批次 {study_dir.name} · {scope} · 训练至 {training_end}"


def _load_factor_manifest(study_dir: Path) -> dict[str, object]:
    return json.loads((study_dir / "研究清单.json").read_text(encoding="utf-8"))


def _factor_stem(factor_version: str, objective: str, frequency: str) -> str:
    return f"{factor_version}_{objective}_{frequency}"


def _factor_version_label(factor_version: str, enabled_count: int | None = None) -> str:
    if factor_version == "原始因子":
        return "原始8因子"
    if factor_version == "扩展因子":
        return "扩展15因子"
    if factor_version == "自选因子":
        return f"自选{enabled_count}因子" if enabled_count is not None else "自选因子集合"
    return factor_version


def _is_factor_comparison_universe(enabled_columns: tuple[str, ...]) -> bool:
    selected = set(enabled_columns)
    return set(BASE_FACTOR_COLUMNS).issubset(selected) and bool(selected.intersection(EXPANDED_FACTOR_COLUMNS))


def _factor_objective_copy(objective: str) -> str:
    copies = {
        "收益": "收益优先主要比较累计资本利得，同时保留少量超额收益和已平仓胜率权重。排序公式：1.00 × 累计资本利得 + 0.25 × 资本利得超额 + 10 × 已平仓交易胜率。",
        "胜率": "胜率优先主要比较已平仓交易中盈利交易的比例，同时约束收益和回撤。排序公式：0.20 × 累计资本利得 + 0.25 × 资本利得超额 + 100 × 已平仓交易胜率 + 1.00 × 最大回撤。",
        "综合": "综合评价在累计资本利得、已平仓交易胜率和回撤之间取中间权重。排序公式：0.60 × 累计资本利得 + 0.25 × 资本利得超额 + 50 × 已平仓交易胜率 + 0.50 × 最大回撤。",
    }
    suffix = " 最大回撤记录为负数，因此胜率优先和综合评价中的正系数会扣减回撤更大的候选。" if objective in {"胜率", "综合"} else ""
    return copies.get(objective, "不同搜索目标只改变候选排序权重，不改变信号或仓位计算方式。") + suffix


def _factor_summary(study_dir: Path, mode: str) -> pd.DataFrame:
    if mode == "滚动样本外":
        path = study_dir / "滚动定参" / "滚动定参汇总.csv"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path, encoding="utf-8-sig")
        return frame.rename(columns={
            "样本外累计资本利得_BP": "资本利得_BP",
            "样本外交易胜率": "交易胜率",
            "样本外已平仓交易数": "已平仓交易数",
            "样本外最大回撤_BP": "最大回撤_BP",
        })
    frame = pd.read_csv(study_dir / "静态研究汇总.csv", encoding="utf-8-sig")
    if "训练目标函数" not in frame.columns:
        required = {
            "训练累计资本利得_BP", "训练资本利得超额_BP", "训练交易胜率", "训练最大回撤_BP",
        }
        if required.issubset(frame.columns):
            def legacy_training_objective(row: pd.Series) -> float:
                objective = OBJECTIVES[str(row["目标"])]
                return float(
                    objective.capital_gain_bp_weight * float(row["训练累计资本利得_BP"])
                    + objective.capital_gain_excess_bp_weight * float(row["训练资本利得超额_BP"])
                    + objective.capital_trade_win_rate_weight * float(row["训练交易胜率"])
                    + objective.capital_gain_drawdown_bp_penalty * float(row["训练最大回撤_BP"])
                )

            frame["训练目标函数"] = frame.apply(legacy_training_objective, axis=1)
    return frame.rename(columns={
        "样本外累计资本利得_BP": "资本利得_BP",
        "样本外交易胜率": "交易胜率",
        "样本外已平仓交易数": "已平仓交易数",
        "样本外最大回撤_BP": "最大回撤_BP",
    })


def _factor_delta_html(value: object, suffix: str = "") -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "-"
    tone = "positive" if float(numeric) >= 0 else "negative"
    sign = "+" if float(numeric) > 0 else ""
    return f'<span class="{tone}">{sign}{float(numeric):.2f}{suffix}</span>'


def _factor_strategy_detail_link(study_dir: Path, objective: str, frequency: str, mode: str, factor_version: str) -> str:
    """Legacy detail link retained only for old bookmarked URLs."""
    query = {
        "factor_study": study_dir.name,
        "factor_route": f"{objective}_{frequency}",
        "factor_mode": mode,
        "factor_version": factor_version,
    }
    return "/history?" + "&".join(f"{quote(key)}={quote(str(value))}" for key, value in query.items())


@st.cache_data(show_spinner=False)
def _factor_archive_routes(study_name: str) -> dict[tuple[str, str, str], str]:
    """Find the frozen history archive for each static factor-research route.

    A factor study is an input/output workspace, while an archive is the
    independently viewable strategy result.  Keep this lookup provenance-based
    so that it survives a browser refresh and does not depend on session state.
    """
    routes: dict[tuple[str, str, str], str] = {}
    experiment_root = ROOT / "backtest_outputs" / "experiments"
    if not experiment_root.exists():
        return routes
    for experiment_dir in experiment_root.iterdir():
        config_path = experiment_dir / "config.json"
        if not experiment_dir.is_dir() or not config_path.exists():
            continue
        try:
            provenance = load_strategy_config(config_path).research_provenance or {}
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        if provenance.get("研究类型") != "因子增加单一策略归档":
            continue
        if str(provenance.get("因子增加研究批次", "")) != study_name:
            continue
        route = provenance.get("因子增加研究路线", {})
        if not isinstance(route, dict):
            continue
        objective = str(route.get("目标", ""))
        frequency = str(route.get("频率", ""))
        factor_version = str(route.get("因子版本", ""))
        if objective and frequency and factor_version:
            routes[(objective, frequency, factor_version)] = experiment_dir.name
    return routes


@st.cache_data(show_spinner=False)
def _factor_rolling_archive_routes(study_name: str) -> dict[tuple[str, str, str], str]:
    """Find independent history archives made from saved dynamic executions."""
    routes: dict[tuple[str, str, str], str] = {}
    experiment_root = ROOT / "backtest_outputs" / "experiments"
    if not experiment_root.exists():
        return routes
    for experiment_dir in experiment_root.iterdir():
        config_path = experiment_dir / "config.json"
        if not experiment_dir.is_dir() or not config_path.exists():
            continue
        try:
            provenance = load_strategy_config(config_path).research_provenance or {}
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        if provenance.get("研究类型") != "因子增加滚动单一路线归档":
            continue
        if str(provenance.get("因子增加研究批次", "")) != study_name:
            continue
        route = provenance.get("因子增加研究路线", {})
        if not isinstance(route, dict):
            continue
        objective = str(route.get("目标", ""))
        frequency = str(route.get("频率", ""))
        factor_version = str(route.get("因子版本", ""))
        if objective and frequency and factor_version:
            routes[(objective, frequency, factor_version)] = experiment_dir.name
    return routes


def _factor_result_link(study_dir: Path, objective: str, frequency: str, mode: str, factor_version: str) -> str:
    """Every visible route resolves to the normal immutable history result."""
    if mode == "静态样本外":
        archive_name = _factor_archive_routes(study_dir.name).get((objective, frequency, factor_version))
    else:
        archive_name = _factor_rolling_archive_routes(study_dir.name).get((objective, frequency, factor_version))
    if archive_name:
        return f"/history?experiment={quote(archive_name)}"
    # No research-workspace fallback: a missing archive must be visible rather
    # than silently sending the user to a different page.
    return "/history"


def _factor_result_label(study_dir: Path, objective: str, frequency: str, mode: str, factor_version: str, label: str) -> str:
    """Use the immutable F identifier wherever a static route is presented."""
    archive_name = (
        _factor_archive_routes(study_dir.name).get((objective, frequency, factor_version))
        if mode == "静态样本外"
        else _factor_rolling_archive_routes(study_dir.name).get((objective, frequency, factor_version))
    )
    if archive_name:
        archive_id = existing_short_archive_id(ROOT, archive_id_category('F'), archive_name)
        return f"[{archive_id}] {label}" if archive_id else label
    return label


def _factor_comparison_table(study_dir: Path, summary: pd.DataFrame, mode: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    values = ["资本利得_BP", "交易胜率", "已平仓交易数", "最大回撤_BP"]
    has_training_objective = mode == "静态样本外" and "训练目标函数" in summary.columns
    if has_training_objective:
        values.append("训练目标函数")
    pivot = summary.pivot(index=["目标", "频率"], columns="因子版本", values=values)
    rows: list[dict[str, object]] = []
    for (objective, frequency), values_row in pivot.iterrows():
        frequency_code = "daily" if frequency == "日频" else "weekly"
        raw = {key: values_row.get((key, "原始因子")) for key in values}
        expanded = {key: values_row.get((key, "扩展因子")) for key in values}
        row = {
            "策略路线": f"{objective}优先 · {frequency}",
            "原始8因子 · 样本外收益 / 胜率 / 回撤": f"{float(raw['资本利得_BP']):.2f} BP / {_pct(raw['交易胜率'])} / {float(raw['最大回撤_BP']):.2f} BP",
            "扩展15因子 · 样本外收益 / 胜率 / 回撤": f"{float(expanded['资本利得_BP']):.2f} BP / {_pct(expanded['交易胜率'])} / {float(expanded['最大回撤_BP']):.2f} BP",
            "样本外变化 · 收益 / 胜率": (
                f"{_factor_delta_html(float(expanded['资本利得_BP']) - float(raw['资本利得_BP']), ' BP')} / "
                f"{_factor_delta_html((float(expanded['交易胜率']) - float(raw['交易胜率'])) * 100.0, 'pp')}"
            ),
            "查看单策略结果": (
                '<div class="factor-detail-actions">'
                f'<a class="factor-detail-link" href="{_factor_result_link(study_dir, objective, frequency_code, mode, "原始因子")}" target="_blank">{_factor_result_label(study_dir, objective, frequency_code, mode, "原始因子", "原始8因子")}</a>'
                f'<a class="factor-detail-link" href="{_factor_result_link(study_dir, objective, frequency_code, mode, "扩展因子")}" target="_blank">{_factor_result_label(study_dir, objective, frequency_code, mode, "扩展因子", "扩展15因子")}</a>'
                "</div>"
            ),
        }
        if has_training_objective:
            original_objective = float(values_row.get(("训练目标函数", "原始因子")))
            expanded_objective = float(values_row.get(("训练目标函数", "扩展因子")))
            row["搜索期目标分 · 原始 → 扩展"] = (
                f"{original_objective:.2f} → {expanded_objective:.2f} "
                f"({_factor_delta_html(expanded_objective - original_objective)})"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _factor_daily_path(study_dir: Path, stem: str, mode: str) -> Path:
    parent = study_dir / "滚动定参" if mode == "滚动样本外" else study_dir
    suffix = "样本外日度.csv"
    return parent / f"{stem}_{suffix}"


def _factor_capital_comparison_chart(study_dir: Path, objective: str, frequency: str, mode: str) -> go.Figure | None:
    figure = go.Figure()
    for version, color in (("原始因子", INK), ("扩展因子", GOLD)):
        path = _factor_daily_path(study_dir, _factor_stem(version, objective, frequency), mode)
        if not path.exists():
            continue
        daily = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
        figure.add_trace(go.Scatter(
            x=daily["date"], y=pd.to_numeric(daily["strategy_capital_cum_bp"], errors="coerce"),
            mode="lines", name=version, line={"color": color, "width": 2.5},
        ))
    if not figure.data:
        return None
    figure.update_layout(
        height=360, margin={"l": 8, "r": 8, "t": 18, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(251,250,246,.45)",
        font={"family": "Geist, Microsoft YaHei, sans-serif", "color": INK},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False, linecolor=GRID)
    figure.update_yaxes(title="累计资本利得 BP", gridcolor="rgba(217,221,216,.72)", zerolinecolor=GRID)
    return figure


def _factor_single_baseline_controls() -> DashboardStrategyConfig:
    selected_path = _strategy_config_picker(
        "策略基线",
        "factor_expansion_baseline",
        help_text="可选择基础配置、保留基线、搜索结果或任一历史实验保存的配置。",
        include_factor_archives=True,
    )
    base = load_strategy_config(selected_path)
    token = f"{selected_path.parent.name}_{selected_path.stem}"
    st.caption(
        f"当前选择：{base.name}。研究会继承这套配置的因子判定阈值、仓位与执行规则、因子窗口；"
        "原有因子权重会作为候选起点，但不会锁定；原始8因子和扩展15因子会分别重新搜索权重。"
    )
    with st.expander("查看或调整继承的策略条件", expanded=False):
        st.caption("这里调整的是搜索前提。各因子权重和总分入场阈值由研究自动搜索，不需要在这里填写。")
        threshold_tab, position_tab, window_tab = st.tabs(["因子判定阈值", "仓位与执行", "因子窗口"])
        with threshold_tab:
            st.caption("这些阈值把原始数据转换成利多、中性或利空判断；它们不同于搜索自动选择的总分入场阈值。")
            threshold_columns = st.columns(3)
            threshold_specs = [
                ("supply_low", "供给低分位", 0.0, 50.0, 5.0), ("supply_high", "供给高分位", 50.0, 100.0, 5.0),
                ("demand_low", "需求低分位", 0.0, 50.0, 5.0), ("demand_high", "需求高分位", 50.0, 100.0, 5.0),
                ("spread_low", "地方债利差低分位", 0.0, 50.0, 5.0), ("spread_high", "地方债利差高分位", 50.0, 100.0, 5.0),
                ("ncd_low", "NCD利差低分位", 0.0, 50.0, 5.0), ("ncd_high", "NCD利差高分位", 50.0, 100.0, 5.0),
                ("spread_change_bp", "利差变化阈值 BP", 0.5, 10.0, 0.5),
            ]
            threshold_values = {
                key: threshold_columns[index % 3].number_input(
                    label, minimum, maximum, float(getattr(base.thresholds, key)), step,
                    key=f"factor_base_{token}_threshold_{key}",
                )
                for index, (key, label, minimum, maximum, step) in enumerate(threshold_specs)
            }
        with position_tab:
            position_columns = st.columns(3)
            bullish_position = position_columns[0].number_input("看多仓位", -1.0, 1.5, float(base.positions.bullish_position), 0.1, key=f"factor_base_{token}_bullish_position")
            neutral_position = position_columns[1].number_input("中性仓位", -1.0, 1.5, float(base.positions.neutral_position), 0.1, key=f"factor_base_{token}_neutral_position")
            bearish_position = position_columns[2].number_input("看空仓位", -1.5, 1.0, float(base.positions.bearish_position), 0.1, key=f"factor_base_{token}_bearish_position")
            min_modules = position_columns[0].selectbox("看空所需核心利空模块数", [0, 1, 2, 3, 4], index=int(base.positions.bearish_min_core_factors), key=f"factor_base_{token}_min_modules")
            require_supply = position_columns[1].toggle("看空必须包含供给或需求利空", value=bool(base.positions.bearish_require_supply_or_demand), key=f"factor_base_{token}_require_supply")
            confirmations = position_columns[2].selectbox("看空连续确认期数", [1, 2, 3], index=int(base.positions.bearish_confirmation_periods) - 1, key=f"factor_base_{token}_confirmations")
            take_profit = position_columns[0].number_input("止盈阈值 BP", 0.0, 100.0, float(base.positions.take_profit_bp), 0.5, key=f"factor_base_{token}_take_profit")
            stop_loss = position_columns[1].number_input("止损阈值 BP", 0.0, 100.0, float(base.positions.stop_loss_bp), 0.5, key=f"factor_base_{token}_stop_loss")
        with window_tab:
            st.caption("窗口表示计算因子历史分位时向前观察的月份数。")
            window_columns = st.columns(3)
            supply_months = window_columns[0].number_input("供给窗口 月", 3, 60, int(base.factor_windows.supply_months), 3, key=f"factor_base_{token}_supply_months")
            bank_months = window_columns[1].number_input("银行需求窗口 月", 3, 60, int(base.factor_windows.bank_months), 3, key=f"factor_base_{token}_bank_months")
            valuation_months = window_columns[2].number_input("估值窗口 月", 3, 60, int(base.factor_windows.valuation_months), 3, key=f"factor_base_{token}_valuation_months")
    positions = DashboardPositionPolicy(
        bullish_threshold=base.positions.bullish_threshold,
        bearish_threshold=base.positions.bearish_threshold,
        bullish_position=float(bullish_position),
        neutral_position=float(neutral_position),
        bearish_position=float(bearish_position),
        bearish_min_core_factors=int(min_modules),
        bearish_require_supply_or_demand=int(require_supply),
        bearish_confirmation_periods=int(confirmations),
        take_profit_bp=float(take_profit),
        stop_loss_bp=float(stop_loss),
    )
    return DashboardStrategyConfig(
        name=base.name,
        weights=base.weights,
        thresholds=DashboardThresholds(**threshold_values),
        positions=positions,
        objective=base.objective,
        factor_windows=FactorWindowConfig(
            supply_months=int(supply_months), bank_months=int(bank_months), valuation_months=int(valuation_months)
        ),
        benchmark_id=base.benchmark_id,
        signal_frequency=base.signal_frequency,
        research_provenance=base.research_provenance,
        source_config_path=str(selected_path.resolve()),
    )


_FACTOR_GROUP_SELECTION_LABELS = {
    "supply": "供给",
    "bank": "需求",
    "valuation": "估值",
    "sentiment": "情绪",
    "interaction": "供需交互",
}
_FACTOR_PRESELECTION_KEY = "factor_expansion_enabled_factor_columns"


def _factor_preselection_state() -> tuple[tuple[str, ...], tuple[str, ...], set[str]]:
    """Read the selected factors from one shared, lightweight session value."""
    enabled = set(st.session_state.get(_FACTOR_PRESELECTION_KEY, ALL_FACTOR_COLUMNS)).intersection(ALL_FACTOR_COLUMNS)
    enabled_columns = tuple(column for column in ALL_FACTOR_COLUMNS if column in enabled)
    disabled_columns = tuple(column for column in ALL_FACTOR_COLUMNS if column not in enabled)
    enabled_groups = {
        group for group, members in FACTOR_GROUPS.items()
        if any(column in enabled for column in members)
    }
    return enabled_columns, disabled_columns, enabled_groups


def _set_factor_preselection(enabled: set[str]) -> None:
    st.session_state[_FACTOR_PRESELECTION_KEY] = [
        factor for factor in ALL_FACTOR_COLUMNS if factor in enabled
    ]


def _sync_factor_group_toggle(group: str) -> None:
    """Apply a category segmented-control change before the fragment reruns."""
    enabled = set(st.session_state.get(_FACTOR_PRESELECTION_KEY, ALL_FACTOR_COLUMNS)).intersection(ALL_FACTOR_COLUMNS)
    columns = FACTOR_GROUPS[group]
    toggle_key = f"factor_group_toggle_{group}"
    is_selected = bool(st.session_state.get(toggle_key))
    if is_selected:
        enabled.update(columns)
        st.session_state[f"factor_pills_{group}"] = list(columns)
    else:
        enabled.difference_update(columns)
        st.session_state[f"factor_pills_{group}"] = []
    _set_factor_preselection(enabled)


def _sync_factor_group_members(group: str) -> None:
    """Apply individual factor changes and keep the category control in sync."""
    enabled = set(st.session_state.get(_FACTOR_PRESELECTION_KEY, ALL_FACTOR_COLUMNS)).intersection(ALL_FACTOR_COLUMNS)
    columns = FACTOR_GROUPS[group]
    selected = set(st.session_state.get(f"factor_pills_{group}", [])).intersection(columns)
    enabled.difference_update(columns)
    enabled.update(selected)
    st.session_state[f"factor_group_toggle_{group}"] = [_FACTOR_GROUP_SELECTION_LABELS[group]] if selected else []
    _set_factor_preselection(enabled)


@st.fragment
def _render_factor_preselection_controls() -> None:
    """Keep factor picking local so each click does not rebuild research results."""
    if _FACTOR_PRESELECTION_KEY not in st.session_state:
        st.session_state[_FACTOR_PRESELECTION_KEY] = list(ALL_FACTOR_COLUMNS)
    enabled_set = set(st.session_state[_FACTOR_PRESELECTION_KEY]).intersection(ALL_FACTOR_COLUMNS)
    st.markdown(
        "<p class='factor-toggle-intro'>点击大类可整体启用或停用；点击因子可单独切换。浅绿表示纳入本轮搜索，白底表示事前停用。启用后，搜索仍可将其权重设为 0。</p>",
        unsafe_allow_html=True,
    )
    for group, columns in FACTOR_GROUPS.items():
        active_count = sum(column in enabled_set for column in columns)
        group_col, factor_col = st.columns([1.15, 8.85], gap="small", vertical_alignment="center")
        with group_col:
            st.segmented_control(
                f"{_FACTOR_GROUP_SELECTION_LABELS[group]}整类切换",
                [_FACTOR_GROUP_SELECTION_LABELS[group]],
                selection_mode="multi",
                default=[_FACTOR_GROUP_SELECTION_LABELS[group]] if active_count else [],
                key=f"factor_group_toggle_{group}",
                label_visibility="collapsed",
                help=f"点击整体启用或停用{_FACTOR_GROUP_SELECTION_LABELS[group]}下的全部因子。",
                on_change=_sync_factor_group_toggle,
                args=(group,),
            )
        with factor_col:
            st.segmented_control(
                f"{_FACTOR_GROUP_SELECTION_LABELS[group]}因子",
                list(columns),
                selection_mode="multi",
                default=[column for column in columns if column in enabled_set],
                format_func=lambda column: FACTOR_LABELS[column].partition("|")[2],
                key=f"factor_pills_{group}",
                label_visibility="collapsed",
                help="点击因子名称切换是否纳入本轮搜索。",
                on_change=_sync_factor_group_members,
                args=(group,),
            )
    enabled_columns, disabled_columns, enabled_groups = _factor_preselection_state()
    if not enabled_columns:
        st.error("请至少启用一个因子。")
    elif not _is_factor_comparison_universe(enabled_columns):
        st.caption(
            "当前将运行自选因子集合研究：只搜索当前启用的因子，不生成原始8因子对照。"
            "这适用于只看一个大类、单个新增因子或任意自选子集。"
        )
    if len(enabled_groups) == 1:
        only_group = next(iter(enabled_groups))
        st.caption(
            f"当前只启用一个大类（{_FACTOR_GROUP_SELECTION_LABELS.get(only_group, only_group)}）。"
            "该大类在V2搜索中允许占满100分，可用于只观察一个因子或一类因子的实验；后续静态搜索和滚动定参均可执行。"
        )
    elif disabled_columns:
        st.caption("事前停用：" + "、".join(FACTOR_LABELS[column] for column in disabled_columns))
    else:
        st.caption("当前15个因子均参与搜索；搜索仍可把任一因子权重设为0。")


def _render_factor_expansion_launch_panel() -> None:
    st.markdown(
        "<div class='section-head'><h2>因子增加研究</h2><p>完整原始8因子加新增因子时做对照；任意子集则作为一条独立策略搜索。</p></div>",
        unsafe_allow_html=True,
    )
    available_start, available_end = _benchmark_date_bounds()
    scope = st.segmented_control(
        "研究范围", ["全量六路线比较", "单一策略对测试"], default="全量六路线比较", key="factor_expansion_scope_kind",
        help="全量比较运行收益、胜率、综合与日频、周频共六组；单一策略对只运行所选目标和频率。实际是否生成原始/扩展对照，由事前启用因子决定。",
    )
    if scope == "全量六路线比较":
        st.caption("会运行收益优先、胜率优先、综合评价三种搜索目标，并分别测试日频和周频，共6组策略路线。")
    else:
        st.caption("只运行下方选定基线、搜索目标和信号频率对应的1组策略路线。")
    left, right = st.columns([1, 1])
    with left:
        training_end = st.date_input(
            "训练截止日", value=min(pd.Timestamp("2025-01-01").date(), available_end),
            min_value=available_start, max_value=available_end, key="factor_expansion_training_end",
        )
    with right:
        run_scope = st.segmented_control(
            "执行范围", ["仅静态研究", "直接加入滚动定参队列"], default="仅静态研究", key="factor_expansion_scope"
        )
    if run_scope == "仅静态研究":
        st.caption("仅执行一次因子权重搜索和总分入场阈值搜索；参数选定后，在训练截止日之后的样本外区间保持不变。")
    else:
        st.caption("不先生成静态 F 研究归档。当前因子集合、目标、频率和执行规则会直接冻结并加入滚动任务队列；每期从训练窗口重新搜索。")
    selected_objectives: tuple[str, ...] = tuple(OBJECTIVES)
    selected_frequencies: tuple[str, ...] = ("daily", "weekly")
    factor_base: DashboardStrategyConfig | None = None
    if scope == "单一策略对测试":
        factor_base = _factor_single_baseline_controls()
        objective_col, frequency_col = st.columns(2)
        with objective_col:
            objective_name = st.segmented_control("搜索目标", ["收益", "胜率", "综合"], default="收益", key="factor_expansion_single_objective")
        with frequency_col:
            frequency_name = st.segmented_control(
                "信号频率", ["日频", "周频"],
                default="日频" if factor_base.signal_frequency == "daily" else "周频",
                key=f"factor_expansion_single_frequency_{Path(factor_base.source_config_path or factor_base.name).stem}",
            )
        selected_objectives = (str(objective_name),)
        selected_frequencies = ("daily" if frequency_name == "日频" else "weekly",)
        st.caption(_factor_objective_copy(str(objective_name)))
        st.caption(f"{'每个交易日' if frequency_name == '日频' else '每周'}生成一次信号；所有本轮路线使用相同频率。")
        factor_base = DashboardStrategyConfig(
            name=factor_base.name,
            weights=factor_base.weights,
            thresholds=factor_base.thresholds,
            positions=factor_base.positions,
            objective=OBJECTIVES[str(objective_name)],
            factor_windows=factor_base.factor_windows,
            benchmark_id=factor_base.benchmark_id,
            signal_frequency=selected_frequencies[0],
            research_provenance=factor_base.research_provenance,
            source_config_path=factor_base.source_config_path,
        )
        st.caption(
            "完整原始8因子加新增因子时，扩展搜索会纳入新增因子全为0的原始最优策略作为保底；"
            "其他自选集合仅搜索当前启用因子。所选基线的因子判定阈值、仓位、止盈止损和因子窗口保持不变。"
        )
    else:
        st.caption("全量研究采用统一的1 / 0 / 0仓位起点；权重搜索允许把无效因子设为0。")
    st.markdown("<div class='section-head compact'><h3>事前因子筛选</h3><p>在运行前明确哪些因子进入搜索。取消启用后，该因子不会进入权重或阈值搜索；启用后仍允许搜索赋为0。</p></div>", unsafe_allow_html=True)
    _render_factor_preselection_controls()
    enabled_factor_columns, _, _ = _factor_preselection_state()
    st.caption("静态研究完成后会展示在本页下方；直接滚动验证完成后会作为独立 R 策略归档到历史实验。")
    if st.button("运行因子增加研究", type="primary", use_container_width=True, key="run_factor_expansion_web"):
        if not enabled_factor_columns:
            st.error("请至少启用一个因子后再运行。")
            return
        if run_scope == "直接加入滚动定参队列":
            comparison_ready = _is_factor_comparison_universe(enabled_factor_columns)
            factor_version = "扩展因子" if comparison_ready else "自选因子"
            if factor_base is None:
                # Full six-route comparison intentionally uses the same 1/0/0
                # base that static factor research has always used.
                factor_base = DashboardStrategyConfig(
                    name="因子研究统一1/0/0起点",
                    weights=DashboardWeights(),
                    thresholds=ROOT_THRESHOLDS,
                    positions=ROOT_POSITIONS,
                    objective=OBJECTIVES["收益"],
                    factor_windows=FactorWindowConfig(),
                    benchmark_id=GOV_10Y,
                    signal_frequency="daily",
                )
            try:
                tasks = []
                factor_label = _factor_version_label(factor_version, len(enabled_factor_columns))
                for selected_objective in selected_objectives:
                    for selected_frequency in selected_frequencies:
                        direct_task_name = (
                            f"因子完整滚动·{selected_objective}优先·"
                            f"{'日频' if selected_frequency == 'daily' else '周频'}·{factor_label}"
                        )
                        queue_base = replace(
                            factor_base,
                            objective=OBJECTIVES[selected_objective],
                            signal_frequency=selected_frequency,
                        )
                        tasks.append(enqueue_factor_rolling_task(ROOT, {
                            "task_name": direct_task_name,
                            "base_config": queue_base.as_dict(),
                            "objective_name": selected_objective,
                            "signal_frequency": selected_frequency,
                            "factor_version": factor_version,
                            "enabled_factor_columns": list(enabled_factor_columns),
                            "minimum_training_months": 24,
                            "recalibration_months": 3,
                            "beam_width": 160,
                        }))
                worker_started = start_rolling_queue_worker(ROOT)
            except (OSError, ValueError, TimeoutError) as exc:
                st.error(f"未能加入因子滚动队列：{exc}")
                return
            if len(tasks) == 1 and (tasks[0].get("status") == "启动中" or worker_started):
                st.success(f"已直接开始因子滚动验证：{tasks[0]['task_id']}。")
            else:
                st.success(f"已加入 {len(tasks)} 条因子滚动任务；首条为 {tasks[0]['task_id']}，将按队列顺序执行。")
            return
        with st.status("正在生成研究快照...", expanded=True) as status:
            try:
                static_dir = run_factor_expansion_static_research(
                    ROOT,
                    training_end.isoformat(),
                    objectives=selected_objectives,
                    frequencies=selected_frequencies,
                    research_scope=str(scope),
                    base_config=factor_base,
                    enabled_factor_columns=enabled_factor_columns,
                )
                st.write(f"静态研究完成：{static_dir.name}")
                factor_versions = (
                    ("原始因子", "扩展因子")
                    if _is_factor_comparison_universe(enabled_factor_columns)
                    else ("自选因子",)
                )
                experiment_dirs = [
                    _archive_single_factor_expansion_result(
                        static_dir, selected_objective, selected_frequency,
                        training_end.isoformat(), factor_base, factor_version,
                    )
                    for selected_objective in selected_objectives
                    for selected_frequency in selected_frequencies
                    for factor_version in factor_versions
                ]
                archive_kind = "原始 / 扩展因子路线" if len(factor_versions) == 2 else "自选因子集合路线"
                st.write(f"已将 {len(experiment_dirs)} 条{archive_kind}分别归档为独立历史策略。")
                with st.expander("本次归档", expanded=True):
                    for index, experiment_dir in enumerate(experiment_dirs):
                        archive_config = load_strategy_config(experiment_dir / "config.json")
                        left, open_result, favorite = st.columns([3.2, 1.35, 0.45])
                        left.caption(_run_display_name(experiment_dir, archive_config.name))
                        with open_result:
                            st.link_button(
                                "查看历史结果",
                                f"/history?experiment={quote(experiment_dir.name)}",
                                use_container_width=True,
                            )
                        with favorite:
                            _render_favorite_button(experiment_dir, key=f"favorite_factor_new_{experiment_dir.name}_{index}")
                status.update(label="因子增加研究完成", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="因子增加研究未完成", state="error", expanded=True)
                st.error(str(exc))
                return
        st.session_state["factor_research_output_dir"] = str(static_dir)
        st.session_state["factor_study_choice"] = static_dir
        st.session_state["factor_research_just_completed"] = static_dir.name
        st.session_state["factor_research_experiment_dirs"] = [str(path) for path in experiment_dirs]
        st.query_params["study"] = static_dir.name
        st.query_params["result_mode"] = "静态样本外"
        st.rerun()

    last_run = st.session_state.get("factor_research_output_dir")
    if last_run:
        if st.session_state.pop("factor_research_just_completed", None):
            st.success(f"研究已完成，下方正在展示批次 {Path(str(last_run)).name}。")
        recent_columns = st.columns(2)
        recent_columns[0].link_button(
            "定位到最近生成的研究",
            f"/factor-research?study={quote(Path(str(last_run)).name)}",
            use_container_width=True,
        )
        archived_results = [Path(value) for value in st.session_state.get("factor_research_experiment_dirs", [])]
        if archived_results:
            recent_columns[1].caption(f"本批次已生成 {len(archived_results)} 条独立历史策略，可在历史实验中按 F 编号查看。")


def _archive_single_factor_expansion_result(
    study_dir: Path,
    objective_name: str,
    frequency: str,
    training_end: str,
    base_config: DashboardStrategyConfig | None = None,
    factor_version: str = "扩展因子",
) -> Path:
    """Expose one factor-version route as an independent history strategy."""
    if factor_version not in {"原始因子", "扩展因子", "自选因子"}:
        raise ValueError("未知因子版本")
    stem = _factor_stem(factor_version, objective_name, frequency)
    config_path = study_dir / f"{stem}_最终配置.json"
    daily_path = study_dir / f"{stem}_样本外日度.csv"
    signals_path = study_dir / f"{stem}_样本外信号.csv"
    if not all(path.exists() for path in (config_path, daily_path, signals_path)):
        raise FileNotFoundError("单一策略对缺少归档所需的配置、日度或信号文件")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    factor_weights = {str(key): float(value) for key, value in payload.get("权重", {}).items()}
    legacy_weights = {
        key: factor_weights.get(key, 0.0)
        for key in DashboardWeights().as_dict()
    }
    factor_label = _factor_version_label(factor_version, len(payload.get("启用因子", [])))
    config = DashboardStrategyConfig(
        name=f"因子研究_{objective_name}优先_{'日频' if frequency == 'daily' else '周频'}_{factor_label}",
        weights=DashboardWeights(**legacy_weights),
        thresholds=DashboardThresholds(**payload.get("阈值", {})),
        positions=DashboardPositionPolicy(**payload.get("仓位", {})),
        objective=ObjectiveConfig(**payload.get("目标函数", {})),
        factor_windows=base_config.factor_windows if base_config else FactorWindowConfig(),
        signal_frequency=frequency,
    )
    # Factor research is a child of its selected baseline, not a replacement
    # origin.  Use the same record_step path as rolling research so every
    # parent experiment and intermediate search remains visible and linkable.
    source_config = base_config or config
    config = record_step(
        source_config,
        config,
        ROOT,
        "联合搜索",
        output_path=config_path,
        training_end=training_end,
        search_version="V2 权重搜索 + 阈值搜索",
        entrypoint="run_factor_expansion_research.run_factor_expansion_static_research",
        arguments={
            "因子集合": factor_label,
            "事前启用因子": list(payload.get("启用因子", [])),
            "事前停用因子": list(payload.get("事前停用因子", [])),
            "搜索目标": objective_name,
        },
        note=f"在{factor_label}集合中执行 V2 权重搜索及阈值搜索。",
    )
    provenance = dict(config.research_provenance or {})
    notes = list(provenance.get("notes", [])) if isinstance(provenance.get("notes"), list) else []
    notes.append(
        f"该归档是因子增加研究的{factor_label}路线。"
        + ("同一研究批次保留原始与扩展因子对照。" if factor_version in {"原始因子", "扩展因子"} else "本研究未生成原始8因子对照。")
    )
    provenance.update({
        "notes": list(dict.fromkeys(notes)),
        "因子研究权重": factor_weights,
        "扩展因子权重": factor_weights if factor_version == "扩展因子" else None,
        "事前停用因子": payload.get("事前停用因子", []),
        "因子增加研究批次": study_dir.name,
        "因子增加研究路线": {"目标": objective_name, "频率": frequency, "因子版本": factor_version, "结果口径": "静态样本外", "事前启用因子": payload.get("启用因子", [])},
        "研究类型": "因子增加单一策略归档",
    })
    config = replace(config, research_provenance=provenance)
    daily, signals = _load_factor_static_full_history(study_dir, stem)
    strategy_metrics, benchmark_metrics = _factor_detail_metrics(daily)
    return archive_dashboard_experiment(
        ROOT,
        config,
        daily,
        signals,
        strategy_metrics,
        benchmark_metrics,
        source="因子增加单一策略研究",
        research_metadata={
            "训练截止日": training_end,
            "因子研究批次": study_dir.name,
            "因子版本": factor_label,
            "目标": objective_name,
            "信号频率": frequency,
            "研究基线": base_config.name if base_config else "统一1 / 0 / 0起点",
            "事前停用因子": payload.get("事前停用因子", []),
        },
    )


def _render_factor_research_page() -> None:
    st.markdown(
        "<section class='hero-shell'><div><p class='eyebrow'>因子对照与自选集合</p><h1 class='hero-title'>因子研究</h1><p class='hero-copy'>先设定研究条件并运行搜索，再查看对照结果或自选集合策略的收益路径、交易和因子筛选结果。</p></div></section>",
        unsafe_allow_html=True,
    )
    _render_factor_expansion_launch_panel()
    st.markdown(
        "<div class='section-head'><h2>研究结果</h2><p>选择已完成的研究批次和参数更新方式，下面所有比较与详情会同步切换。</p></div>",
        unsafe_allow_html=True,
    )
    studies = _factor_studies()
    if not studies:
        st.info("尚无已登记的因子研究快照。请先在上方设定研究条件并运行。")
        return
    st.markdown("<div class='section-head'><h2>1. 选择结果口径</h2><p>下面的比较结果、收益路径和因子筛选都会随这里的选择改变。</p></div>", unsafe_allow_html=True)
    requested = str(st.query_params.get("study", "")).strip()
    default_index = next((index for index, path in enumerate(studies) if path.name == requested), 0)
    scope_a, scope_b = st.columns((1, 1.7))
    with scope_a:
        study_dir = st.selectbox("研究批次", studies, index=default_index, format_func=_factor_study_label, key="factor_study_choice")
    with scope_b:
        study_has_rolling = (study_dir / "滚动定参" / "滚动定参汇总.csv").exists()
        mode_options = ["固定参数", "动态定参"] if study_has_rolling else ["固定参数"]
        requested_mode = str(st.query_params.get("result_mode", "静态样本外"))
        default_mode = "动态定参" if requested_mode == "滚动样本外" and study_has_rolling else "固定参数"
        mode_key = f"factor_mode_{study_dir.name}"
        query_token = f"{study_dir.name}:{requested_mode}"
        if st.session_state.get("factor_mode_query_token") != query_token:
            st.session_state[mode_key] = default_mode
            st.session_state["factor_mode_query_token"] = query_token
        mode_label = st.segmented_control(
            "参数更新方式", mode_options,
            default=default_mode,
            key=mode_key,
        )
    manifest = _load_factor_manifest(study_dir)
    rolling_exists = (study_dir / "滚动定参" / "滚动定参汇总.csv").exists()
    mode = "滚动样本外" if mode_label == "动态定参" else "静态样本外"
    if mode == "滚动样本外":
        st.caption("从满足24个月最低训练长度的日期开始，每3个月扩大训练区间并重新搜索参数；展示的是逐段拼接的样本外结果。")
    else:
        st.caption("只在页面所示训练截止日前搜索一次参数，之后整个样本外区间使用同一套参数；历史实验中的因子研究归档采用这一口径。")
    summary = _factor_summary(study_dir, str(mode))
    if summary.empty:
        st.warning("该研究快照缺少所选口径的汇总文件。")
        return
    is_custom_universe = set(summary["因子版本"].dropna().astype(str)) == {"自选因子"}
    if is_custom_universe:
        enabled_count = len(manifest.get("静态搜索", {}).get("事前启用因子", []))
        st.markdown(
            "<section class='metric-grid'>"
            + _metric_cell("研究模式", "自选因子集合", "不生成原始8因子对照", 0.0, compact=True)
            + _metric_cell("启用因子数", str(enabled_count), "仅这些因子进入权重与阈值搜索", float(enabled_count))
            + _metric_cell("策略路线数", str(len(summary)), "每个目标与频率各生成一条独立策略", float(len(summary)))
            + _metric_cell("训练截止日", str(manifest.get("训练截止日", "未登记")), "固定参数以此日前的数据选择", 0.0, compact=True)
            + "</section>",
            unsafe_allow_html=True,
        )
        display = summary.copy()
        display["策略"] = display.apply(
            lambda row: _factor_result_label(
                study_dir,
                str(row["目标"]),
                "daily" if str(row["频率"]) == "日频" else "weekly",
                str(mode),
                "自选因子",
                f"{row['目标']}优先 · {row['频率']} · 自选{enabled_count}因子",
            ),
            axis=1,
        )
        display["查看单策略结果"] = display.apply(
            lambda row: (
                f'<a class="factor-detail-link" href="{_factor_result_link(study_dir, str(row["目标"]), "daily" if str(row["频率"]) == "日频" else "weekly", str(mode), "自选因子")}" target="_blank">'
                f'{_factor_result_label(study_dir, str(row["目标"]), "daily" if str(row["频率"]) == "日频" else "weekly", str(mode), "自选因子", "查看策略")}</a>'
            ),
            axis=1,
        )
        display = display.rename(columns={
            "资本利得_BP": "样本外累计资本利得_BP",
            "交易胜率": "样本外交易胜率",
            "已平仓交易数": "样本外已平仓交易数",
            "最大回撤_BP": "样本外最大回撤_BP",
        })
        visible = [column for column in ["策略", "样本外累计资本利得_BP", "样本外交易胜率", "样本外已平仓交易数", "样本外最大回撤_BP", "有效因子数", "查看单策略结果"] if column in display]
        st.markdown("<div class='section-head'><h2>自选集合结果</h2><p>只评估当前启用的因子集合，不与不在集合中的原始因子比较。</p></div>", unsafe_allow_html=True)
        _render_theme_table(display.loc[:, visible], html_columns={"查看单策略结果"}, scrollable=True, wide=True)
        _render_factor_replay_and_artifacts(study_dir, manifest, rolling_exists)
        return
    original = summary.loc[summary["因子版本"] == "原始因子"].copy()
    expanded = summary.loc[summary["因子版本"] == "扩展因子"].copy()
    paired = original.merge(expanded, on=["目标", "频率"], suffixes=("_原始", "_扩展"))
    oos_improved = int(
        (
            pd.to_numeric(paired["资本利得_BP_扩展"], errors="coerce")
            > pd.to_numeric(paired["资本利得_BP_原始"], errors="coerce")
        ).sum()
    )
    training_non_decrease: int | None = None
    if mode == "静态样本外" and {"训练目标函数_原始", "训练目标函数_扩展"}.issubset(paired.columns):
        training_non_decrease = int(
            (
                pd.to_numeric(paired["训练目标函数_扩展"], errors="coerce")
                >= pd.to_numeric(paired["训练目标函数_原始"], errors="coerce") - 1e-12
            ).sum()
        )
    st.markdown(
        "<section class='metric-grid'>"
        + _metric_cell("策略对数量", str(len(paired)), "当前研究批次实际包含的路线", float(len(paired)))
        + (
            _metric_cell(
                "搜索期目标不下降", f"{training_non_decrease} / {len(paired)}",
                "同一路线原始最优策略作为保底", float(training_non_decrease),
            )
            if training_non_decrease is not None
            else _metric_cell("每期搜索保底", "已启用", "扩展搜索包含同一期原始策略", 0.0, compact=True)
        )
        + _metric_cell("样本外收益更高", f"{oos_improved} / {len(paired)}", "只作未来表现检验，不参与选参", float(oos_improved))
        + _metric_cell("静态选参截止日", str(manifest.get("训练截止日", "未登记")), "固定参数以此日前的数据选择", 0.0, compact=True)
        + "</section>",
        unsafe_allow_html=True,
    )
    mode_caption = "每三个月仅用此前数据重搜参数，再把新参数用于下一段。" if mode == "滚动样本外" else "先用训练期选定一次参数，再将同一套参数用于训练截止日之后。"
    route_count = len(paired)
    scope_name = str(manifest.get("研究范围", "全量六路线比较"))
    st.markdown(
        f"<div class='section-head'><h2>2. 比较{route_count}组策略对</h2><p>{escape(scope_name)}。{mode_caption} 表内变化均为样本外表现；红色表示改善，绿色表示下降。每个策略可直接进入完整研究工作区。</p></div>",
        unsafe_allow_html=True,
    )
    comparison = _factor_comparison_table(study_dir, summary, str(mode))
    _render_theme_table(
        comparison,
        html_columns={"样本外变化 · 收益 / 胜率", "搜索期目标分 · 原始 → 扩展", "查看单策略结果"},
        scrollable=True,
        wide=True,
    )

    route_value = str(st.query_params.get("route", ""))
    route_parts = route_value.rsplit("_", 1)
    available_objectives = summary["目标"].dropna().astype(str).drop_duplicates().tolist()
    available_frequencies = summary["频率"].dropna().astype(str).drop_duplicates().tolist()
    default_frequency = route_parts[1] if len(route_parts) == 2 and route_parts[1] in {"daily", "weekly"} else ("daily" if "日频" in available_frequencies else "weekly")
    default_objective = route_parts[0] if len(route_parts) == 2 and route_parts[0] in available_objectives else available_objectives[0]
    st.markdown("<div class='section-head'><h2>3. 选择一组查看细节</h2><p>此处只切换下方的双策略路径和因子筛选证据，不会改变上方的比较结果。</p></div>", unsafe_allow_html=True)
    route_a, route_b = st.columns(2)
    with route_a:
        objective = st.segmented_control("搜索目标", available_objectives, default=default_objective, key=f"factor_objective_{study_dir.name}")
    with route_b:
        frequency = st.segmented_control("发出信号的频率", available_frequencies, default="日频" if default_frequency == "daily" else "周频", key=f"factor_frequency_{study_dir.name}")
    frequency_code = "daily" if frequency == "日频" else "weekly"
    st.caption(_factor_objective_copy(str(objective)))
    st.markdown(f"<div class='section-head'><h2>当前策略对：{escape(str(objective))}优先 · {escape(str(frequency))}</h2><p>仅比较原始8因子与扩展15因子这两条策略，不是六条路线的合并曲线。</p></div>", unsafe_allow_html=True)
    chart = _factor_capital_comparison_chart(study_dir, str(objective), frequency_code, str(mode))
    if chart is not None:
        st.markdown("#### 两套策略的累计资本利得路径")
        _render_interactive_chart(chart, key=f"factor_path_{study_dir.name}_{objective}_{frequency_code}_{mode}")
    else:
        st.info("该路线尚无可画图的日度结果文件。重新运行该研究后会自动补齐。")
    if mode == "静态样本外":
        routes = _factor_archive_routes(study_dir.name)
        historical_routes = [
            ("原始8因子", routes.get((str(objective), frequency_code, "原始因子")), "原始因子"),
            ("扩展15因子", routes.get((str(objective), frequency_code, "扩展因子")), "扩展因子"),
        ]
        available_archives = [(label, archive_name, factor_version) for label, archive_name, factor_version in historical_routes if archive_name]
        if available_archives:
            st.markdown("#### 查看历史结果")
            link_columns = st.columns(len(available_archives))
            for column, (label, archive_name, factor_version) in zip(link_columns, available_archives):
                with column:
                    st.link_button(
                        f"{_factor_result_label(study_dir, str(objective), frequency_code, str(mode), factor_version, label)} 历史结果",
                        f"/history?experiment={quote(str(archive_name))}",
                        use_container_width=True,
                    )
        else:
            st.caption("该历史研究批次尚未生成独立历史归档，因此暂只能查看本页研究明细。")
    else:
        st.caption("动态定参按每期不同参数执行，目前在本页展示路线明细；不会错误跳转到固定参数的历史结果。")
    _render_factor_route_evidence(study_dir, "扩展因子", str(objective), frequency_code, str(mode))
    _render_factor_replay_and_artifacts(study_dir, manifest, rolling_exists)


def _render_factor_route_evidence(study_dir: Path, factor_version: str, objective: str, frequency: str, mode: str) -> None:
    stem = _factor_stem(factor_version, objective, frequency)
    config_path = study_dir / f"{stem}_最终配置.json"
    if not config_path.exists():
        st.warning("该路线缺少静态配置快照。")
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weights = {str(key): float(value) for key, value in config.get("权重", {}).items()}
    disabled = {str(key) for key in config.get("事前停用因子", [])}
    rolling_path = study_dir / "滚动定参" / f"{stem}_逐期定参.csv"
    rolling_weights: list[dict[str, float]] = []
    if rolling_path.exists() and mode == "滚动样本外":
        periods = pd.read_csv(rolling_path, encoding="utf-8-sig")
        for raw in periods["权重"]:
            rolling_weights.append({str(key): float(value) for key, value in json.loads(raw).items()})
    rows = []
    ordered_weight_keys = [key for key in FACTOR_DISPLAY_COLUMNS if key in weights]
    ordered_weight_keys.extend(key for key in weights if key not in ordered_weight_keys)
    for key in ordered_weight_keys:
        value = weights[key]
        values = [item.get(key, 0.0) for item in rolling_weights]
        used_periods = sum(weight > 0 for weight in values)
        is_active = used_periods > 0 if mode == "滚动样本外" and values else value > 0
        if values:
            conclusion = "动态定参始终剔除" if used_periods == 0 else f"动态定参使用 {used_periods} / {len(values)} 期"
        else:
            conclusion = "固定参数使用" if value > 0 else "固定参数剔除"
        rows.append({
            "因子": FACTOR_LABELS.get(key, key),
            "来源": "本轮新增" if key in EXPANDED_FACTOR_COLUMNS else "原始因子",
            "固定参数权重": f"{value:.0f}",
            "动态定参使用期数": f"{used_periods} / {len(values)}" if values else "-",
            "动态平均权重": f"{sum(values) / len(values):.1f}" if values else "-",
            "选择结论": conclusion,
            "_row_class": "factor-active" if is_active else "factor-inactive",
        })
    for key in FACTOR_DISPLAY_COLUMNS:
        if key not in disabled:
            continue
        rows.append({
            "因子": FACTOR_LABELS.get(key, key),
            "来源": "本轮新增" if key in EXPANDED_FACTOR_COLUMNS else "原始因子",
            "固定参数权重": "未启用",
            "动态定参使用期数": "未启用",
            "动态平均权重": "未启用",
            "选择结论": "事前停用，未进入搜索",
            "_row_class": "factor-inactive",
        })
    display_rank = {FACTOR_LABELS[key]: index for index, key in enumerate(FACTOR_DISPLAY_COLUMNS)}
    rows.sort(key=lambda row: display_rank.get(str(row["因子"]), len(display_rank)))
    st.markdown("#### 因子筛选证据")
    st.caption("事前停用表示该因子从未进入搜索；固定参数权重为0则表示进入搜索后被筛掉。动态定参使用期数显示每次重搜后实际被保留的次数。")
    _render_theme_table(pd.DataFrame(rows), qualitative_column="选择结论", row_class_column="_row_class", scrollable=True)
    if rolling_path.exists() and mode == "滚动样本外":
        with st.expander("查看逐期定参与权重"):
            periods = pd.read_csv(rolling_path, encoding="utf-8-sig")
            _render_theme_table(periods, numeric_columns=set(periods.columns), scrollable=True, wide=True)


def _render_factor_replay_and_artifacts(study_dir: Path, manifest: dict[str, object], rolling_exists: bool) -> None:
    st.markdown("<div class='section-head'><h2>研究记录与操作</h2><p>先查看本次研究使用的规则或完整报告；需要时再按相同清单生成一个新的研究批次。</p></div>", unsafe_allow_html=True)
    report_path = study_dir / "因子增加研究报告.html"
    summary_path = study_dir / "静态研究汇总.csv"
    manifest_path = study_dir / "研究清单.json"
    info_a, info_b, info_c = st.columns(3)
    with info_a:
        if report_path.exists():
            st.link_button("打开完整研究报告", report_path.as_uri(), use_container_width=True)
    with info_b:
        st.download_button("下载比较结果（CSV）", summary_path.read_bytes(), file_name=summary_path.name, mime="text/csv", use_container_width=True)
    with info_c:
        st.download_button("下载研究清单（JSON）", manifest_path.read_bytes(), file_name=manifest_path.name, mime="application/json", use_container_width=True)

    with st.expander("按相同研究清单重新运行", expanded=False):
        st.caption("会生成新的研究批次，不覆盖当前页面展示的结果；使用同一训练截止日和同一搜索规则。")
        replay_options = ["只重跑固定参数", "固定参数和动态定参"] if rolling_exists else ["只重跑固定参数"]
        replay_run_mode = st.segmented_control(
            "重新运行范围", replay_options,
            default="固定参数和动态定参" if rolling_exists else "只重跑固定参数", key=f"factor_replay_scope_{study_dir.name}",
        )
        st.caption("只重跑固定参数仅执行一次选参；固定参数和动态定参还会按原规则逐期扩大训练区间并重新选参。")
        if st.button("开始重新运行", type="primary", use_container_width=True, key=f"factor_replay_{study_dir.name}"):
            training_end = str(manifest.get("训练截止日", "2025-01-01"))
            static_spec = manifest.get("静态搜索", {}) if isinstance(manifest.get("静态搜索"), dict) else {}
            static_width = int(static_spec.get("搜索宽度", 160))
            replay_objectives = tuple(str(value) for value in static_spec.get("目标", ("收益", "胜率", "综合")))
            replay_frequencies = tuple(str(value) for value in static_spec.get("频率", ("daily", "weekly")))
            replay_enabled_factors = tuple(str(value) for value in static_spec.get("事前启用因子", ALL_FACTOR_COLUMNS))
            research_scope = str(manifest.get("研究范围", "全量六路线比较"))
            replay_base = strategy_config_from_dict(manifest["研究基线"]) if isinstance(manifest.get("研究基线"), dict) else None
            with st.status("正在按研究清单复现...", expanded=True) as status:
                try:
                    replay_dir = run_factor_expansion_static_research(
                        ROOT,
                        training_end,
                        static_width,
                        objectives=replay_objectives,
                        frequencies=replay_frequencies,
                        research_scope=research_scope,
                        base_config=replay_base,
                        enabled_factor_columns=replay_enabled_factors,
                    )
                    if replay_run_mode == "固定参数和动态定参":
                        run_factor_expansion_rolling(replay_dir, ROOT, base_config=replay_base)
                    status.update(label="研究复现完成", state="complete", expanded=False)
                except Exception as exc:
                    status.update(label="研究复现未完成", state="error", expanded=True)
                    st.error(str(exc))
                    return
            st.session_state["factor_research_output_dir"] = str(replay_dir)
            st.link_button("查看复现结果", f"/factor-research?study={quote(replay_dir.name)}", use_container_width=True)


def _render_factor_strategy_history(study_name: str) -> None:
    """Render a saved factor route from the history surface."""
    study_dir = ROOT / FACTOR_RESEARCH_DIR / study_name
    if not study_dir.is_dir() or not (study_dir / "研究清单.json").exists():
        st.error("因子研究快照不存在或未登记。")
        st.page_link(st.session_state.get("factor_research_navigation_page"), label="返回因子研究") if st.session_state.get("factor_research_navigation_page") else None
        return
    objective, frequency = str(st.query_params.get("factor_route", "收益_daily")).rsplit("_", 1)
    mode = str(st.query_params.get("factor_mode", "滚动样本外"))
    version = str(st.query_params.get("factor_version", "扩展因子"))
    frequency_name = "日频" if frequency == "daily" else "周频"
    mode_name = "动态定参 · 每3个月扩大训练区间并重搜" if mode == "滚动样本外" else "固定参数 · 训练期只选一次"
    version_name = _factor_version_label(version)
    stem = _factor_stem(version, objective, frequency)
    daily_path = _factor_daily_path(study_dir, stem, mode)
    config_path = study_dir / f"{stem}_最终配置.json"
    if not daily_path.exists():
        st.error("该路线没有可展示的日度结果文件，请重新运行研究。")
        return
    rolling_training_end: str | None = None
    if mode == "静态样本外":
        daily, signals = _load_factor_static_full_history(study_dir, stem)
    else:
        daily = pd.read_csv(daily_path, encoding="utf-8-sig", parse_dates=["date"])
        signals = _factor_detail_signals(study_dir, stem, daily, mode)
        rolling_search_daily_path = study_dir / "滚动定参" / f"{stem}_搜索期日度.csv"
        rolling_search_signal_path = study_dir / "滚动定参" / f"{stem}_搜索期信号.csv"
        rolling_period_path = study_dir / "滚动定参" / f"{stem}_逐期定参.csv"
        if rolling_search_daily_path.exists() and rolling_search_signal_path.exists() and rolling_period_path.exists():
            search_daily = pd.read_csv(rolling_search_daily_path, encoding="utf-8-sig", parse_dates=["date", "signal_date"])
            daily = pd.concat([search_daily, daily], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
            daily = rebuild_expansion_stitched_cumulatives(daily)
            search_signals = pd.read_csv(rolling_search_signal_path, encoding="utf-8-sig", parse_dates=["signal_date"])
            signals = pd.concat([search_signals, signals], ignore_index=True)
            # CSV archives from older runs may leave one side as plain text
            # while the other is parsed as Timestamp. Normalize before any
            # de-duplication or ordering so mixed-type comparisons cannot
            # raise ``TypeError: '<' not supported ...``.
            signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")
            signals = (
                signals.dropna(subset=["signal_date"])
                .drop_duplicates("signal_date", keep="last")
                .sort_values("signal_date")
                .reset_index(drop=True)
            )
            first_period = pd.read_csv(rolling_period_path, encoding="utf-8-sig").iloc[0]
            rolling_training_end = str(first_period["训练截止日"])
    payload = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    strategy_metrics, benchmark_metrics = _factor_detail_metrics(daily)
    st.markdown(
        f"<div class='archive-breadcrumb'><a href='/history' target='_self'>历史实验</a><span>/</span><strong>{escape(version_name)}策略 · {escape(objective)}优先 · {escape(frequency_name)}</strong></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='section-head'><h2>{escape(version_name)} · {escape(objective)}优先 · {escape(frequency_name)}</h2><p>{escape(mode_name)}。以下各模块均来自该路线已保存的实际执行结果。</p></div>",
        unsafe_allow_html=True,
    )
    if mode == "静态样本外":
        manifest = _load_factor_manifest(study_dir)
        daily, signals, strategy_metrics, benchmark_metrics, _ = _render_historical_period_summary(
            daily,
            signals,
            benchmark_metrics,
            str(manifest.get("训练截止日", payload.get("训练截止日", DEFAULT_TRAINING_END))),
            f"factor_{study_name}_{stem}",
            default="样本外",
        )
    else:
        if rolling_training_end:
            daily, signals, strategy_metrics, benchmark_metrics, _ = _render_historical_period_summary(
                daily,
                signals,
                benchmark_metrics,
                rolling_training_end,
                f"factor_rolling_{study_name}_{stem}",
                default="样本外",
            )
        else:
            st.info(
                "动态定参每个季度都用起点至当期截止日的全部历史重新搜索，随后只记录下一季度的样本外表现。"
                "当前旧批次未保存首段训练明细，因此无法单独画出第一段搜索期。"
            )
            _render_summary(strategy_metrics, benchmark_metrics, signals)
    _render_factor_strategy_workspace(
        daily, signals, strategy_metrics, benchmark_metrics, payload,
        study_dir, stem, mode, frequency,
    )


def _load_factor_static_full_history(study_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_paths = [study_dir / f"{stem}_训练期日度.csv", study_dir / f"{stem}_样本外日度.csv"]
    signal_paths = [study_dir / f"{stem}_训练期信号.csv", study_dir / f"{stem}_样本外信号.csv"]
    if not all(path.exists() for path in daily_paths + signal_paths):
        raise FileNotFoundError("该因子策略缺少训练期或样本外明细，无法展示完整历史区间")
    daily = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date", "signal_date"]) for path in daily_paths],
        ignore_index=True,
    ).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    daily = rebuild_expansion_stitched_cumulatives(daily)
    signals = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig", parse_dates=["signal_date"]) for path in signal_paths],
        ignore_index=True,
    ).drop_duplicates("signal_date", keep="last").sort_values("signal_date").reset_index(drop=True)
    return daily, signals


def _factor_detail_signals(study_dir: Path, stem: str, daily: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Use stored signal details when available; rolling results retain executed totals."""
    if mode != "滚动样本外":
        path = study_dir / f"{stem}_样本外信号.csv"
        if path.exists():
            signals = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["signal_date"])
            start, end = daily["date"].min(), daily["date"].max()
            signals = signals.loc[signals["signal_date"].between(start, end)].copy()
            if not signals.empty:
                signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="raise")
                return signals
    columns = [column for column in ["signal_date", "总分", "结论", "仓位"] if column in daily.columns]
    signals = daily.loc[:, columns].copy()
    signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="raise")
    return signals.drop_duplicates("signal_date", keep="last").sort_values("signal_date").reset_index(drop=True)


def _factor_signal_period_metrics(daily: pd.DataFrame, return_column: str) -> dict[str, object]:
    returns = (
        daily.groupby("signal_date")[return_column]
        .apply(lambda values: (1.0 + pd.to_numeric(values, errors="coerce").fillna(0.0)).prod() - 1.0)
        .dropna()
    )
    if returns.empty:
        return {"signal_period_count": 0, "winning_signal_periods": 0, "signal_period_win_rate": None, "avg_signal_period_return": None}
    return {
        "signal_period_count": int(len(returns)),
        "winning_signal_periods": int((returns > 0).sum()),
        "signal_period_win_rate": float((returns > 0).mean()),
        "avg_signal_period_return": float(returns.mean()),
    }


def _factor_detail_metrics(daily: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    strategy = performance_metrics(daily, return_col="strategy_return", nav_col="strategy_nav")
    benchmark = performance_metrics(daily, return_col="total_return", nav_col="benchmark_nav_rebased")
    strategy.update(_factor_signal_period_metrics(daily, "strategy_return"))
    benchmark.update(_factor_signal_period_metrics(daily, "total_return"))
    strategy.update(capital_gain_trade_metrics(daily, "strategy_capital_bp", position_col="仓位"))
    benchmark.update(capital_gain_trade_metrics(daily, "benchmark_capital_bp", position_col="comparison_position"))
    strategy["capital_gain_bp_definition"] = "收益率方向变动BP（不乘久期）"
    benchmark["benchmark_name"] = CONDITIONAL_BENCHMARK_NAME
    return strategy, benchmark


def _render_factor_strategy_workspace(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
    payload: dict[str, object],
    study_dir: Path,
    stem: str,
    mode: str,
    frequency: str,
) -> None:
    diagnostics = _build_period_diagnostics(daily, signals)
    st.markdown("<div class='section-head'><h2>研究工作区</h2><p>与历史实验使用相同的交易、净值、归因、错判和信号模块。</p></div>", unsafe_allow_html=True)
    tabs = st.tabs(["交易表现", "传统净值", "收益归因", "错判诊断", "最新信号", "研究记录与参数"])
    with tabs[0]:
        _render_trading_tab(daily, strategy_metrics, benchmark_metrics)
    with tabs[1]:
        _render_nav_tab(daily, strategy_metrics, benchmark_metrics)
    with tabs[2]:
        _render_attribution_tab(daily)
    with tabs[3]:
        _render_diagnostics_tab(diagnostics, frequency)
    with tabs[4]:
        _render_latest_signal(
            signals,
            factor_weights=payload.get("权重") if isinstance(payload.get("权重"), dict) else None,
            factor_thresholds=payload.get("阈值") if isinstance(payload.get("阈值"), dict) else None,
        )
        if mode == "滚动样本外":
            st.caption("动态定参路线的总分、结论和仓位来自逐段实际执行结果；每段权重在“研究记录与参数”中查看。")
    with tabs[5]:
        _render_factor_config_snapshot(payload, study_dir, stem, mode)


def _render_factor_config_snapshot(payload: dict[str, object], study_dir: Path, stem: str, mode: str) -> None:
    st.markdown("#### 当前配置快照")
    weights = {str(key): float(value) for key, value in (payload.get("权重") or {}).items()}
    ordered_keys = [key for key in FACTOR_DISPLAY_COLUMNS if key in weights]
    ordered_keys.extend(key for key in weights if key not in ordered_keys)
    weight_rows = [
        {"因子": FACTOR_LABELS.get(key, key), "权重": weights[key], "_row_class": "factor-active" if weights[key] != 0.0 else "factor-inactive"}
        for key in ordered_keys
    ]
    _render_theme_table(pd.DataFrame(weight_rows), numeric_columns={"权重"}, row_class_column="_row_class", scrollable=True)
    rows = []
    for group, values in (("阈值", payload.get("阈值", {})), ("仓位规则", payload.get("仓位", {})), ("目标函数", payload.get("目标函数", {}))):
        for key, value in values.items():
            rows.append({"参数组": group, "项目": key, "数值": value})
    if rows:
        st.markdown("#### 阈值、仓位与搜索目标")
        _render_theme_table(pd.DataFrame(rows), numeric_columns={"数值"}, scrollable=True)
    if mode == "滚动样本外":
        path = study_dir / "滚动定参" / f"{stem}_逐期定参.csv"
        if path.exists():
            st.markdown("#### 每段实际使用的参数")
            st.caption("训练起始日至训练截止日只用于重新选参；表中的收益、胜率和回撤均来自紧随其后的样本外区间。")
            periods = pd.read_csv(path, encoding="utf-8-sig").rename(columns={
                "累计资本利得_BP": "下一段样本外累计资本利得_BP",
                "资本利得超额_BP": "下一段样本外资本利得超额_BP",
                "资本利得交易胜率": "下一段样本外交易胜率",
                "资本利得最大回撤_BP": "下一段样本外最大回撤_BP",
                "已平仓交易数": "下一段样本外已平仓交易数",
            })
            _render_theme_table(periods, scrollable=True, wide=True)


def _research_training_end(experiment_dir: Path) -> str:
    manifest_path = experiment_dir / "run_manifest.json"
    if not manifest_path.exists():
        return DEFAULT_TRAINING_END
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        research_range = manifest.get("研究区间", {})
        if not isinstance(research_range, dict):
            research_range = {}
        # Rolling archives register their first search window explicitly. This
        # keeps the history page's search-period metrics aligned with the run.
        return str(
            research_range.get("训练截止日")
            or research_range.get("首次搜索期结束日")
            or DEFAULT_TRAINING_END
        )
    except (OSError, json.JSONDecodeError, AttributeError):
        return DEFAULT_TRAINING_END


def _decimal_text(value: object, digits: int = 2, fallback: str = "暂无") -> str:
    """Format an optional numeric metric without treating NaN/None as zero."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return fallback if pd.isna(numeric) else f"{float(numeric):.{digits}f}"


def _manual_period_result(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    """Evaluate an ad-hoc display window without changing a saved experiment."""
    data_start = pd.Timestamp(daily["date"].min()).normalize()
    data_end = pd.Timestamp(daily["date"].max()).normalize()
    if start > end:
        st.warning("起始日不能晚于结束日，已暂按全区间展示。")
        start, end = data_start, data_end
    selected_daily, selected_signals, strategy_metrics, selected_benchmark_metrics = evaluate_period(
        daily, signals, start, end, benchmark_name
    )
    return {
        "daily": selected_daily,
        "signals": selected_signals,
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": selected_benchmark_metrics,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
    }


def _mark_period_custom(period_key: str) -> None:
    """Date edits are display-only, but must visibly switch the scope selector."""
    st.session_state[period_key] = "自定义时间窗口"


def _set_period_half(
    period_key: str,
    start_key: str,
    end_key: str,
    year: int,
    half: str,
    data_start: object,
    data_end: object,
) -> None:
    """Apply a half-year display window without changing the underlying strategy."""
    lower = pd.Timestamp(data_start).normalize()
    upper = pd.Timestamp(data_end).normalize()
    if half == "上半年":
        half_start, half_end = pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=6, day=30)
    else:
        half_start, half_end = pd.Timestamp(year=year, month=7, day=1), pd.Timestamp(year=year, month=12, day=31)
    st.session_state[start_key] = max(lower, half_start).date()
    st.session_state[end_key] = min(upper, half_end).date()
    _mark_period_custom(period_key)


def _render_period_controls(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark_name: str,
    periods: dict[str, dict[str, object]],
    available: list[str],
    key: str,
    label: str,
    default: str,
) -> tuple[str, dict[str, object]]:
    """Render a preset selector with always-visible, display-only date bounds."""
    options = [*available, "自定义时间窗口"]
    fallback = default if default in available else available[-1]
    start_key = f"{key}_manual_start"
    end_key = f"{key}_manual_end"
    last_key = f"{key}_last_selected"
    data_start = pd.Timestamp(daily["date"].min()).normalize()
    data_end = pd.Timestamp(daily["date"].max()).normalize()

    if st.session_state.get(key) not in options:
        st.session_state[key] = fallback
    if start_key not in st.session_state or end_key not in st.session_state:
        initial = periods[fallback]
        st.session_state[start_key] = pd.Timestamp(initial["start"]).date()
        st.session_state[end_key] = pd.Timestamp(initial["end"]).date()

    selected = st.selectbox(
        label,
        options,
        format_func=lambda scope: (
            "自定义时间窗口" if scope == "自定义时间窗口"
            else f"{scope}（{periods[scope]['start']} ~ {periods[scope]['end']}）"
        ),
        key=key,
        help="只刷新当前页指标和图表，不重新运行策略，也不会新增实验归档。",
    )
    prior_selection = st.session_state.get(last_key, fallback)
    if selected != prior_selection and selected != "自定义时间窗口":
        selected_period = periods[selected]
        st.session_state[start_key] = pd.Timestamp(selected_period["start"]).date()
        st.session_state[end_key] = pd.Timestamp(selected_period["end"]).date()
    st.session_state[last_key] = selected

    # Keep stale widget state within the data available to this experiment.
    current_start = max(data_start, min(pd.Timestamp(st.session_state[start_key]).normalize(), data_end))
    current_end = max(data_start, min(pd.Timestamp(st.session_state[end_key]).normalize(), data_end))
    st.session_state[start_key] = current_start.date()
    st.session_state[end_key] = current_end.date()
    start_column, end_column = st.columns(2)
    start = pd.Timestamp(start_column.date_input(
        "开始日",
        min_value=data_start.date(),
        max_value=data_end.date(),
        key=start_key,
        on_change=_mark_period_custom,
        args=(key,),
    )).normalize()
    end = pd.Timestamp(end_column.date_input(
        "结束日",
        min_value=data_start.date(),
        max_value=data_end.date(),
        key=end_key,
        on_change=_mark_period_custom,
        args=(key,),
    )).normalize()

    same_year = start.year == end.year
    selected_year = int(start.year)
    first_half_end = pd.Timestamp(year=selected_year, month=6, day=30)
    second_half_start = pd.Timestamp(year=selected_year, month=7, day=1)
    first_half_available = same_year and first_half_end >= data_start
    second_half_available = same_year and second_half_start <= data_end
    with st.container(key=f"period_half_shortcuts_{key}"):
        st.markdown(
            "<p class='period-half-title'>快捷切换</p>" if same_year
            else "<p class='period-half-title disabled'>切换至同一年可半年快捷勾选</p>",
            unsafe_allow_html=True,
        )
        first_half_column, second_half_column = st.columns(2)
        first_half_column.button(
            "上半年",
            key=f"{key}_first_half",
            use_container_width=True,
            disabled=not first_half_available,
            on_click=_set_period_half,
            args=(key, start_key, end_key, selected_year, "上半年", data_start.date(), data_end.date()),
        )
        second_half_column.button(
            "下半年",
            key=f"{key}_second_half",
            use_container_width=True,
            disabled=not second_half_available,
            on_click=_set_period_half,
            args=(key, start_key, end_key, selected_year, "下半年", data_start.date(), data_end.date()),
        )
    if selected == "自定义时间窗口":
        result = _manual_period_result(daily, signals, benchmark_name, start, end)
        periods[selected] = result
    else:
        result = periods[selected]
    return selected, result


def _render_period_scope(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark_metrics: dict[str, object],
    training_end: str,
    key: str,
    default: str = "全区间",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object], dict[str, dict[str, object]], str]:
    periods = evaluate_periods(
        daily,
        signals,
        training_end,
        str(benchmark_metrics.get("benchmark_name", CONDITIONAL_BENCHMARK_NAME)),
        include_shortcuts=True,
    )
    available = [label for label, result in periods.items() if not result["daily"].empty]
    selected, result = _render_period_controls(
        daily,
        signals,
        str(benchmark_metrics.get("benchmark_name", CONDITIONAL_BENCHMARK_NAME)),
        periods,
        available,
        key,
        "观察区间",
        default,
    )
    _render_period_comparison(periods, selected)
    return (
        result["daily"],
        result["signals"],
        result["strategy_metrics"],
        result["benchmark_metrics"],
        periods,
        selected,
    )


def _render_period_comparison(periods: dict[str, dict[str, object]], selected: str) -> None:
    blocks = []
    for label, result in periods.items():
        metrics = result.get("strategy_metrics", {})
        if not metrics:
            continue
        active = " active" if label == selected else ""
        blocks.append(
            f'<article class="period-snapshot{active}"><h4>{escape(label)}</h4>'
            f'<div class="period-range">{escape(str(result["start"]))} 至 {escape(str(result["end"]))}</div><dl>'
            f'<dt>累计资本利得</dt><dd>{_bp_text(metrics.get("capital_gain_total_bp"))}</dd>'
            f'<dt>年化资本利得</dt><dd>{_bp_text(metrics.get("capital_gain_annualized_bp"))}</dd>'
            f'<dt>资本利得超额</dt><dd>{_bp_text(metrics.get("capital_gain_excess_bp"))}</dd>'
            f'<dt>已平仓胜率</dt><dd>{_pct(metrics.get("capital_gain_trade_win_rate"))}</dd>'
            f'<dt>最大回撤</dt><dd>{_bp_text(metrics.get("capital_gain_max_drawdown_bp"))}</dd>'
            "</dl></article>"
        )
    st.markdown(f'<section class="period-comparison">{"".join(blocks)}</section>', unsafe_allow_html=True)


def _render_historical_period_summary(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark_metrics: dict[str, object],
    training_end: str,
    experiment_id: str,
    default: str = "全区间",
    favorite_experiment: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object], str]:
    periods = evaluate_periods(
        daily,
        signals,
        training_end,
        str(benchmark_metrics.get("benchmark_name", CONDITIONAL_BENCHMARK_NAME)),
        include_shortcuts=True,
    )
    available = [label for label, result in periods.items() if not result["daily"].empty]
    with st.container(key="historical_hero"):
        main_col, position_col, period_col = st.columns([7.0, 1.7, 3.3], vertical_alignment="bottom")
        with period_col:
            with st.container(key="period_compact"):
                selected, result = _render_period_controls(
                    daily,
                    signals,
                    str(benchmark_metrics.get("benchmark_name", CONDITIONAL_BENCHMARK_NAME)),
                    periods,
                    available,
                    f"history_period_{experiment_id}",
                    "区间观察",
                    default,
                )
        selected_daily = result["daily"]
        selected_signals = result["signals"]
        strategy_metrics = result["strategy_metrics"]
        selected_benchmark_metrics = result["benchmark_metrics"]
        latest = selected_signals.iloc[-1]
        signal_date = _latest_signal_date(latest)
        with main_col:
            # The collection control deliberately sits directly above the
            # first character of "回测结果", rather than floating in a spare
            # right-hand column detached from the result it saves.
            if favorite_experiment is not None:
                _render_favorite_button(favorite_experiment, key=f"favorite_result_heading_{favorite_experiment.name}")
            st.markdown(f'<p class="eyebrow result-eyebrow">回测表现截至 / {escape(signal_date)}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="historical-hero-main"><h1 class="hero-title">当前结论：{escape(str(latest.get("结论", "未识别")))}</h1>'
                f'<p class="hero-copy">总分 {float(latest.get("总分", 0)):.1f}，目标仓位 {float(latest["仓位"]):.1f}。'
                "资本利得BP按 -仓位 × YTM变化BP 计算，不乘久期。</p></div>",
                unsafe_allow_html=True,
            )
        with position_col:
            sharpe_text = _decimal_text(strategy_metrics.get("sharpe"))
            st.markdown(
                f'<aside class="historical-hero-side"><div class="aside-label">最新仓位</div>'
                f'<div class="aside-value">{float(latest["仓位"]):.1f}</div>'
                f'<div class="aside-note">回测截止 {escape(signal_date)}<br>策略夏普 {sharpe_text}</div></aside>',
                unsafe_allow_html=True,
            )
    _render_metric_summary_grid(strategy_metrics, selected_benchmark_metrics)
    return selected_daily, selected_signals, strategy_metrics, selected_benchmark_metrics, selected


def _render_metric_summary_grid(strategy_metrics: dict[str, object], benchmark_metrics: dict[str, object]) -> None:
    capital_bp = float(strategy_metrics["capital_gain_total_bp"])
    benchmark_capital_bp = float(benchmark_metrics["capital_gain_total_bp"])
    capital_excess_bp = capital_bp - benchmark_capital_bp
    benchmark_name = str(benchmark_metrics.get("benchmark_name", CONDITIONAL_BENCHMARK_NAME))
    capital_win_rate = strategy_metrics.get("capital_gain_trade_win_rate")
    average_win_bp = strategy_metrics.get("capital_gain_avg_win_bp")
    capital_drawdown_bp = strategy_metrics.get("capital_gain_max_drawdown_bp")
    annualized_capital_bp = strategy_metrics.get("capital_gain_annualized_bp")
    benchmark_annualized_capital_bp = benchmark_metrics.get("capital_gain_annualized_bp")
    average_loss_bp = strategy_metrics.get("capital_gain_avg_loss_bp")
    worst_trade_bp = strategy_metrics.get("capital_gain_worst_trade_bp")
    average_holding_days = strategy_metrics.get("capital_gain_avg_holding_days")
    max_holding_days = strategy_metrics.get("capital_gain_max_holding_days")
    payoff = strategy_metrics.get("capital_gain_profit_loss_ratio")
    drawdown_start = strategy_metrics.get("capital_gain_max_drawdown_start")
    drawdown_end = strategy_metrics.get("capital_gain_max_drawdown_end")
    st.markdown(
        "<section class='metric-grid'>"
        + _metric_cell("累计资本利得", f"{capital_bp:.2f} BP", f"{benchmark_name}累计 {benchmark_capital_bp:.2f} BP", capital_bp)
        + _metric_cell("已平仓交易胜率", "暂无已平仓" if capital_win_rate is None else _pct(capital_win_rate), f"盈利 {strategy_metrics['capital_gain_winning_trades']:.0f} / 已平仓 {strategy_metrics['capital_gain_closed_trade_count']:.0f} 笔", 0.0 if capital_win_rate is None else float(capital_win_rate) - 0.5)
        + _metric_cell("平均每笔盈利", "暂无盈利交易" if average_win_bp is None else f"{float(average_win_bp):.2f} BP", _payoff_detail(payoff), 0.0 if average_win_bp is None else float(average_win_bp))
        + _metric_cell("资本利得最大回撤", "暂无" if capital_drawdown_bp is None else f"{float(capital_drawdown_bp):.2f} BP", _drawdown_period_detail(drawdown_start, drawdown_end), 0.0 if capital_drawdown_bp is None else float(capital_drawdown_bp))
        + _metric_cell("超额资本利得", f"{capital_excess_bp:.2f} BP", _relative_excess_detail(benchmark_name, capital_excess_bp, benchmark_capital_bp), capital_excess_bp)
        + _metric_cell("年化资本利得", "暂无" if annualized_capital_bp is None else f"{float(annualized_capital_bp):.2f} BP", f"{benchmark_name} 暂无" if benchmark_annualized_capital_bp is None else f"{benchmark_name} {float(benchmark_annualized_capital_bp):.2f} BP", 0.0 if annualized_capital_bp is None else float(annualized_capital_bp))
        + _metric_cell("平均单笔亏损", "暂无亏损" if average_loss_bp is None else f"{float(average_loss_bp):.2f} BP", "区间内无亏损交易" if average_loss_bp is None else ("最大亏损暂无" if worst_trade_bp is None or float(worst_trade_bp) >= 0 else f"最大亏损 {float(worst_trade_bp):.2f} BP"), 0.0 if average_loss_bp is None else float(average_loss_bp))
        + _metric_cell("平均每笔持有时间", "暂无交易" if average_holding_days is None else f"{float(average_holding_days):.1f} 交易日", "最长持有暂无" if max_holding_days is None else f"最长 {int(max_holding_days)} 交易日", 0.0)
        + "</section>",
        unsafe_allow_html=True,
    )


def _render_historical_result_page(experiment_id: str) -> None:
    experiment_root = (ROOT / "backtest_outputs" / "experiments").resolve()
    experiment_dir = (experiment_root / Path(experiment_id).name).resolve()
    if experiment_dir.parent != experiment_root or not (experiment_dir / "run_manifest.json").exists():
        st.error("未找到该历史实验，可能已移动或归档不完整。")
        st.page_link(st.session_state["history_navigation_page"], label="返回历史实验")
        return
    try:
        daily, signals, strategy_metrics, benchmark_metrics, config = load_experiment_result(experiment_dir)
    except Exception as exc:
        st.error(f"历史实验加载失败：{exc}")
        return
    full_daily, full_signals = daily, signals
    provenance = config.research_provenance or {}
    factor_archive_type = str(provenance.get("研究类型", "")) if isinstance(provenance, dict) else ""
    is_factor_expansion_archive = factor_archive_type == "因子增加单一策略归档"
    is_direct_factor_rolling_archive = factor_archive_type == "因子完整滚动定参归档"
    rolling_archive = _is_rolling_archive(experiment_dir, provenance)
    current_display_signals: pd.DataFrame | None = None
    latest_signal_config: DashboardStrategyConfig | None = None
    if _is_factor_research_strategy(config):
        current_display_signals = _current_factor_strategy_signals(config, full_signals)
    elif rolling_archive:
        current_display_signals, latest_signal_config = _current_rolling_signals(experiment_dir, full_signals)
    else:
        current_display_signals = _current_fixed_strategy_signals(config, full_signals)
    if current_display_signals is None:
        st.warning("未能读取该策略的已保存参数；当前信号暂以冻结归档显示。")
    if is_factor_expansion_archive:
        st.sidebar.markdown("## 因子增加研究归档")
        st.sidebar.caption("该结果使用因子研究引擎，作为只读历史记录展示；不能直接改成旧8因子策略后重新运行。")
        study_name = str(provenance.get("因子增加研究批次", "")).strip()
        if study_name:
            metadata = provenance.get("因子增加研究路线", {}) if isinstance(provenance.get("因子增加研究路线"), dict) else {}
            objective_name = str(metadata.get("目标", "收益"))
            frequency = str(metadata.get("频率", config.signal_frequency))
            study_dir = ROOT / FACTOR_RESEARCH_DIR / study_name
            factor_version = str(metadata.get("因子版本", "扩展因子"))
            if factor_version not in {"原始因子", "扩展因子", "自选因子"}:
                factor_version = "扩展因子"
            stem = _factor_stem(factor_version, objective_name, frequency)
            try:
                full_daily, full_signals = _load_factor_static_full_history(study_dir, stem)
                _, benchmark_metrics = _factor_detail_metrics(full_daily)
            except (OSError, ValueError, KeyError) as exc:
                st.sidebar.warning(f"未能补载搜索期明细：{exc}")
            st.sidebar.link_button(
                "查看同口径因子研究",
                f"/factor-research?study={quote(study_name)}&result_mode={quote('静态样本外')}&route={quote(f'{objective_name}_{frequency}')}",
                use_container_width=True,
            )
    elif is_direct_factor_rolling_archive:
        st.sidebar.markdown("## 因子完整滚动归档")
        st.sidebar.caption(
            "该结果由当前因子集合直接加入滚动任务队列生成，未先创建静态 F 研究归档。"
            "逐期参数与样本外表现均已冻结保存。"
        )
    else:
        st.sidebar.markdown("## 策略控制台")
        if rolling_archive:
            st.sidebar.caption(
                "重新运行会直接复用已保存的逐期参数并重算表现；只有数据跨过下一个定参点时，"
                "才搜索新增期间。运行会新增实验，不覆盖原归档。"
            )
        else:
            st.sidebar.caption("以该历史实验为基线修改；重新运行会新增实验，不覆盖原归档。")
        edited_config, run_clicked, save_clicked = _sidebar_config(
            config,
            key_prefix=f"history_{experiment_dir.name}",
            default_start=strategy_metrics.get("start_date"),
            default_end=strategy_metrics.get("end_date"),
        )
        if save_clicked:
            _save_config(edited_config)
        if run_clicked:
            new_experiment_dir = _run_backtest(edited_config)
            if new_experiment_dir is not None:
                _switch_to_history_experiment(new_experiment_dir)
    display_name = _run_display_name(experiment_dir, config.name)
    archive_note = _experiment_notes().get(experiment_dir.name, "")
    note_class = " has-note" if archive_note else ""
    note_label = f"备注：{archive_note}" if archive_note else "添加备注"
    st.markdown(
        f'<div class="archive-breadcrumb"><a href="/history" target="_self">历史实验</a><span>/</span>'
        f'<strong>{escape(display_name)}</strong><button class="archive-rename" type="button" '
        f'data-archive="{escape(experiment_dir.name, quote=True)}" data-title="{escape(_normalise_display_name(display_name), quote=True)}" '
        f'aria-label="重命名 {escape(display_name, quote=True)}" title="重命名">'
        '<span class="material-symbols-rounded" aria-hidden="true">edit</span></button>'
        f'<button class="archive-note{note_class}" type="button" data-archive="{escape(experiment_dir.name, quote=True)}" '
        f'data-note="{escape(archive_note, quote=True)}" aria-label="{escape(note_label, quote=True)}" '
        f'title="{escape(note_label, quote=True)}"><span class="material-symbols-rounded" aria-hidden="true">sticky_note_2</span></button></div>',
        unsafe_allow_html=True,
    )
    _render_result_rename_bridge()
    daily, signals, strategy_metrics, benchmark_metrics, selected_period = _render_historical_period_summary(
        full_daily,
        full_signals,
        benchmark_metrics,
        _research_training_end(experiment_dir),
        experiment_dir.name,
        default="样本外" if (is_factor_expansion_archive or is_direct_factor_rolling_archive) else "全区间",
        favorite_experiment=experiment_dir,
    )
    diagnostics = _build_period_diagnostics(daily, signals)
    _render_workspace(
        daily,
        signals,
        strategy_metrics,
        benchmark_metrics,
        diagnostics,
        config.name,
        config,
        experiment_dir,
        current_signals=current_display_signals if current_display_signals is not None else full_signals,
        latest_signal_config=latest_signal_config,
    )


def _render_historical_sidebar(config: DashboardStrategyConfig, experiment_dir: Path) -> None:
    objective = config.objective.as_dict()
    st.sidebar.markdown("## 历史结果参数")
    st.sidebar.caption("参数来自该次实验归档，只读展示，不会被当前首页配置替换。")
    st.sidebar.markdown(f"**策略名称**  \n{_run_display_name(experiment_dir, config.name)}")
    custom_title = _experiment_display_names().get(experiment_dir.name)
    if custom_title:
        st.sidebar.caption(f"归档原名：{_normalise_display_name(config.name)}")
    st.sidebar.caption(f"信号频率：{'日频' if config.signal_frequency == 'daily' else '周频'}")
    st.sidebar.caption(f"归档编号：{_archive_run_id(experiment_dir) or '未登记'}")
    with st.sidebar.expander("搜索目标函数（归档）", expanded=True):
        rows = [
            {"项目": "累计资本利得BP", "系数": objective["capital_gain_bp_weight"]},
            {"项目": "资本利得超额BP", "系数": objective["capital_gain_excess_bp_weight"]},
            {"项目": "已平仓交易胜率", "系数": objective["capital_trade_win_rate_weight"]},
            {"项目": "平均每笔盈利BP", "系数": objective["capital_gain_avg_win_bp_weight"]},
            {"项目": "资本利得回撤BP", "系数": objective["capital_gain_drawdown_bp_penalty"]},
            {"项目": "累计收益", "系数": objective["total_return_weight"]},
            {"项目": "超额收益", "系数": objective["excess_return_weight"]},
            {"项目": "夏普", "系数": objective["sharpe_weight"]},
            {"项目": "调仓周期胜率", "系数": objective["signal_win_rate_weight"]},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.code(
            f"目标分 = {objective['capital_gain_bp_weight']:g}×资本BP + "
            f"{objective['capital_gain_excess_bp_weight']:g}×超额BP + "
            f"{objective['capital_trade_win_rate_weight']:g}×交易胜率 + "
            f"{objective['capital_gain_avg_win_bp_weight']:g}×平均每笔盈利BP + "
            f"{objective['capital_gain_drawdown_bp_penalty']:g}×资本回撤BP + "
            f"{objective['total_return_weight']:g}×收益 + "
            f"{objective['excess_return_weight']:g}×超额收益 + "
            f"{objective['sharpe_weight']:g}×夏普 + "
            f"{objective['signal_win_rate_weight']:g}×周期胜率",
            language=None,
        )
    with st.sidebar.expander("仓位、权重与阈值（归档）"):
        st.json({
            "positions": config.positions.as_dict(),
            "weights": config.weights.as_dict(),
            "thresholds": config.thresholds.as_dict(),
        })


def _render_home_research_snapshot() -> None:
    with st.spinner("正在载入历史实验…"):
        experiments = list_experiments(ROOT)
    st.markdown(
        "<div class='section-head'><h2>研究快照</h2><p>周频与日频分别要求覆盖各自完整历史 · 点击策略名称查看归档结果</p></div>",
        unsafe_allow_html=True,
    )
    valid = experiments.dropna(subset=["累计资本利得_BP"]) if not experiments.empty else pd.DataFrame()
    if not valid.empty:
        valid = valid.loc[valid["比较基准"] == CONDITIONAL_BENCHMARK_NAME].copy()
    if not valid.empty:
        valid_rows = []
        for _, row in valid.iterrows():
            frequency = "daily" if row.get("信号频率") == "日频" else "weekly"
            full_start, full_end = _full_strategy_date_bounds(frequency)
            start = pd.to_datetime(row.get("回测起始日期"), errors="coerce")
            end = pd.to_datetime(row.get("回测结束日期"), errors="coerce")
            if pd.notna(start) and pd.notna(end) and start.strftime("%Y-%m-%d") == full_start and end.strftime("%Y-%m-%d") == full_end:
                valid_rows.append(row)
        valid = pd.DataFrame(valid_rows)
    if valid.empty:
        st.caption("尚无与当前数据源完整区间一致的归档实验。请用完整区间运行策略后再进行横向比较。")
        return
    valid["_仓位组合"] = valid["实验目录"].map(_snapshot_position_label)
    position_options = ["全部仓位", *sorted(valid["_仓位组合"].dropna().unique().tolist())]
    selected_position = str(st.session_state.get("home_snapshot_position_filter", "全部仓位"))
    if selected_position not in position_options:
        selected_position = "全部仓位"
        st.session_state["home_snapshot_position_filter"] = selected_position
    position_counts = valid["_仓位组合"].value_counts().to_dict()
    if selected_position != "全部仓位":
        valid = valid.loc[valid["_仓位组合"] == selected_position].copy()
    featured = _pick_featured_experiments(valid)
    slide_count = len(featured)
    cycle_seconds = max(slide_count * 3, 3)
    slot_pct = 100.0 / max(slide_count, 1)
    enter_pct = min(0.35 / cycle_seconds * 100.0, slot_pct * 0.22)
    hold_pct = max(slot_pct - enter_pct, enter_pct)
    animation_name = f"featuredCycle{slide_count}"
    slides = []
    for index, (advantage, highlight_keys, row) in enumerate(featured):
        advantage = f"{row.get('信号频率', '周频')} · {advantage}"
        experiment_id = quote(Path(str(row["实验目录"])).name)
        benchmark_name = str(row.get("比较基准") or CONDITIONAL_BENCHMARK_NAME)
        capital_bp = pd.to_numeric(pd.Series([row.get("累计资本利得_BP")]), errors="coerce").iloc[0]
        benchmark_capital_bp = pd.to_numeric(pd.Series([row.get("基准累计资本利得_BP")]), errors="coerce").iloc[0]
        excess_bp = pd.to_numeric(pd.Series([row.get("资本利得超额_BP")]), errors="coerce").iloc[0]
        annualized_bp = pd.to_numeric(pd.Series([row.get("年化资本利得_BP")]), errors="coerce").iloc[0]
        benchmark_annualized_bp = pd.to_numeric(pd.Series([row.get("基准年化资本利得_BP")]), errors="coerce").iloc[0]
        avg_loss_bp = pd.to_numeric(pd.Series([row.get("平均单笔亏损_BP")]), errors="coerce").iloc[0]
        avg_win_bp = pd.to_numeric(pd.Series([row.get("平均每笔盈利_BP")]), errors="coerce").iloc[0]
        worst_bp = pd.to_numeric(pd.Series([row.get("最差交易_BP")]), errors="coerce").iloc[0]
        drawdown_start = row.get("资本利得最大回撤起点")
        drawdown_end = row.get("资本利得最大回撤终点")
        average_holding_days = pd.to_numeric(pd.Series([row.get("平均每笔持有交易日")]), errors="coerce").iloc[0]
        max_holding_days = pd.to_numeric(pd.Series([row.get("最长单笔持有交易日")]), errors="coerce").iloc[0]
        winning_trades_text = _count_text(row.get("盈利交易数"))
        closed_trades_text = _count_text(row.get("已平仓交易数"))
        metrics = [
            ("capital", "累计资本利得", _bp_text(capital_bp), _bp_detail(benchmark_name, "累计", benchmark_capital_bp), False),
            ("win", "已平仓交易胜率", _pct(row.get("资本利得交易胜率")), f"盈利 {winning_trades_text} / 已平仓 {closed_trades_text} 笔", False),
            ("average", "平均每笔盈利", _bp_text(avg_win_bp), _payoff_detail(row.get("资本利得盈亏比")), False),
            ("capital_drawdown", "资本利得最大回撤", _bp_text(row.get("资本利得最大回撤_BP")), _drawdown_period_detail(drawdown_start, drawdown_end), False),
            ("excess", "超额资本利得", _bp_text(excess_bp), _relative_excess_detail(benchmark_name, excess_bp, benchmark_capital_bp), False),
            ("annual", "年化资本利得", _bp_text(annualized_bp), _bp_detail(benchmark_name, "", benchmark_annualized_bp), False),
            ("loss", "平均单笔亏损", "暂无亏损" if pd.isna(avg_loss_bp) else _bp_text(avg_loss_bp), "区间内无亏损交易" if pd.isna(avg_loss_bp) else ("最大亏损暂无" if pd.isna(worst_bp) or float(worst_bp) >= 0 else f"最大亏损 {float(worst_bp):.2f} BP"), False),
            ("holding", "平均每笔持有时间", "暂无交易" if pd.isna(average_holding_days) else f"{average_holding_days:.1f} 交易日", "最长持有暂无" if pd.isna(max_holding_days) else f"最长 {int(max_holding_days)} 交易日", False),
        ]
        metric_html = "".join(
            _featured_metric_html(label, value_text, detail, key in highlight_keys, compact)
            for key, label, value_text, detail, compact in metrics
        )
        slides.append(
            f'<article class="featured-slide" style="animation-name:{animation_name};animation-duration:{cycle_seconds}s;animation-delay:-{index * 3}s">'
            f'<div class="featured-slide-header"><div class="featured-strategy"><a href="/history?experiment={experiment_id}" target="_self">{escape(_run_display_name(Path(str(row["实验目录"])), row["策略名称"]))}</a></div>'
            f'<div class="featured-badge">{escape(advantage)}</div></div><div class="featured-metrics">{metric_html}</div></article>'
        )
    carousel_class = "featured-carousel single" if len(slides) <= 1 else "featured-carousel"
    animation_css = (
        f"<style>@keyframes {animation_name} {{0% {{opacity:0;pointer-events:none;transform:translateY(14px);filter:blur(2px)}} "
        f"{enter_pct:.2f}%, {hold_pct:.2f}% {{opacity:1;pointer-events:auto;transform:translateY(0);filter:blur(0)}} "
        f"{slot_pct:.2f}%, 99.8% {{opacity:0;pointer-events:none;transform:translateY(-9px);filter:blur(1.5px)}} "
        "100% {opacity:0;pointer-events:none;transform:translateY(14px);filter:blur(2px)}}</style>"
        if slide_count > 1 else ""
    )
    st.markdown(f'{animation_css}<section class="{carousel_class}">{"".join(slides)}</section>', unsafe_allow_html=True)
    st.segmented_control(
        "仓位组合",
        position_options,
        default="全部仓位",
        key="home_snapshot_position_filter",
        format_func=lambda value: (
            f"全部仓位 · {sum(position_counts.values())}"
            if value == "全部仓位"
            else f"{value} · {position_counts.get(value, 0)}"
        ),
        help="只比较相同多/中/空仓位制度下、覆盖完整历史区间的策略。",
    )


def _snapshot_position_label(experiment_dir: object) -> str:
    try:
        positions = load_strategy_config(Path(str(experiment_dir)) / "config.json").positions
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "仓位未知"
    return (
        f"多{_name_number(positions.bullish_position)} / "
        f"中{_name_number(positions.neutral_position)} / "
        f"空{_name_number(positions.bearish_position)}"
    )


def _pick_featured_experiments(frame: pd.DataFrame) -> list[tuple[str, set[str], pd.Series]]:
    latest = frame.sort_values("运行时间", ascending=False).drop_duplicates("策略名称").copy()
    for column in ["累计资本利得_BP", "年化资本利得_BP", "资本利得交易胜率", "平均每笔盈利_BP", "平均单笔亏损_BP", "资本利得最大回撤_BP", "已平仓交易数", "亏损交易数"]:
        latest[column] = pd.to_numeric(latest[column], errors="coerce")
    eligible = latest.loc[latest["已平仓交易数"].fillna(0) >= 3].copy()
    if eligible.empty:
        eligible = latest
    specifications = [
        ("累计资本利得领先", "capital", "累计资本利得_BP", False),
        ("年化资本利得领先", "annual", "年化资本利得_BP", False),
        ("已平仓胜率领先", "win", "资本利得交易胜率", False),
        ("平均每笔盈利领先", "average", "平均每笔盈利_BP", False),
        ("平均单笔亏损最小", "loss", "平均单笔亏损_BP", False),
        ("资本回撤绝对值最小", "capital_drawdown", "资本利得最大回撤_BP", True),
    ]
    selected: list[tuple[list[str], set[str], pd.Series]] = []
    selected_index: dict[str, int] = {}
    for advantage, highlight_key, column, absolute_minimum in specifications:
        if column == "平均单笔亏损_BP":
            candidates = eligible.loc[eligible["已平仓交易数"].fillna(0) > 0].copy()
            candidates["_no_loss"] = candidates["亏损交易数"].eq(0)
            candidates = candidates.loc[candidates["亏损交易数"].notna()].copy()
            candidates = candidates.sort_values(
                ["_no_loss", column, "已平仓交易数", "累计资本利得_BP"],
                ascending=[False, False, False, False],
                na_position="first",
            )
        else:
            candidates = eligible.dropna(subset=[column]).copy()
        if absolute_minimum and column != "平均单笔亏损_BP":
            candidates["_rank"] = candidates[column].abs()
            candidates = candidates.sort_values("_rank", ascending=True)
        elif column != "平均单笔亏损_BP":
            candidates = candidates.sort_values(column, ascending=False)
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        experiment_dir = str(row["实验目录"])
        if experiment_dir in selected_index:
            index = selected_index[experiment_dir]
            selected[index][0].append(advantage)
            selected[index][1].add(highlight_key)
        else:
            selected_index[experiment_dir] = len(selected)
            selected.append(([advantage], {highlight_key}, row))
    return [(" · ".join(advantages), highlights, row) for advantages, highlights, row in selected[:6]]


def _featured_metric_html(label: str, value_text: str, detail: str, highlighted: bool, compact: bool = False) -> str:
    highlight_class = " highlight" if highlighted else ""
    compact_class = " compact-value" if compact else ""
    return (
        f'<div class="featured-metric{highlight_class}{compact_class}"><div class="featured-metric-label">{escape(label)}</div>'
        f'<div class="featured-metric-value">{escape(value_text)}</div>'
        f'<div class="featured-metric-benchmark">{escape(detail)}</div></div>'
    )


def _bp_text(value: object) -> str:
    return "暂无" if value is None or pd.isna(value) else f"{float(value):.2f} BP"


def _count_text(value: object) -> str:
    """Format optional archive counts without converting NaN to int."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "暂无" if pd.isna(numeric) else f"{int(float(numeric))}"


def _bp_detail(name: str, prefix: str, value: object) -> str:
    return f"{name}暂无" if value is None or pd.isna(value) else f"{name}{prefix} {float(value):.2f} BP"


def _render_history_table(display: pd.DataFrame, *, initial_row_count: int | None = None) -> None:
    """Render a sortable archive list, progressively filling long histories.

    Streamlit's native ``LinkColumn`` is painted on a canvas.  It can only
    render the literal word ``bookmark`` rather than a styled icon, so the
    collection control needs a small DOM table of its own. The first rows are
    immediately usable; remaining indexed rows are appended in small browser
    batches so the initial history visit never waits for a large DOM table.
    """
    columns: list[tuple[str, str, str]] = [
        ("打开结果", "打开", "text"),
        ("收藏", "收藏", "text"),
        ("运行ID", "运行ID", "text"),
        ("运行时间", "运行时间", "date"),
        ("策略名称", "策略名称", "text"),
        ("备注", "备注", "text"),
        ("信号频率", "信号频率", "text"),
        ("比较基准", "比较基准", "text"),
        ("BP口径", "BP口径", "text"),
        ("运行来源", "运行来源", "text"),
        ("回测区间", "回测区间", "text"),
        ("样本训练区间", "样本训练区间", "text"),
        ("累计资本利得_BP", "累计资本利得", "number"),
        ("资本利得超额_BP", "资本利得超额", "number"),
        ("资本利得交易胜率", "交易胜率", "number"),
        ("已平仓交易数", "已平仓笔数", "number"),
        ("平均每笔盈利_BP", "平均每笔盈利", "number"),
        ("资本利得最大回撤_BP", "资本利得最大回撤", "number"),
        ("策略累计收益率", "策略累计收益", "number"),
        ("最大回撤", "最大回撤", "number"),
        ("夏普比率", "夏普比率", "number"),
        ("样本外累计资本利得_BP", "样本外累计资本利得", "number"),
        ("样本外交易胜率", "样本外交易胜率", "number"),
        ("样本外最大回撤_BP", "样本外最大回撤", "number"),
    ]

    def formatted(column: str, value: object) -> str:
        if value is None or pd.isna(value):
            return "—"
        if column.endswith("_BP"):
            return f"{float(value):.2f} BP"
        if column in {"资本利得交易胜率", "样本外交易胜率"}:
            return f"{float(value):.2%}"
        if column in {"策略累计收益率", "最大回撤"}:
            return f"{float(value):.2f}%"
        if column == "夏普比率":
            return f"{float(value):.3f}"
        if column == "已平仓交易数":
            return f"{int(value)}"
        return str(value)

    def render_row(row: pd.Series) -> str:
        cells: list[str] = []
        for column_index, (column, _, kind) in enumerate(columns):
            value = row[column]
            if column == "打开结果":
                cells.append(
                    f'<td data-column-index="{column_index}" data-sort=""><a class="history-open" href="{escape(str(value), quote=True)}" target="_blank" rel="noopener">查看</a></td>'
                )
                continue
            if column == "收藏":
                is_saved = "favorite_current=saved" in str(value)
                label = "已收藏，点击取消收藏" if is_saved else "收藏"
                # Do not rely on the Material icon font here.  This table lives
                # in a Streamlit component iframe, where that font is not
                # guaranteed to be inherited and would otherwise expose the
                # literal token ``bookmark_border``.  The two paths intentionally
                # use the same bookmark silhouette as the result-page control:
                # outline = not saved, filled = saved.
                if is_saved:
                    bookmark_svg = (
                        '<svg class="bookmark-icon" viewBox="0 0 24 24" aria-hidden="true">'
                        '<path d="M6 21V5.5Q6 4.675 6.588 4.088Q7.175 3.5 8 3.5H16Q16.825 3.5 17.413 4.088Q18 4.675 18 5.5V21L12 18.425Z"/>'
                        '</svg>'
                    )
                else:
                    bookmark_svg = (
                        '<svg class="bookmark-icon" viewBox="0 0 24 24" aria-hidden="true">'
                        '<path d="M6 21V5.5Q6 4.675 6.588 4.088Q7.175 3.5 8 3.5H16Q16.825 3.5 17.413 4.088Q18 4.675 18 5.5V21L12 18.425L6 21ZM8 17.95L12 16.25L16 17.95V5.5H8Z"/>'
                        '</svg>'
                    )
                saved_class = " is-saved" if is_saved else ""
                cells.append(
                    f'<td data-column-index="{column_index}" data-sort="{1 if is_saved else 0}"><a class="history-bookmark{saved_class}" '
                    f'href="{escape(str(value), quote=True)}" aria-label="{label}" title="{label}">{bookmark_svg}</a></td>'
                )
                continue
            raw = "" if value is None or pd.isna(value) else str(value)
            # Text columns previously received empty keys, so their headers
            # appeared inert. Every cell needs its source value for sorting.
            sort_value = raw
            if column == "策略名称":
                archive_name = Path(str(row.get("_实验目录", ""))).name
                rename_title = _normalise_display_name(raw)
                cells.append(
                    f'<td class="strategy-name" data-column-index="{column_index}" title="{escape(raw, quote=True)}" data-sort="{escape(raw, quote=True)}">'
                    f'<span class="strategy-title">{escape(formatted(column, value))}</span>'
                    f'<button class="history-rename" type="button" data-archive="{escape(archive_name, quote=True)}" '
                    f'data-title="{escape(rename_title, quote=True)}" aria-label="重命名 {escape(raw, quote=True)}" '
                    f'title="重命名"><span class="material-symbols-rounded" aria-hidden="true">edit</span></button></td>'
                )
                continue
            if column == "备注":
                archive_name = Path(str(row.get("_实验目录", ""))).name
                note = raw.strip()
                visible_note = note or "—"
                cells.append(
                    f'<td class="history-note" data-column-index="{column_index}" title="{escape(note or "未填写备注", quote=True)}" data-sort="{escape(note, quote=True)}">'
                    f'<span class="history-note-title">{escape(visible_note)}</span>'
                    f'<button class="history-note-edit" type="button" data-archive="{escape(archive_name, quote=True)}" '
                    f'data-note="{escape(note, quote=True)}" aria-label="编辑备注" title="编辑备注">'
                    f'<span class="material-symbols-rounded" aria-hidden="true">edit</span></button></td>'
                )
                continue
            cells.append(
                f'<td data-column-index="{column_index}" data-sort="{escape(sort_value, quote=True)}">{escape(formatted(column, value))}</td>'
            )
        return "<tr>" + "".join(cells) + "</tr>"

    all_rows = [render_row(row) for _, row in display.iterrows()]
    initial_count = len(all_rows) if initial_row_count is None else max(0, min(initial_row_count, len(all_rows)))
    initial_rows = all_rows[:initial_count]
    deferred_rows = all_rows[initial_count:]
    # Kept as JSON (not parsed table DOM) until the first 50 visible rows are
    # usable. This keeps the first history paint responsive.
    deferred_payload = json.dumps(deferred_rows, ensure_ascii=False).replace("</", "<\\/")

    headers = "".join(
        f'<th data-kind="{kind}" data-index="{index}" draggable="true">'
        f'<button class="history-sort" type="button">{escape(label)}<span class="sort-mark"></span></button>'
        '<span class="history-column-resize" aria-hidden="true"></span>'
        f'<button class="history-column-close" type="button" aria-label="隐藏{escape(label)}列" title="隐藏此列">'
        '<span class="material-symbols-rounded" aria-hidden="true">close</span></button></th>'
        for index, (_, label, kind) in enumerate(columns)
    )
    quick_compare_labels = [
        "打开", "收藏", "运行ID", "策略名称", "信号频率", "样本训练区间",
        "累计资本利得", "资本利得超额", "交易胜率", "已平仓笔数", "平均每笔盈利",
        "资本利得最大回撤", "样本外累计资本利得", "样本外交易胜率", "样本外最大回撤",
    ]
    column_labels = [label for _, label, _ in columns]
    table_html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0..1,0&display=swap');
      :root {{ --ink:#18201d; --muted:#66716c; --line:#d9ddd8; --paper:#f3f2ed; --green:#176b5b; --coral:#bb654f; }}
      * {{ box-sizing:border-box; }}
      body {{ margin:0; background:transparent; color:var(--ink); font-family:Geist,"Microsoft YaHei",Arial,sans-serif; }}
      .history-toolbar {{ display:flex; align-items:center; justify-content:space-between; gap:.6rem; margin:0 0 .45rem; }}
      .history-toolbar-actions {{ display:flex; align-items:center; gap:.35rem; flex-wrap:wrap; }}
      .history-toolbar-action {{ min-height:1.95rem; padding:0 .62rem; border:1px solid var(--line); border-radius:4px; background:#fbfaf6; color:#53605a; cursor:pointer; font:600 .73rem Geist,"Microsoft YaHei",sans-serif; white-space:nowrap; transition:background .14s ease,border-color .14s ease,color .14s ease; }}
      .history-toolbar-action:hover {{ border-color:#aeb8b1; background:#f0f3ed; color:var(--ink); }}
      .history-toolbar-action:active {{ transform:translateY(1px); }}
      .history-toolbar-action:focus-visible {{ outline:2px solid rgba(23,107,91,.35); outline-offset:1px; }}
      .history-column-restore {{ position:relative; }}
      #history-restore-columns {{ min-width:1.95rem; padding:0 .42rem; font-size:1rem; line-height:1; }}
      #history-restore-columns:disabled {{ border-color:#e0e3df; background:#f7f7f4; color:#bac0bb; cursor:default; transform:none; }}
      .history-restore-panel {{ position:absolute; z-index:12; top:calc(100% + .35rem); left:0; display:none; width:14rem; min-width:14rem; max-width:14rem; padding:.38rem; border:1px solid var(--line); border-radius:5px; background:#fbfaf6; box-shadow:0 10px 22px rgba(24,32,29,.14); }}
      .history-restore-panel.is-open {{ display:block; }}
      /* This panel sits inside a table component which has broad label/input
         rules. Lock each restore choice to a two-column row so label length
         can never shift its checkbox horizontally. */
      .history-restore-panel .history-restore-option {{ box-sizing:border-box; display:grid !important; grid-template-columns:1rem minmax(0,1fr); align-items:center; justify-content:start; width:100%; min-width:0; margin:0 !important; gap:.48rem; padding:.42rem .48rem; border-radius:3px; color:#4f5b56; cursor:pointer; font:500 .74rem Geist,"Microsoft YaHei",sans-serif; text-align:left; white-space:normal; }}
      .history-restore-panel .history-restore-option input[type="checkbox"] {{ box-sizing:border-box; display:block; width:.9rem; min-width:.9rem; height:.9rem; margin:0 !important; padding:0; justify-self:start; accent-color:var(--green); }}
      .history-restore-panel .history-restore-option span {{ display:block; min-width:0; justify-self:start; text-align:left !important; line-height:1.35; }}
      .history-restore-option:hover {{ background:#edf1ec; color:var(--ink); }}
      .history-toolbar input {{ width:min(18rem,100%); padding:.43rem .65rem; border:1px solid var(--line); border-radius:4px; background:#fbfaf6; color:var(--ink); font:500 .78rem Geist,"Microsoft YaHei",sans-serif; outline:none; }}
      .history-toolbar input:focus {{ border-color:#8d9a93; box-shadow:0 0 0 2px rgba(23,107,91,.12); }}
      .history-table-wrap {{ max-height:49rem; overflow:auto; border:1px solid var(--line); border-radius:5px; background:#fbfaf6; }}
      table {{ width:max-content; min-width:100%; border-collapse:collapse; font-size:.78rem; font-variant-numeric:tabular-nums; }}
      th {{ position:sticky; top:0; z-index:1; background:#e7ebe5; border-bottom:1px solid #cdd4cf; text-align:left; white-space:nowrap; }}
      th[draggable="true"] {{ cursor:grab; }} th.history-column-dragging {{ opacity:.46; cursor:grabbing; }} th.history-column-drop-before {{ box-shadow:inset 2px 0 0 var(--green); }} th.history-column-drop-after {{ box-shadow:inset -2px 0 0 var(--green); }}
      th .history-sort {{ display:flex; align-items:center; gap:.28rem; width:100%; padding:.62rem 1.9rem .62rem .7rem; color:#4f5b56; border:0; background:transparent; text-align:left; cursor:pointer; font:600 .73rem Geist,"Microsoft YaHei",sans-serif; }}
      th .history-sort:hover {{ color:var(--ink); background:#dfe5dd; }}
      .history-column-resize {{ position:absolute; z-index:5; top:0; right:-3px; width:7px; height:100%; cursor:col-resize; touch-action:none; }}
      .history-column-resize::after {{ content:''; position:absolute; top:22%; right:3px; width:1px; height:56%; background:transparent; transition:background .13s ease; }}
      th:hover .history-column-resize::after, .history-column-resize:hover::after {{ background:#9aa49d; }}
      .history-column-close {{ position:absolute; z-index:3; top:50%; right:.3rem; display:flex; align-items:center; justify-content:center; width:1.3rem; height:1.3rem; padding:0; color:#5f6a64; border:0; border-radius:3px; background:rgba(231,235,229,.82); cursor:pointer; opacity:0; pointer-events:none; transform:translateY(-50%); transition:opacity .13s ease,background .13s ease,color .13s ease,transform .13s ease; }}
      th:hover .history-column-close, .history-column-close:focus-visible {{ opacity:.58; pointer-events:auto; }}
      .history-column-close:hover {{ color:var(--ink); background:#d6ddd6; opacity:1; }} .history-column-close:active {{ transform:translateY(-50%) scale(.88); }}
      .history-column-close:focus-visible {{ outline:2px solid rgba(23,107,91,.34); outline-offset:1px; }}
      .sort-mark::after {{ content:'↕'; color:#98a19b; font-size:.7rem; }}
      th.asc .sort-mark::after {{ content:'↑'; color:var(--ink); }} th.desc .sort-mark::after {{ content:'↓'; color:var(--ink); }}
      .column-hidden {{ display:none !important; }}
      td {{ padding:.58rem .7rem; border-bottom:1px solid #e1e4df; background:rgba(251,250,246,.72); white-space:nowrap; }}
      th[data-index="4"], td.strategy-name {{ width:14rem; min-width:14rem; max-width:14rem; }}
      td.strategy-name {{ position:relative; overflow:visible; vertical-align:top; white-space:normal; }}
      .strategy-title {{ display:block; min-width:0; padding-right:1.85rem; overflow-wrap:anywhere; line-height:1.48; white-space:normal; }}
      .history-rename {{ position:absolute; z-index:2; top:50%; right:.34rem; display:flex; align-items:center; justify-content:center; box-sizing:border-box; width:1.55rem; min-width:1.55rem; height:1.55rem; min-height:1.55rem; padding:0; color:#989898; background:rgba(255,255,255,.88); border:1px solid rgba(222,222,222,.9); border-radius:4px; cursor:pointer; touch-action:manipulation; font:inherit; line-height:1; opacity:0; pointer-events:none; transform:translateY(-50%); transition:opacity .14s ease, color .14s ease, background .14s ease, border-color .14s ease, transform .14s ease; }}
      td.strategy-name:hover .history-rename, .history-rename:focus-visible {{ opacity:.88; pointer-events:auto; }}
      .history-rename:hover {{ color:var(--green); background:#f4f7f4; }} .history-rename:active {{ transform:translateY(-50%) scale(.9); }}
      .history-rename:focus-visible {{ outline:2px solid rgba(23,107,91,.38); outline-offset:1px; }}
      .history-rename .material-symbols-rounded {{ pointer-events:none; }}
      td.strategy-name.is-editing {{ overflow:visible; }} td.strategy-name.is-editing .strategy-title, td.strategy-name.is-editing .history-rename {{ opacity:0; pointer-events:none; }}
      .history-name-input {{ position:absolute; z-index:4; inset:.22rem .28rem; width:calc(100% - .56rem); padding:0 .42rem; border:1px solid var(--green); border-radius:4px; background:#fff; color:var(--ink); font:500 .77rem Geist,"Microsoft YaHei",sans-serif; outline:none; box-shadow:0 0 0 2px rgba(23,107,91,.12); }}
      td.history-note {{ position:relative; width:12rem; min-width:12rem; max-width:12rem; overflow:hidden; text-overflow:ellipsis; }}
      .history-note-title {{ display:block; min-height:1rem; overflow:hidden; padding-right:1.85rem; color:#59635f; text-overflow:ellipsis; white-space:nowrap; }}
      .history-note-title:empty {{ color:var(--muted); }}
      .history-note-edit {{ position:absolute; z-index:2; top:50%; right:.34rem; display:flex; align-items:center; justify-content:center; box-sizing:border-box; width:1.55rem; min-width:1.55rem; height:1.55rem; min-height:1.55rem; padding:0; color:#989898; background:rgba(255,255,255,.88); border:1px solid rgba(222,222,222,.9); border-radius:4px; cursor:pointer; touch-action:manipulation; font:inherit; line-height:1; opacity:0; pointer-events:none; transform:translateY(-50%); transition:opacity .14s ease,color .14s ease,background .14s ease,border-color .14s ease,transform .14s ease; }}
      td.history-note:hover .history-note-edit, .history-note-edit:focus-visible {{ opacity:.88; pointer-events:auto; }}
      .history-note-edit:hover {{ color:var(--green); background:#f4f7f4; }} .history-note-edit:active {{ transform:translateY(-50%) scale(.9); }}
      .history-note-edit:focus-visible {{ outline:2px solid rgba(23,107,91,.38); outline-offset:1px; }}
      .history-note-edit .material-symbols-rounded {{ pointer-events:none; }}
      td.history-note.is-editing {{ overflow:visible; }} td.history-note.is-editing .history-note-title, td.history-note.is-editing .history-note-edit {{ opacity:0; pointer-events:none; }}
      .history-note-input {{ position:absolute; z-index:4; inset:.22rem .28rem; width:calc(100% - .56rem); padding:0 .42rem; border:1px solid var(--green); border-radius:4px; background:#fff; color:var(--ink); font:500 .77rem Geist,"Microsoft YaHei",sans-serif; outline:none; box-shadow:0 0 0 2px rgba(23,107,91,.12); }}
      .material-symbols-rounded {{ font-family:'Material Symbols Rounded'; font-size:1rem; font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 20; }}
      tbody tr:hover td {{ background:#f0f3ed; }} tbody tr:last-child td {{ border-bottom:0; }}
      .history-open {{ color:#2b5f54; text-decoration:none; font-weight:600; }} .history-open:hover {{ color:#143e36; text-decoration:underline; }}
      .history-bookmark {{ display:inline-flex; align-items:center; justify-content:center; width:1.4rem; height:1.4rem; color:var(--coral); text-decoration:none; transition:color .18s ease, transform .18s ease; }}
      .history-bookmark:hover {{ color:#a94f3d; }} .history-bookmark:active {{ transform:scale(.88); }}
      .history-bookmark:focus-visible {{ outline:2px solid rgba(187,101,79,.45); outline-offset:2px; }}
      .bookmark-icon {{ width:1.05rem; height:1.05rem; display:block; fill:currentColor; }}
      #history-load-status {{ display:none; }}
      #history-favorite-writer, #history-rename-writer, #history-note-writer {{ display:none; width:0; height:0; border:0; }}
      .history-rename-confirm {{ display:none; position:fixed; inset:0; z-index:20; align-items:center; justify-content:center; padding:1rem; background:rgba(24,32,29,.18); }}
      .history-rename-confirm.is-open {{ display:flex; }}
      .history-rename-confirm section {{ width:min(22rem,100%); padding:1rem; border:1px solid var(--line); border-radius:6px; background:#fbfaf6; box-shadow:0 14px 30px rgba(24,32,29,.16); }}
      .history-rename-confirm strong {{ display:block; font-size:.88rem; }} .history-rename-confirm p {{ margin:.32rem 0 .78rem; color:var(--muted); font-size:.74rem; line-height:1.5; overflow-wrap:anywhere; }}
      .history-rename-confirm section > div {{ display:flex; justify-content:flex-end; gap:.4rem; }}
      .history-rename-confirm button {{ min-height:1.9rem; padding:0 .62rem; border:1px solid #d5d9d5; border-radius:4px; background:#f7f7f7; color:#59635f; cursor:pointer; font:600 .73rem Geist,"Microsoft YaHei",sans-serif; }}
      .history-rename-confirm button.primary {{ border-color:var(--green); background:var(--green); color:#fff; }}
      .no-match td {{ padding:1.15rem .7rem; color:var(--muted); text-align:center; }}
    </style>
    <div class="history-toolbar">
      <div class="history-toolbar-actions">
        <button id="history-quick-compare" class="history-toolbar-action" type="button">快速比较</button>
        <button id="history-show-all" class="history-toolbar-action" type="button">显示全部</button>
        <div class="history-column-restore">
          <button id="history-restore-columns" class="history-toolbar-action" type="button" aria-label="恢复已隐藏列" title="恢复已隐藏列" aria-expanded="false">+</button>
          <div id="history-restore-panel" class="history-restore-panel" role="menu" aria-label="恢复已隐藏列"></div>
        </div>
      </div>
      <input id="history-search" placeholder="筛选运行ID、策略名称或来源" aria-label="筛选历史实验">
    </div>
    <div class="history-table-wrap"><table id="history-table"><thead><tr>{headers}</tr></thead><tbody>{''.join(initial_rows)}</tbody></table></div>
    <div id="history-load-status" aria-live="polite"></div>
    <iframe id="history-favorite-writer" title="收藏状态写入" aria-hidden="true"></iframe>
    <iframe id="history-rename-writer" title="名称写入" aria-hidden="true"></iframe>
    <iframe id="history-note-writer" title="备注写入" aria-hidden="true"></iframe>
    <div id="history-rename-confirm" class="history-rename-confirm" role="dialog" aria-modal="true" aria-label="确认修改名称">
      <section><strong>确认修改名称？</strong><p id="history-rename-confirm-text"></p><div><button type="button" data-action="cancel">取消</button><button type="button" class="primary" data-action="save">确认修改</button></div></section>
    </div>
    <div id="history-note-confirm" class="history-rename-confirm" role="dialog" aria-modal="true" aria-label="确认修改备注">
      <section><strong>确认修改备注？</strong><p id="history-note-confirm-text"></p><div><button type="button" data-action="cancel">取消</button><button type="button" class="primary" data-action="save">确认修改</button></div></section>
    </div>
    <script>
      const table = document.getElementById('history-table');
      const body = table.tBodies[0];
      const search = document.getElementById('history-search');
      const status = document.getElementById('history-load-status');
      const deferredRows = {deferred_payload};
      const totalRows = {len(all_rows)};
      const columnLabels = {json.dumps(column_labels, ensure_ascii=False)};
      const quickCompareLabels = new Set({json.dumps(quick_compare_labels, ensure_ascii=False)});
      const batchSize = 25;
      let sortedColumn = -1, ascending = true;
      const outlineBookmark = '<svg class="bookmark-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 21V5.5Q6 4.675 6.588 4.088Q7.175 3.5 8 3.5H16Q16.825 3.5 17.413 4.088Q18 4.675 18 5.5V21L12 18.425L6 21ZM8 17.95L12 16.25L16 17.95V5.5H8Z"/></svg>';
      const filledBookmark = '<svg class="bookmark-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 21V5.5Q6 4.675 6.588 4.088Q7.175 3.5 8 3.5H16Q16.825 3.5 17.413 4.088Q18 4.675 18 5.5V21L12 18.425Z"/></svg>';
      const historyStorage = (() => {{ try {{ return window.parent.localStorage; }} catch (_) {{ return null; }} }})();
      const cacheKey = 'local-bond-history-table-complete-v1';
      const columnStorageKey = 'local-bond-history-table-hidden-columns-v1';
      const columnLayoutStorageKey = 'local-bond-history-table-layout-v1';
      const restoreColumnsButton = document.getElementById('history-restore-columns');
      const restoreColumnsPanel = document.getElementById('history-restore-panel');
      const headerRow = table.tHead.rows[0];
      const columnIndex = node => Number(node && node.dataset ? node.dataset.index ?? node.dataset.columnIndex : -1);
      const allColumnIndexes = () => columnLabels.map((_, index) => index);
      let hiddenColumns = new Set();
      let columnOrder = allColumnIndexes();
      let columnWidths = {{}};
      try {{
        const savedColumns = historyStorage && JSON.parse(historyStorage.getItem(columnStorageKey) || '[]');
        if (Array.isArray(savedColumns)) hiddenColumns = new Set(savedColumns.filter(index => Number.isInteger(index) && index >= 0 && index < columnLabels.length));
      }} catch (_) {{ hiddenColumns = new Set(); }}
      try {{
        const savedLayout = historyStorage && JSON.parse(historyStorage.getItem(columnLayoutStorageKey) || '{{}}');
        if (Array.isArray(savedLayout.order)) {{
          const valid = savedLayout.order.filter(index => Number.isInteger(index) && index >= 0 && index < columnLabels.length);
          if (new Set(valid).size === columnLabels.length) columnOrder = valid;
        }}
        if (savedLayout.widths && typeof savedLayout.widths === 'object') {{
          columnWidths = Object.fromEntries(Object.entries(savedLayout.widths).filter(([index, width]) =>
            Number.isInteger(Number(index)) && Number(index) >= 0 && Number(index) < columnLabels.length &&
            Number.isFinite(Number(width)) && Number(width) >= 64 && Number(width) <= 720
          ));
        }}
      }} catch (_) {{ columnOrder = allColumnIndexes(); columnWidths = {{}}; }}
      const persistColumns = () => {{ if (historyStorage) historyStorage.setItem(columnStorageKey, JSON.stringify([...hiddenColumns])); }};
      const persistColumnLayout = () => {{
        if (historyStorage) historyStorage.setItem(columnLayoutStorageKey, JSON.stringify({{ order: columnOrder, widths: columnWidths }}));
      }};
      const renderRestoreOptions = () => {{
        const hidden = [...hiddenColumns].sort((left, right) => left - right);
        restoreColumnsButton.disabled = hidden.length === 0;
        restoreColumnsButton.title = hidden.length ? '恢复已隐藏列' : '暂无隐藏列';
        restoreColumnsButton.setAttribute('aria-expanded', String(restoreColumnsPanel.classList.contains('is-open') && hidden.length > 0));
        restoreColumnsPanel.innerHTML = hidden.map(index =>
          `<label class="history-restore-option"><input type="checkbox" data-column-index="${{index}}"><span>${{columnLabels[index]}}</span></label>`
        ).join('');
        if (!hidden.length) restoreColumnsPanel.classList.remove('is-open');
      }};
      const applyColumnVisibility = () => {{
        table.querySelectorAll('th').forEach(header => header.classList.toggle('column-hidden', hiddenColumns.has(columnIndex(header))));
        body.querySelectorAll('tr:not(.no-match)').forEach(row => {{
          [...row.cells].forEach(cell => cell.classList.toggle('column-hidden', hiddenColumns.has(columnIndex(cell))));
        }});
        renderRestoreOptions();
      }};
      const applyColumnWidths = () => {{
        allColumnIndexes().forEach(index => {{
          const width = Number(columnWidths[index]);
          if (!Number.isFinite(width)) return;
          table.querySelectorAll(`[data-index="${{index}}"], td[data-column-index="${{index}}"]`).forEach(cell => {{
            cell.style.width = width + 'px';
            cell.style.minWidth = width + 'px';
            cell.style.maxWidth = width + 'px';
          }});
        }});
      }};
      const applyColumnOrder = () => {{
        const headerByIndex = new Map([...headerRow.cells].map(header => [columnIndex(header), header]));
        columnOrder.forEach(index => {{ const header = headerByIndex.get(index); if (header) headerRow.appendChild(header); }});
        body.querySelectorAll('tr:not(.no-match)').forEach(row => {{
          const cellByIndex = new Map([...row.cells].map(cell => [columnIndex(cell), cell]));
          columnOrder.forEach(index => {{ const cell = cellByIndex.get(index); if (cell) row.appendChild(cell); }});
        }});
        applyColumnWidths();
        applyColumnVisibility();
      }};
      const parseValue = (value, kind) => kind === 'number' ? (Number.parseFloat(value) || -Infinity) : String(value || '').toLowerCase();
      const sortRows = () => {{
        if (sortedColumn < 0) return;
        const header = headerRow.querySelector(`th[data-index="${{sortedColumn}}"]`);
        if (!header) return;
        const kind = header.dataset.kind;
        [...body.querySelectorAll('tr:not(.no-match)')].sort((left, right) => {{
          const leftCell = left.querySelector(`td[data-column-index="${{sortedColumn}}"]`);
          const rightCell = right.querySelector(`td[data-column-index="${{sortedColumn}}"]`);
          const a = parseValue(leftCell && leftCell.dataset.sort, kind), b = parseValue(rightCell && rightCell.dataset.sort, kind);
          return (a > b ? 1 : a < b ? -1 : 0) * (ascending ? 1 : -1);
        }}).forEach(row => body.appendChild(row));
      }};
      const applyFilter = () => {{
        const query = search.value.trim().toLowerCase();
        body.querySelectorAll('tr:not(.no-match)').forEach(row => {{ row.hidden = Boolean(query) && !row.innerText.toLowerCase().includes(query); }});
      }};
      const publishStatus = (text, loaded) => {{
        try {{
          const parentStatus = window.parent.document.getElementById('history-load-status-parent');
          if (parentStatus) {{ parentStatus.textContent = text; parentStatus.dataset.loaded = String(loaded); }}
        }} catch (_) {{ /* The table remains usable if the host blocks DOM access. */ }}
      }};
      const updateStatus = () => {{
        const loaded = body.querySelectorAll('tr:not(.no-match)').length;
        const text = `已加载 ${{loaded}} / ${{totalRows}} 条`;
        status.textContent = text;
        publishStatus(text, loaded);
      }};
      let draggedColumn = null;
      const clearDropTargets = () => headerRow.querySelectorAll('th').forEach(header => header.classList.remove('history-column-drop-before', 'history-column-drop-after'));
      const moveColumn = (sourceIndex, targetIndex, insertAfter) => {{
        if (sourceIndex === targetIndex) return;
        const sourcePosition = columnOrder.indexOf(sourceIndex);
        const targetPosition = columnOrder.indexOf(targetIndex);
        if (sourcePosition < 0 || targetPosition < 0) return;
        columnOrder.splice(sourcePosition, 1);
        const nextTargetPosition = columnOrder.indexOf(targetIndex);
        columnOrder.splice(nextTargetPosition + (insertAfter ? 1 : 0), 0, sourceIndex);
        persistColumnLayout();
        applyColumnOrder();
      }};
      headerRow.addEventListener('dragstart', event => {{
        if (event.target.closest('.history-column-close, .history-column-resize')) {{ event.preventDefault(); return; }}
        const header = event.target.closest('th[data-index]');
        if (!header) return;
        draggedColumn = columnIndex(header);
        header.classList.add('history-column-dragging');
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', String(draggedColumn));
      }});
      headerRow.addEventListener('dragover', event => {{
        if (draggedColumn === null) return;
        const header = event.target.closest('th[data-index]');
        if (!header || columnIndex(header) === draggedColumn) return;
        event.preventDefault();
        clearDropTargets();
        const before = event.clientX < header.getBoundingClientRect().left + header.offsetWidth / 2;
        header.classList.add(before ? 'history-column-drop-before' : 'history-column-drop-after');
      }});
      headerRow.addEventListener('drop', event => {{
        const header = event.target.closest('th[data-index]');
        if (draggedColumn === null || !header) return;
        event.preventDefault();
        const insertAfter = event.clientX >= header.getBoundingClientRect().left + header.offsetWidth / 2;
        moveColumn(draggedColumn, columnIndex(header), insertAfter);
        clearDropTargets();
        draggedColumn = null;
      }});
      headerRow.addEventListener('dragend', () => {{
        headerRow.querySelectorAll('.history-column-dragging').forEach(header => header.classList.remove('history-column-dragging'));
        clearDropTargets();
        draggedColumn = null;
      }});
      table.querySelectorAll('th').forEach(header => {{
        const index = columnIndex(header);
        header.querySelector('.history-sort').addEventListener('click', () => {{
          ascending = sortedColumn === index ? !ascending : true; sortedColumn = index;
          sortRows();
          table.querySelectorAll('th').forEach(th => th.classList.remove('asc','desc')); header.classList.add(ascending ? 'asc' : 'desc');
        }});
        header.querySelector('.history-column-close').addEventListener('click', event => {{
          event.preventDefault(); event.stopPropagation();
          hiddenColumns.add(index); persistColumns(); applyColumnVisibility();
        }});
        const resizeHandle = header.querySelector('.history-column-resize');
        resizeHandle.addEventListener('pointerdown', event => {{
          event.preventDefault(); event.stopPropagation();
          const startX = event.clientX;
          const startWidth = header.getBoundingClientRect().width;
          resizeHandle.setPointerCapture(event.pointerId);
          const resize = moveEvent => {{
            const nextWidth = Math.max(64, Math.min(720, Math.round(startWidth + moveEvent.clientX - startX)));
            columnWidths[index] = nextWidth;
            applyColumnWidths();
          }};
          const stopResize = moveEvent => {{
            resizeHandle.removeEventListener('pointermove', resize);
            resizeHandle.removeEventListener('pointerup', stopResize);
            resizeHandle.removeEventListener('pointercancel', stopResize);
            if (resizeHandle.hasPointerCapture(moveEvent.pointerId)) resizeHandle.releasePointerCapture(moveEvent.pointerId);
            persistColumnLayout();
          }};
          resizeHandle.addEventListener('pointermove', resize);
          resizeHandle.addEventListener('pointerup', stopResize);
          resizeHandle.addEventListener('pointercancel', stopResize);
        }});
      }});
      restoreColumnsButton.addEventListener('click', () => {{
        if (restoreColumnsButton.disabled) return;
        const willOpen = !restoreColumnsPanel.classList.contains('is-open');
        restoreColumnsPanel.classList.toggle('is-open', willOpen);
        restoreColumnsButton.setAttribute('aria-expanded', String(willOpen));
      }});
      restoreColumnsPanel.addEventListener('change', event => {{
        const input = event.target.closest('input[data-column-index]');
        if (!input || !input.checked) return;
        hiddenColumns.delete(Number(input.dataset.columnIndex));
        persistColumns(); applyColumnVisibility();
      }});
      document.addEventListener('click', event => {{
        if (!event.target.closest('.history-column-restore')) {{
          restoreColumnsPanel.classList.remove('is-open');
          restoreColumnsButton.setAttribute('aria-expanded', 'false');
        }}
      }});
      search.addEventListener('input', () => {{
        applyFilter();
      }});
      document.getElementById('history-quick-compare').addEventListener('click', () => {{
        hiddenColumns = new Set(columnLabels.map((label, index) => quickCompareLabels.has(label) ? -1 : index).filter(index => index >= 0));
        persistColumns(); applyColumnVisibility();
      }});
      document.getElementById('history-show-all').addEventListener('click', () => {{
        hiddenColumns.clear(); persistColumns(); applyColumnVisibility();
      }});
      body.addEventListener('click', event => {{
        const bookmark = event.target.closest('.history-bookmark');
        if (bookmark) {{
          event.preventDefault();
          const isSaved = bookmark.classList.toggle('is-saved');
          bookmark.innerHTML = isSaved ? filledBookmark : outlineBookmark;
          bookmark.setAttribute('aria-label', isSaved ? '已收藏，点击取消收藏' : '收藏');
          bookmark.setAttribute('title', isSaved ? '已收藏，点击取消收藏' : '收藏');
          bookmark.closest('td').dataset.sort = isSaved ? '1' : '0';
          sortRows();
          // Write in a hidden document rather than navigating this component.
          // The visible table stays interactive, so no nested page or whole-page
          // refresh appears when a bookmark is toggled.
          document.getElementById('history-favorite-writer').src = bookmark.href;
          return;
        }}
        const rename = event.target.closest('.history-rename');
        if (rename) {{
          event.preventDefault();
          const cell = rename.closest('td.strategy-name');
          if (cell.classList.contains('is-editing')) return;
          beginRename(cell, rename);
          return;
        }}
        const noteEdit = event.target.closest('.history-note-edit');
        if (!noteEdit) return;
        event.preventDefault();
        const noteCell = noteEdit.closest('td.history-note');
        if (noteCell.classList.contains('is-editing')) return;
        beginNote(noteCell, noteEdit);
      }});
      const renameConfirm = document.getElementById('history-rename-confirm');
      const renameConfirmText = document.getElementById('history-rename-confirm-text');
      let pendingRename = null;
      let awaitingRenameConfirmation = false;
      const archivePrefix = value => (String(value || '').match(/^\\[[A-Za-z]\\d{{3,}}\\]\\s*/) || [''])[0];
      const cleanTitle = value => String(value || '').replace(/^\\s*\\[[A-Za-z]\\d{{3,}}\\]\\s*/, '').trim();
      const finishRename = (save) => {{
        if (!pendingRename) return;
        const {{ cell, button, input, title }} = pendingRename;
        const titleNode = cell.querySelector('.strategy-title');
        if (save) {{
          const nextDisplay = archivePrefix(titleNode.textContent) + title;
          titleNode.textContent = nextDisplay;
          cell.title = nextDisplay;
          cell.dataset.sort = nextDisplay;
          button.dataset.title = title;
          button.setAttribute('aria-label', '重命名 ' + nextDisplay);
          document.getElementById('history-rename-writer').src = '/history?rename_save=' + encodeURIComponent(button.dataset.archive) + '&rename_title=' + encodeURIComponent(title) + '&_=' + Date.now();
        }}
        input.remove();
        cell.classList.remove('is-editing');
        renameConfirm.classList.remove('is-open');
        pendingRename = null;
        awaitingRenameConfirmation = false;
      }};
      const requestRenameConfirmation = () => {{
        if (!pendingRename || awaitingRenameConfirmation) return;
        const nextTitle = pendingRename.input.value.trim();
        if (!nextTitle || nextTitle === pendingRename.button.dataset.title) {{ finishRename(false); return; }}
        pendingRename.title = nextTitle;
        awaitingRenameConfirmation = true;
        renameConfirmText.textContent = '将显示名称修改为“' + nextTitle + '”？';
        renameConfirm.classList.add('is-open');
      }};
      const beginRename = (cell, button) => {{
        if (pendingRename) finishRename(false);
        const input = document.createElement('input');
        input.type = 'text'; input.maxLength = 80; input.className = 'history-name-input';
        // The table text is the single source of truth for the current name.
        // Reading it here prevents a stale data attribute from ever producing
        // a placeholder-like value in the inline editor.
        const currentTitle = cleanTitle(cell.querySelector('.strategy-title').textContent);
        input.value = currentTitle;
        input.defaultValue = currentTitle;
        input.setAttribute('aria-label', '编辑策略名称');
        cell.appendChild(input);
        cell.classList.add('is-editing');
        pendingRename = {{ cell, button, input, title: input.value.trim() }};
        input.addEventListener('keydown', event => {{
          if (event.key === 'Enter') {{ event.preventDefault(); requestRenameConfirmation(); }}
          if (event.key === 'Escape') {{ event.preventDefault(); finishRename(false); }}
        }});
        input.addEventListener('blur', () => window.setTimeout(requestRenameConfirmation, 0), {{ once: true }});
        input.focus(); input.select();
      }};
      renameConfirm.addEventListener('click', event => {{
        if (event.target === renameConfirm || event.target.closest('[data-action="cancel"]')) finishRename(false);
        if (event.target.closest('[data-action="save"]')) finishRename(true);
      }});
      const noteConfirm = document.getElementById('history-note-confirm');
      const noteConfirmText = document.getElementById('history-note-confirm-text');
      let pendingNote = null;
      let awaitingNoteConfirmation = false;
      const finishNote = (save) => {{
        if (!pendingNote) return;
        const {{ cell, button, input, note }} = pendingNote;
        const noteNode = cell.querySelector('.history-note-title');
        if (save) {{
          noteNode.textContent = note || '—';
          cell.title = note || '未填写备注';
          cell.dataset.sort = note;
          button.dataset.note = note;
          document.getElementById('history-note-writer').src = '/history?note_save=' + encodeURIComponent(button.dataset.archive) + '&note_text=' + encodeURIComponent(note) + '&_=' + Date.now();
        }}
        input.remove();
        cell.classList.remove('is-editing');
        noteConfirm.classList.remove('is-open');
        pendingNote = null;
        awaitingNoteConfirmation = false;
      }};
      const requestNoteConfirmation = () => {{
        if (!pendingNote || awaitingNoteConfirmation) return;
        const nextNote = pendingNote.input.value.trim();
        if (nextNote === pendingNote.button.dataset.note) {{ finishNote(false); return; }}
        pendingNote.note = nextNote;
        awaitingNoteConfirmation = true;
        noteConfirmText.textContent = nextNote ? '将备注修改为“' + nextNote + '”？' : '清空该策略的备注？';
        noteConfirm.classList.add('is-open');
      }};
      const beginNote = (cell, button) => {{
        if (pendingRename) finishRename(false);
        if (pendingNote) finishNote(false);
        const input = document.createElement('input');
        input.type = 'text'; input.maxLength = 280; input.className = 'history-note-input';
        const currentNote = button.dataset.note || '';
        input.value = currentNote;
        input.defaultValue = currentNote;
        input.setAttribute('aria-label', '编辑备注');
        cell.appendChild(input);
        cell.classList.add('is-editing');
        pendingNote = {{ cell, button, input, note: currentNote }};
        input.addEventListener('keydown', event => {{
          if (event.key === 'Enter') {{ event.preventDefault(); requestNoteConfirmation(); }}
          if (event.key === 'Escape') {{ event.preventDefault(); finishNote(false); }}
        }});
        input.addEventListener('blur', () => window.setTimeout(requestNoteConfirmation, 0), {{ once: true }});
        input.focus(); input.select();
      }};
      noteConfirm.addEventListener('click', event => {{
        if (event.target === noteConfirm || event.target.closest('[data-action="cancel"]')) finishNote(false);
        if (event.target.closest('[data-action="save"]')) finishNote(true);
      }});
      const appendBatch = () => {{
        const batch = deferredRows.splice(0, batchSize);
        if (batch.length) {{ body.insertAdjacentHTML('beforeend', batch.join('')); applyColumnOrder(); applyFilter(); sortRows(); updateStatus(); window.setTimeout(appendBatch, 45); return; }}
        if (historyStorage) historyStorage.setItem(cacheKey, '1');
        updateStatus();
      }};
      const appendAll = () => {{
        if (deferredRows.length) {{ body.insertAdjacentHTML('beforeend', deferredRows.splice(0).join('')); applyColumnOrder(); applyFilter(); sortRows(); }}
        updateStatus();
      }};
      applyColumnOrder();
      updateStatus();
      if (deferredRows.length) {{
        if (historyStorage && historyStorage.getItem(cacheKey) === '1') appendAll();
        else window.setTimeout(appendBatch, 35);
      }}
    </script>
    """
    # ``st.html`` paints the markup but does not reliably retain script event
    # handlers across Streamlit reruns. The component host does: it keeps the
    # table's header sorting and progressive browser-side row insertion alive.
    # Links explicitly use ``_blank``, so opening an archive still leaves the
    # embedded component and its own scroll state behind.
    # Leave enough room for the status line immediately below the scrollable
    # table.  At 820px the iframe clipped that line once the table reached its
    # 49rem internal scroll area, so users could not see the live progress.
    components.html(table_html, height=900, scrolling=False)


def _render_experiment_history(show_report: bool = False) -> None:
    # The table's visible bookmark is updated in place.  Its hidden writer
    # opens this action route so the saved choice persists without reloading
    # the table component.
    favorite_target = str(st.query_params.get("favorite", "")).strip()
    if favorite_target:
        target_dir = ROOT / "backtest_outputs" / "experiments" / Path(favorite_target).name
        if (target_dir / "run_manifest.json").exists():
            requested_state = str(st.query_params.get("favorite_set", "")).strip().lower()
            if requested_state in {"saved", "empty"}:
                _set_experiment_favorite(target_dir, requested_state == "saved")
            else:
                _set_experiment_favorite(target_dir, target_dir.name not in _favorite_experiment_ids())
        st.query_params.pop("favorite", None)
        st.query_params.pop("favorite_state", None)
        st.query_params.pop("favorite_current", None)
        st.query_params.pop("favorite_set", None)
        st.rerun()
    experiments = list_experiments(ROOT)
    if experiments.empty:
        st.caption("尚无已归档实验。运行一次回测后，这里会保留配置、数据版本、指标、诊断和HTML报告。")
        return

    favorites = _favorite_experiment_ids()
    notes = _experiment_notes()
    # This is a result-list filter, not a bookmark action.  Spell it out so
    # the toolbar remains understandable instead of exposing the icon-font
    # token ("bookmark") as visible text.
    only_favorites = st.toggle("仅看收藏", value=False, key="history_only_favorites")
    table_source = experiments.copy()
    table_source["策略名称"] = [
        _run_display_name(Path(str(path)), name)
        for path, name in zip(table_source["实验目录"], table_source["策略名称"])
    ]
    table_source["备注"] = [
        notes.get(Path(str(path)).name, "")
        for path in table_source["实验目录"]
    ]
    if only_favorites:
        table_source = table_source.loc[
            table_source["实验目录"].map(lambda value: Path(str(value)).name in favorites)
        ].copy()
        if table_source.empty:
            st.caption("暂无收藏记录。可在任一结果页点击书签图标。")
            return

    archive_query = st.text_input(
        "检索历史实验",
        placeholder="输入运行ID、策略名称或来源",
        key="history_archive_query",
        label_visibility="collapsed",
    ).strip().lower()
    if archive_query:
        query_mask = table_source.astype(str).apply(lambda column: column.str.lower().str.contains(archive_query, regex=False)).any(axis=1)
        table_source = table_source.loc[query_mask].copy()

    first_paint_rows = 50
    # Search and the favorites-only view are usually compact, and should show
    # their complete result directly. The unfiltered archive first paints 50
    # rows, then lets the browser append the rest from the already-loaded index.
    progressive_history = not archive_query and not only_favorites
    display = table_source.copy()
    st.caption("运行ID规则：B = 普通回测，S = 参数搜索，R = 滚动定参，F = 因子研究及其策略结果；后三位为该类记录的永久序号。")
    display.insert(
        0,
        "打开结果",
        [f"/history?experiment={quote(Path(str(path)).name)}" for path in display["实验目录"]],
    )
    display.insert(
        1,
        "收藏",
        [
            f"/history?favorite={quote(Path(str(path)).name)}"
            f"&favorite_current={'saved' if Path(str(path)).name in favorites else 'empty'}"
            f"&favorite_set={'empty' if Path(str(path)).name in favorites else 'saved'}"
            for path in display["实验目录"]
        ],
    )
    display = display.rename(columns={"实验目录": "_实验目录"})
    display = display[
        [
            "打开结果", "收藏", "运行ID", "运行时间", "策略名称", "备注", "信号频率", "比较基准", "BP口径", "运行来源", "回测区间", "样本训练区间", "累计资本利得_BP", "资本利得超额_BP",
            "资本利得交易胜率", "已平仓交易数", "平均每笔盈利_BP", "资本利得最大回撤_BP",
            "策略累计收益率", "最大回撤", "夏普比率", "样本外累计资本利得_BP", "样本外交易胜率", "样本外最大回撤_BP", "_实验目录",
        ]
    ]
    numeric_columns = [
        "累计资本利得_BP", "资本利得超额_BP", "平均每笔盈利_BP", "资本利得最大回撤_BP", "样本外累计资本利得_BP",
        "策略累计收益率", "最大回撤", "资本利得交易胜率", "样本外交易胜率", "样本外最大回撤_BP", "夏普比率",
    ]
    for column in numeric_columns:
        display[column] = pd.to_numeric(display[column], errors="coerce")
    _render_history_table(display, initial_row_count=first_paint_rows if progressive_history else None)
    initial_loaded = min(first_paint_rows, len(display)) if progressive_history else len(display)
    st.markdown(
        f'<div id="history-load-status-parent" class="history-load-status-parent" '
        f'aria-live="polite" data-loaded="{initial_loaded}">已加载 {initial_loaded} / {len(display)} 条</div>',
        unsafe_allow_html=True,
    )

    options = experiments["实验目录"].astype(str).tolist()
    selected = st.selectbox(
        "下载指定实验文件",
        options,
        format_func=lambda value: Path(value).name,
        key="experiment_history_result",
    )
    report_path = Path(selected) / "performance_report.html"
    manifest_path = Path(selected) / "run_manifest.json"
    download_col, manifest_col = st.columns(2)
    if report_path.exists():
        download_col.download_button(
            "下载HTML报告",
            data=report_path.read_bytes(),
            file_name=f"{Path(selected).name}_回测报告.html",
            mime="text/html",
            use_container_width=True,
        )
    if manifest_path.exists():
        manifest_col.download_button(
            "下载运行清单",
            data=manifest_path.read_bytes(),
            file_name=f"{Path(selected).name}_运行清单.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_search_page() -> None:
    st.markdown(
        "<section class='hero-shell'><div><p class='eyebrow'>批量研究 / 向量化计算</p><h1 class='hero-title'>搜索研究</h1><p class='hero-copy'>单策略控制台用于手动调参；这里运行权重搜索和阈值搜索，并直接查看各自的综合HTML报告。</p></div></section>",
        unsafe_allow_html=True,
    )
    st.caption("使用现有因子，搜索权重、因子判定阈值和看空规则。新增因子的对照搜索统一在“因子研究”页运行。")
    baseline = _search_baseline_controls()
    objective = _search_objective_controls(baseline.objective)
    available_start, available_end = _benchmark_date_bounds()
    default_cutoff = min(max(pd.Timestamp("2025-01-01").date(), available_start), available_end)
    training_end = st.date_input(
        "训练截止日",
        value=default_cutoff,
        min_value=available_start,
        max_value=available_end,
        help="所有候选只使用截止日及以前的数据排名；之后的数据仅评价样本外表现。",
        key="search_training_end",
    ).isoformat()
    st.caption(f"搜索期：首个可用信号至 {training_end}；样本外：{(pd.Timestamp(training_end) + pd.Timedelta(days=1)).date()} 至最新。")
    previous_baseline = baseline
    baseline = DashboardStrategyConfig(
        name=baseline.name,
        weights=baseline.weights,
        thresholds=baseline.thresholds,
        positions=baseline.positions,
        objective=objective,
        factor_windows=baseline.factor_windows,
        backtest_start=None,
        backtest_end=None,
        benchmark_id=GOV_10Y,
        signal_frequency=baseline.signal_frequency,
        research_provenance=baseline.research_provenance,
        source_config_path=baseline.source_config_path,
    )
    baseline = record_manual_changes(previous_baseline, baseline, ROOT)
    _persist_search_draft(baseline, st.session_state.get("search_draft_source_path", ""))

    with st.container(key="combined_search_band"):
        combined_copy, combined_action = st.columns([3, 1])
        with combined_copy:
            st.markdown("### 权重 → 阈值联合搜索")
            st.caption("先在训练期选出最优因子权重，再自动将该权重作为阈值搜索基线；最终只归档联合搜索策略，两个阶段的研究报告分别保留。")
            combined_weight_version_label = st.selectbox(
                "联合搜索的权重版本", ["V2", "V1"], index=0, key="combined_weight_search_version",
                help="联合搜索先运行所选版本的权重搜索，再将其最优权重交给阈值搜索。",
            )
            if combined_weight_version_label == "V2":
                st.caption("V2允许任一因子权重为0，单个模块总权重最高50，总权重保持100；这是当前主搜索规则。")
            else:
                st.caption("V1保留旧版模块最低权重约束，仅用于与历史搜索规则对照。")
        with combined_action:
            run_combined = st.button(
                "运行联合搜索",
                type="primary",
                use_container_width=True,
                key="run_combined_search_web",
            )
    combined_weight_version = "v2" if str(combined_weight_version_label).upper() == "V2" else "v1"
    if run_combined:
        with st.status("正在执行联合搜索...", expanded=True) as status:
            metrics = run_combined_search(
                ROOT,
                base_config=baseline,
                objective_config=objective,
                training_end=training_end,
                progress=st.write,
                weight_search_version=combined_weight_version,
            )
            status.update(label="联合搜索完成", state="complete", expanded=False)
        experiment_dir = Path(str(metrics["experiment_dir"]))
        st.toast(
            f"联合搜索完成：累计资本利得 {metrics['capital_gain_total_bp']:.2f} BP，逐笔胜率 {metrics['capital_gain_trade_win_rate']:.2%}"
        )
        _switch_to_history_experiment(experiment_dir)

    left, middle, right = st.columns(3)
    with left:
        st.markdown("#### 因子权重搜索")
        st.caption("遍历因子权重；使用上方设置的定性阈值和仓位制度。基线权重只用于对照，不限制候选空间。")
        if st.button("运行权重搜索", type="primary", use_container_width=True, key="run_weight_search_web"):
            with st.spinner("正在搜索权重并生成报告..."):
                metrics = run_dashboard_weight_search_v1(ROOT, objective_config=objective, base_config=baseline, training_end=training_end)
            st.success(f"权重搜索完成：累计资本利得 {metrics['capital_gain_total_bp']:.2f} BP，逐笔胜率 {metrics['capital_gain_trade_win_rate']:.2%}")
            experiment_dir = Path(str(metrics.get("experiment_dir", "")))
            if experiment_dir.exists():
                _render_favorite_button(experiment_dir, key=f"favorite_weight_v1_{experiment_dir.name}")
        _render_latest_search_result_link("权重向量化搜索", "查看最近权重搜索最优策略", baseline.signal_frequency)
    with middle:
        st.markdown("#### 权重搜索 v2")
        st.caption("取消供给、需求、估值、非银模块最低权重；每个模块可为0、上限50，总权重仍为100。")
        if st.button("运行权重搜索 v2", type="primary", use_container_width=True, key="run_weight_search_v2_web"):
            with st.spinner("正在搜索无模块最低权重的候选组合..."):
                metrics = run_dashboard_weight_search_v2(ROOT, objective_config=objective, base_config=baseline, training_end=training_end)
            st.success(f"v2完成：{metrics['candidate_count']:,} 组，累计资本利得 {metrics['capital_gain_total_bp']:.2f} BP")
            experiment_dir = Path(str(metrics.get("experiment_dir", "")))
            if experiment_dir.exists():
                _render_favorite_button(experiment_dir, key=f"favorite_weight_v2_{experiment_dir.name}")
        _render_latest_search_result_link("权重向量化搜索v2", "查看最近权重搜索v2最优策略", baseline.signal_frequency)
    with right:
        st.markdown("#### 定性阈值与看空规则搜索")
        st.caption("使用上方设置的因子权重和仓位作为基线，搜索定性阈值、看空总分和确认条件。")
        if st.button("运行阈值搜索", type="primary", use_container_width=True, key="run_threshold_search_web"):
            with st.spinner("正在搜索阈值并生成报告..."):
                metrics = run_threshold_research(ROOT, objective_config=objective, base_config_override=baseline, training_end=training_end)
            st.success(f"阈值搜索完成：累计资本利得 {metrics['capital_gain_total_bp']:.2f} BP，逐笔胜率 {metrics['capital_gain_trade_win_rate']:.2%}")
            experiment_dir = Path(str(metrics.get("experiment_dir", "")))
            if experiment_dir.exists():
                _render_favorite_button(experiment_dir, key=f"favorite_threshold_{experiment_dir.name}")
        _render_latest_search_result_link("阈值向量化搜索", "查看最近阈值搜索最优策略", baseline.signal_frequency)

    report_choice = st.segmented_control("查看搜索报告", ["权重搜索 v1", "权重搜索 v2", "阈值搜索"], default="权重搜索 v1")
    if report_choice == "权重搜索 v1":
        report_path = ROOT / "backtest_outputs" / ("dashboard_weight_search_v1_daily" if baseline.signal_frequency == "daily" else "dashboard_weight_search_v1") / "权重搜索报告.html"
        selected_source = "权重向量化搜索"
    elif report_choice == "权重搜索 v2":
        report_path = ROOT / "backtest_outputs" / ("dashboard_weight_search_v2_daily" if baseline.signal_frequency == "daily" else "dashboard_weight_search_v2") / "权重搜索报告.html"
        selected_source = "权重向量化搜索v2"
    else:
        report_path = ROOT / "backtest_outputs" / ("阈值调参实验_v1_日频" if baseline.signal_frequency == "daily" else "阈值调参实验_v1") / "阈值调参报告.html"
        selected_source = "阈值向量化搜索"
    _render_latest_search_dashboard(selected_source, report_choice, baseline.signal_frequency)
    if report_path.exists():
        with st.expander("完整搜索报告", expanded=False):
            st.caption(f"报告更新时间：{datetime.fromtimestamp(report_path.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
            report_html = report_path.read_text(encoding="utf-8")
            components.html(report_html, height=_report_embed_height(report_html), scrolling=False)
    else:
        st.info("尚未生成该搜索报告，请先运行对应搜索。")


def _render_rolling_task_queue() -> None:
    """Render a self-refreshing, non-blocking queue snapshot in the sidebar."""

    def readonly_run_display_name(experiment_dir: Path, strategy_name: object) -> str:
        """Display an already registered archive ID without mutating the registry."""
        name = _normalise_display_name(strategy_name)
        archive_name = Path(experiment_dir).name
        if "__" not in archive_name:
            return name
        try:
            manifest = json.loads((Path(experiment_dir) / "run_manifest.json").read_text(encoding="utf-8"))
            id_prefix = archive_id_prefix(str(manifest.get("运行来源", "")))
        except (OSError, json.JSONDecodeError):
            id_prefix = "B"
        archive_id = existing_short_archive_id(ROOT, archive_id_category(id_prefix), archive_name)
        if not archive_id:
            return name
        display_name = _experiment_display_names().get(archive_name, name)
        return f"[{archive_id}] {_normalise_display_name(display_name)}"

    def strategy_display(task: dict[str, object]) -> str:
        if task.get("task_kind") == "factor_rolling":
            return _normalise_display_name(task.get("task_name", "因子完整滚动"))
        raw_config = task.get("config")
        base = raw_config.get("base_config") if isinstance(raw_config, dict) else None
        if not isinstance(base, dict):
            return "未识别策略"
        name = _normalise_display_name(base.get("name", "未命名策略"))
        provenance = base.get("研究溯源")
        if not isinstance(provenance, dict):
            return name
        references: list[object] = []
        steps = provenance.get("steps")
        if isinstance(steps, list):
            references.extend(step.get("output") for step in reversed(steps) if isinstance(step, dict))
        references.append(provenance.get("origin"))
        for reference in references:
            experiment_dir = _provenance_reference_experiment(reference)
            if experiment_dir is not None:
                return readonly_run_display_name(experiment_dir, name)
        return name

    @st.fragment(run_every=3)
    def render_queue_snapshot() -> None:
        """Poll only this small sidebar fragment so form state is untouched."""
        tasks = list_rolling_tasks(ROOT)
        by_id = {str(task.get("task_id")): task for task in tasks if task.get("task_id")}

        def has_completed_retry(task: dict[str, object]) -> bool:
            failed_id = str(task.get("task_id") or "")
            if not failed_id:
                return False
            for candidate in tasks:
                if candidate.get("status") != "完成":
                    continue
                parent_id = str(candidate.get("retry_of") or "")
                seen: set[str] = set()
                while parent_id and parent_id not in seen:
                    if parent_id == failed_id:
                        return True
                    seen.add(parent_id)
                    parent = by_id.get(parent_id)
                    parent_id = str(parent.get("retry_of") or "") if parent else ""
            return False

        def render_task(number: int, task: dict[str, object], state: str) -> None:
            task_id = str(task.get("task_id") or "未编号")
            safe_task_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", task_id)
            with st.container(key=f"rolling_queue_task_{state}_{safe_task_id}"):
                if state != "active":
                    st.markdown(f"**{number:02d} · {escape(task_id)}**")
                    st.caption(strategy_display(task))
                    return
                details_col, cancel_col = st.columns([0.87, 0.13], gap="small")
                with details_col:
                    st.markdown(f"**{number:02d} · {escape(task_id)}**")
                    st.caption(strategy_display(task))
                    completed = pd.to_numeric(pd.Series([task.get("completed_periods")]), errors="coerce").iloc[0]
                    total = pd.to_numeric(pd.Series([task.get("total_periods")]), errors="coerce").iloc[0]
                    if pd.notna(completed) and pd.notna(total) and float(total) > 0:
                        ratio = min(1.0, max(0.0, float(completed) / float(total)))
                        st.caption(f"完成度 {ratio:.0%} · {int(completed)} / {int(total)} 期")
                    st.caption(str(task.get("stage") or "正在运行"))
                with cancel_col:
                    cancel_requested = bool(task.get("cancel_requested"))
                    if st.button(
                        "×",
                        key=f"cancel_rolling_queue_{task_id}",
                        help="正在终止" if cancel_requested else "终止任务",
                        disabled=cancel_requested,
                    ):
                        try:
                            cancel_rolling_task(ROOT, task_id)
                        except (ValueError, TimeoutError) as exc:
                            st.error(str(exc))
                        else:
                            st.toast("已请求终止，当前期搜索结束后会停止。")
                            st.rerun(scope="fragment")

        active = [
            (number, task) for number, task in enumerate(tasks, 1)
            if task.get("status") in {"启动中", "运行中"}
        ]
        waiting = [(number, task) for number, task in enumerate(tasks, 1) if task.get("status") == "等待中"]
        failed = [
            (number, task) for number, task in reversed(list(enumerate(tasks, 1)))
            if task.get("status") == "失败" and not has_completed_retry(task)
        ]

        st.markdown("<div class='rolling-queue-heading is-active'>正在运行</div>", unsafe_allow_html=True)
        for number, task in active:
            render_task(number, task, "active")

        st.divider()
        st.markdown("<div class='rolling-queue-heading is-waiting'>排队</div>", unsafe_allow_html=True)
        for number, task in waiting:
            render_task(number, task, "waiting")

        st.divider()
        st.markdown("<div class='rolling-queue-heading is-failed'>失败任务</div>", unsafe_allow_html=True)
        if failed:
            history_height = min(320, max(88, len(failed) * 72))
            with st.container(height=history_height, border=False):
                for number, task in failed:
                    task_id = str(task.get("task_id") or "")
                    safe_task_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", task_id)
                    with st.container(key=f"rolling_queue_task_failed_{safe_task_id}"):
                        task_col, retry_col, remove_col = st.columns([0.76, 0.12, 0.12])
                        with task_col:
                            st.markdown(f"**{number:02d} · {escape(task_id or '未编号')}**")
                            st.caption(strategy_display(task))
                        with retry_col:
                            if st.button("↻", key=f"retry_rolling_queue_{task_id}", help="再次执行"):
                                try:
                                    retry_task = retry_rolling_task(ROOT, task_id)
                                    start_rolling_queue_worker(ROOT)
                                except (ValueError, TimeoutError) as exc:
                                    st.error(str(exc))
                                else:
                                    st.toast(f"已重新加入队列：{retry_task['task_id']}")
                                    st.rerun(scope="fragment")
                        with remove_col:
                            if st.button("×", key=f"remove_rolling_queue_{task_id}", help="删除失败任务"):
                                try:
                                    remove_failed_rolling_task(ROOT, task_id)
                                except (ValueError, TimeoutError) as exc:
                                    st.error(str(exc))
                                else:
                                    st.rerun(scope="fragment")

    with st.sidebar:
        st.markdown("#### 滚动任务队列")
        render_queue_snapshot()


def _render_rolling_research_page() -> None:
    st.markdown(
        "<section class='hero-shell'><div><p class='eyebrow'>样本外验证 / 参数稳定性</p><h1 class='hero-title'>滚动定参</h1><p class='hero-copy'>每期仅在当期训练窗口搜索；采用的参数只在下一段样本外执行。搜索排序完全沿用现有目标函数。</p></div></section>",
        unsafe_allow_html=True,
    )
    _render_rolling_task_queue()
    baseline = _search_baseline_controls()
    objective = _search_objective_controls(baseline.objective)
    previous_baseline = baseline
    baseline = DashboardStrategyConfig(
        name=baseline.name,
        weights=baseline.weights,
        thresholds=baseline.thresholds,
        positions=baseline.positions,
        objective=objective,
        factor_windows=baseline.factor_windows,
        backtest_start=None,
        backtest_end=None,
        benchmark_id=GOV_10Y,
        signal_frequency=baseline.signal_frequency,
        research_provenance=baseline.research_provenance,
        source_config_path=baseline.source_config_path,
    )
    baseline = record_manual_changes(previous_baseline, baseline, ROOT)
    available_start, available_end = _benchmark_date_bounds()
    default_end = min(max(pd.Timestamp("2025-06-30").date(), available_start), available_end)
    st.markdown(
        "<div class='section-head'><h2>训练与定参规则</h2><p>每次只用训练窗口内的数据搜索；选出的参数只在紧随其后的样本外区间执行。</p></div>",
        unsafe_allow_html=True,
    )
    controls_a, controls_b, controls_c = st.columns(3)
    with controls_a:
        st.markdown("#### 启动与搜索范围")
        automatic_start = st.checkbox("首个满足训练长度的窗口自动启动", value=True, key="rolling_auto_start")
        first_training_end = st.date_input(
            "首次训练截止日", value=default_end, min_value=available_start, max_value=available_end,
            key="rolling_first_training_end",
            disabled=automatic_start,
            help="首段样本外从该日后的首个交易日开始。",
        )
        mode_label = st.segmented_control(
            "搜索流程", ["联合搜索", "权重搜索", "阈值搜索"], default="联合搜索", key="rolling_search_mode"
        )
        weight_version_label = st.selectbox(
            "权重搜索版本", ["V2", "V1"], index=0, key="rolling_weight_search_version",
            help="联合搜索和权重搜索使用的权重候选版本；阈值搜索不受此选项影响。",
        )
        if weight_version_label == "V2":
            st.caption("V2允许因子权重为0，单个模块总权重最高50；这是当前主搜索规则。")
        else:
            st.caption("V1保留旧版模块最低权重约束，用于历史规则对照。")
    with controls_b:
        st.markdown("#### 训练窗口")
        window_label = st.segmented_control(
            "窗口方式", ["自首个数据日扩展", "固定滚动窗口"], default="自首个数据日扩展", key="rolling_window_mode"
        )
        if window_label == "自首个数据日扩展":
            st.caption("每次训练从首个共同可用数据日开始，到本期训练截止日为止；样本会随时间增长。")
        else:
            st.caption("每次训练只使用训练截止日前最近的固定月数；更早数据不会进入本期搜索。")
        rolling_months = st.number_input(
            "固定窗口长度（月）", min_value=3, max_value=120, value=24, step=3,
            disabled=window_label != "固定滚动窗口", key="rolling_window_months",
        )
        minimum_months = st.number_input(
            "扩展窗口最低训练长度（月）", min_value=3, max_value=120, value=24, step=3,
            disabled=window_label == "固定滚动窗口", key="rolling_minimum_months",
        )
    with controls_c:
        st.markdown("#### 定参与参数切换")
        interval_count = st.number_input("定参频率", min_value=1, max_value=36, value=3, step=1, key="rolling_interval_count")
        interval_label = st.selectbox("定参频率单位", ["月", "周", "自然日"], index=0, key="rolling_interval_unit")
        improvement = st.number_input(
            "参数切换门槛", min_value=0.0, value=0.0, step=0.1, key="rolling_improvement",
            help="新参数与上一期参数在当前同一训练窗口比较；默认 0 即当前窗口更优就切换。",
        )
        st.caption("每到一个定参点重新搜索；只有超过切换门槛才替换上一期参数，避免微小噪声导致频繁换参。")
    task_name = st.text_input("滚动任务名称（可选）", value="", key="rolling_task_name")
    st.caption("自动启动按行情与信号共同可用区间计算；指定日期不足训练长度则报错。首次定参与参数相同均不计为切换。")
    if st.button("加入滚动任务队列", type="primary", use_container_width=True, key="run_rolling_research"):
        mode = {"联合搜索": "combined", "权重搜索": "weight", "阈值搜索": "threshold"}[str(mode_label)]
        weight_search_version = "v2" if str(weight_version_label).upper() == "V2" else "v1"
        unit = {"月": "months", "周": "weeks", "自然日": "days"}[str(interval_label)]
        rolling_config = RollingResearchConfig(
            base_config=baseline,
            search_mode=mode,
            training_mode="expanding" if window_label == "自首个数据日扩展" else "rolling_months",
            first_training_end=None if automatic_start else first_training_end.isoformat(),
            recalibration_interval=int(interval_count),
            recalibration_unit=unit,
            rolling_window_months=int(rolling_months) if window_label == "固定滚动窗口" else None,
            weight_search_version=weight_search_version,
            minimum_training_months=int(rolling_months) if window_label == "固定滚动窗口" else int(minimum_months),
            min_objective_improvement=float(improvement),
            task_name=task_name.strip() or None,
        )
        task = enqueue_rolling_task(ROOT, rolling_config)
        worker_started = start_rolling_queue_worker(ROOT)
        if task.get("status") == "启动中" or worker_started:
            st.success(f"已开始执行：{task['task_id']}。")
        else:
            st.success(f"已加入队列：{task['task_id']}。前序任务完成后将自动执行。")
        st.rerun()

    output_value = st.session_state.get("rolling_research_output_dir")
    if not output_value:
        st.info("运行后将显示首段搜索期、逐期参数、同窗新旧参数比较，以及样本外累计资本利得与超额路径。")
        return
    output_dir = Path(str(output_value))
    period_path = output_dir / "逐期定参与样本外表现.csv"
    daily_path = output_dir / "滚动定参样本外拼接_日度.csv"
    rolling_manifest_path = output_dir / "滚动配置.json"
    if not period_path.exists() or not daily_path.exists():
        st.warning("最近一次滚动定参结果文件不存在，请重新运行。")
        return
    rolling_manifest: dict[str, object] = {}
    if rolling_manifest_path.exists():
        try:
            rolling_manifest = json.loads(rolling_manifest_path.read_text(encoding="utf-8"))
            rolling_id = str(rolling_manifest.get("运行ID", "")).strip()
            if rolling_id:
                st.caption(f"运行ID：{rolling_id} · R = 滚动定参，后三位为该类记录的永久序号。")
        except (OSError, json.JSONDecodeError):
            pass
    periods = pd.read_csv(period_path, encoding="utf-8-sig")
    daily = pd.read_csv(daily_path, encoding="utf-8-sig", parse_dates=["date"])
    search_daily_path = output_dir / "滚动定参搜索期_日度.csv"
    search_signals_path = output_dir / "滚动定参搜索期_信号.csv"
    search_period_metrics: dict[str, object] = {}
    if search_daily_path.exists():
        search_daily = pd.read_csv(search_daily_path, encoding="utf-8-sig", parse_dates=["date"])
        search_signals = (
            pd.read_csv(search_signals_path, encoding="utf-8-sig", parse_dates=["signal_date"])
            if search_signals_path.exists()
            else pd.DataFrame()
        )
        st.caption(
            f"首段搜索期：{search_daily['date'].min():%Y-%m-%d} 至 {search_daily['date'].max():%Y-%m-%d}；"
            "该段用于选出第一组参数，下面与样本外使用同一套首页指标。"
        )
        if not search_signals.empty:
            search_periods = evaluate_periods(
                search_daily,
                search_signals,
                str(search_daily["date"].max().date()),
                CONDITIONAL_BENCHMARK_NAME,
            )
            search_result = search_periods.get("全区间", {})
            search_period_metrics = search_result.get("strategy_metrics", {})
    elif rolling_manifest_path.exists():
        try:
            rolling_manifest = json.loads(rolling_manifest_path.read_text(encoding="utf-8"))
            st.caption(
                f"首段搜索期：{rolling_manifest.get('首次搜索期起始日', '未登记')} 至 "
                f"{rolling_manifest.get('首次搜索期结束日', '未登记')}；"
                "当前旧结果未保存搜索期日度明细，无法补算该段指标。"
            )
        except (OSError, json.JSONDecodeError):
            pass
    total = float(pd.to_numeric(daily["strategy_capital_cum_bp"], errors="coerce").iloc[-1])
    excess = float(pd.to_numeric(daily["capital_excess_cum_bp"], errors="coerce").iloc[-1])
    switches = int(periods["是否切换新参数"].fillna(False).astype(bool).sum()) if "是否切换新参数" in periods else 0
    if search_period_metrics:
        st.markdown("#### 搜索期与样本外表现")
        oos_metrics = performance_metrics(daily, return_col="strategy_return", nav_col="strategy_nav")
        oos_metrics.update(capital_gain_trade_metrics(daily, "strategy_capital_bp", position_col="仓位"))
        search_row = {
            "区间": "搜索期",
            "起始日期": str(search_daily["date"].min().date()),
            "结束日期": str(search_daily["date"].max().date()),
            "累计资本利得_BP": search_period_metrics.get("capital_gain_total_bp"),
            "资本利得超额_BP": search_period_metrics.get("capital_gain_excess_bp"),
            "已平仓胜率": search_period_metrics.get("capital_gain_trade_win_rate"),
            "已平仓交易数": search_period_metrics.get("capital_gain_closed_trade_count"),
            "最大回撤_BP": search_period_metrics.get("capital_gain_max_drawdown_bp"),
        }
        comparison_position_col = "comparison_position" if "comparison_position" in daily.columns else None
        oos_benchmark_metrics = capital_gain_trade_metrics(
            daily, "benchmark_capital_bp", position_col=comparison_position_col
        )
        oos_row = {
            "区间": "样本外",
            "起始日期": str(daily["date"].min().date()),
            "结束日期": str(daily["date"].max().date()),
            "累计资本利得_BP": oos_metrics.get("capital_gain_total_bp"),
            "资本利得超额_BP": float(oos_metrics.get("capital_gain_total_bp", 0.0)) - float(
                oos_benchmark_metrics.get("capital_gain_total_bp", 0.0)
            ),
            "已平仓胜率": oos_metrics.get("capital_gain_trade_win_rate"),
            "已平仓交易数": oos_metrics.get("capital_gain_closed_trade_count"),
            "最大回撤_BP": oos_metrics.get("capital_gain_max_drawdown_bp"),
        }
        _render_theme_table(
            pd.DataFrame([search_row, oos_row]),
            numeric_columns={
                "累计资本利得_BP", "资本利得超额_BP", "已平仓胜率",
                "已平仓交易数", "最大回撤_BP",
            },
            scrollable=True,
        )
    st.markdown(
        "<section class='metric-grid'>"
        + _metric_cell("样本外累计资本利得", _bp_text(total), "逐期定参后的样本外区间", total)
        + _metric_cell("样本外超额资本利得", _bp_text(excess), "同仓位国债 / 空仓现金", excess)
        + _metric_cell("定参期数", str(len(periods)), f"实际切换 {switches} 次", float(switches))
        + "</section>",
        unsafe_allow_html=True,
    )
    st.markdown("#### 样本外累计资本利得路径")
    _render_interactive_chart(_capital_bp_chart(daily), key="rolling_stitched_capital")
    st.markdown("#### 逐期定参与样本外表现")
    display_columns = [
        "期数", "权重搜索版本", "首次定参", "训练起始日", "训练截止日", "样本外起始日", "样本外结束日", "候选目标函数",
        "上一期参数在当前窗口目标函数", "目标函数改善", "最小改善阈值", "候选参数是否变化", "是否切换新参数",
        "样本外资本利得_BP", "样本外超额_BP", "样本外交易胜率", "采用参数",
    ]
    _render_theme_table(periods[[c for c in display_columns if c in periods.columns]], numeric_columns=set(periods.columns), scrollable=True)
    report_path = output_dir / "滚动定参报告.html"
    if report_path.exists():
        st.caption(f"结果目录：{output_dir}")
    archived = None
    try:
        archived = rolling_manifest.get("历史实验目录")
    except (NameError, AttributeError):
        pass
    if archived:
        history_column, favorite_column = st.columns(2)
        history_column.link_button("查看历史实验结果", f"/history?experiment={quote(Path(str(archived)).name)}", use_container_width=True)
        with favorite_column:
            _render_favorite_button(Path(str(archived)), key=f"favorite_rolling_{Path(str(archived)).name}")


def _render_latest_search_dashboard(source: str, label: str, signal_frequency: str) -> None:
    experiments = list_experiments(ROOT)
    matches = experiments.loc[experiments["运行来源"] == source] if not experiments.empty else pd.DataFrame()
    if not matches.empty:
        frequency_label = "日频" if signal_frequency == "daily" else "周频"
        matches = matches.loc[matches["信号频率"] == frequency_label]
    if matches.empty:
        st.info(f"尚无{label}结果。运行后这里会展示训练期、样本外、近期和全区间对比。")
        return
    experiment_dir = Path(str(matches.iloc[0]["实验目录"]))
    try:
        daily, signals, _, benchmark_metrics, _ = load_experiment_result(experiment_dir)
    except Exception as exc:
        st.warning(f"最近搜索结果加载失败：{exc}")
        return
    st.markdown(f"<div class='section-head'><h2>{escape(label)}结果</h2><p>训练集选优 · 样本外仅诊断</p></div>", unsafe_allow_html=True)
    daily, signals, strategy_metrics, benchmark_metrics, periods, selected = _render_period_scope(
        daily,
        signals,
        benchmark_metrics,
        _research_training_end(experiment_dir),
        key=f"search_period_{source}",
        default="样本外",
    )
    decay = generalization_summary(periods)
    if decay:
        st.markdown(
            "<section class='metric-grid'>"
            + _metric_cell("年化资本利得衰减", _bp_text(decay.get("年化资本利得衰减_BP")), "样本外年化 - 搜索期年化", float(decay.get("年化资本利得衰减_BP") or 0.0))
            + _metric_cell("年化超额衰减", _bp_text(decay.get("年化超额衰减_BP")), "样本外超额 - 搜索期超额", float(decay.get("年化超额衰减_BP") or 0.0))
            + _metric_cell("样本外保留率", _pct(decay.get("样本外年化保留率")), "相对搜索期年化资本利得", float(decay.get("样本外年化保留率") or 0.0))
            + _metric_cell("胜率变化", _pct(decay.get("样本外胜率变化")), "样本外胜率 - 搜索期胜率", float(decay.get("样本外胜率变化") or 0.0))
            + "</section>",
            unsafe_allow_html=True,
        )
    chart_col, table_col = st.columns([1.7, 1])
    with chart_col:
        st.markdown(f"#### {selected}资本利得路径")
        _render_interactive_chart(_capital_bp_chart(daily), key=f"search_capital_{source}_{selected}")
    with table_col:
        st.markdown("#### Top候选样本外稳定性")
        stability_path = experiment_dir / "top_stability.csv"
        if stability_path.exists():
            stability = pd.read_csv(stability_path, encoding="utf-8-sig")
            display = stability.head(10).rename(columns={
                "training_rank": "训练排名",
                "training_capital_gain_total_bp": "训练资本BP",
                "oos_capital_gain_total_bp": "样本外资本BP",
                "oos_capital_gain_excess_bp": "样本外超额BP",
            })
            _render_theme_table(display[[column for column in ["训练排名", "训练资本BP", "样本外资本BP", "样本外超额BP"] if column in display]], numeric_columns=set(display.columns), scrollable=True)
        else:
            st.caption("该历史搜索运行于分区间功能上线前，暂无Top候选稳定性文件。")


def _render_latest_search_result_link(source: str, label: str, signal_frequency: str) -> None:
    experiments = list_experiments(ROOT)
    if experiments.empty:
        return
    matches = experiments.loc[experiments["运行来源"] == source]
    frequency_label = "日频" if signal_frequency == "daily" else "周频"
    matches = matches.loc[matches["信号频率"] == frequency_label]
    if matches.empty:
        st.caption("尚无可查看的搜索最优策略归档。")
        return
    latest = matches.iloc[0]
    experiment_id = quote(Path(str(latest["实验目录"])).name)
    display_name = _run_display_name(Path(str(latest["实验目录"])), latest["策略名称"])
    st.markdown(
        f'<a class="search-result-link" href="/history?experiment={experiment_id}" target="_self"><strong>{escape(label)}</strong><span>{escape(display_name)}</span></a>',
        unsafe_allow_html=True,
    )


def _search_baseline_controls() -> DashboardStrategyConfig:
    persisted_source, persisted_config = _load_search_draft()
    default_path = Path(persisted_source) if persisted_source else ROOT / CONFIG_DIR / "策略02_最佳权重_中性0.5.json"
    selected_path = _strategy_config_picker(
        "搜索基线配置",
        "search_baseline_config",
        default=default_path,
        help_text="权重搜索继承其阈值和仓位；阈值搜索继承其权重和仓位。",
    )
    base = load_strategy_config(selected_path)
    selected_token = str(selected_path.resolve())
    st.session_state["search_draft_source_path"] = selected_token
    if st.session_state.get("search_draft_loaded_from") != selected_token:
        restored = persisted_config if persisted_source == selected_token and persisted_config is not None else base
        _seed_search_draft(restored)
        st.session_state["search_draft_loaded_from"] = selected_token
        st.rerun()
    key_prefix = "search_draft"
    st.caption(
        "可选项包括基础配置、保留基线、搜索最优配置和历史实验保存的配置。"
        "搜索区间仍由页面训练截止日控制，选择历史实验只继承其策略参数。"
    )

    st.caption(f"搜索统一使用条件基准：{CONDITIONAL_BENCHMARK_NAME}。")
    frequency_label = st.segmented_control(
        "搜索信号频率",
        ["周频", "日频"],
        default="日频" if base.signal_frequency == "daily" else "周频",
        key=f"{key_prefix}_signal_frequency",
        help="搜索候选与样本外评价使用同一信号频率。",
    )
    signal_frequency = "daily" if frequency_label == "日频" else "weekly"

    with st.expander("因子权重", expanded=True):
        columns = st.columns(3)
        weight_values: dict[str, float] = {}
        for index, (key, label) in enumerate(WEIGHT_LABELS.items()):
            if key == "fly_penalty":
                columns[index % 3].number_input(
                    label, value=0.0, disabled=True, key=f"{key_prefix}_weight_{key}_disabled"
                )
                weight_values[key] = 0.0
                continue
            min_value = -50.0 if key == "fly_penalty" else 0.0
            max_value = 0.0 if key == "fly_penalty" else 50.0
            weight_values[key] = columns[index % 3].number_input(
                label, min_value=min_value, max_value=max_value, value=float(getattr(base.weights, key)), step=5.0,
                key=f"{key_prefix}_weight_{key}",
            )

    with st.expander("定性阈值", expanded=False):
        columns = st.columns(3)
        threshold_specs = [
            ("supply_low", "供给低分位", 0.0, 50.0, 5.0), ("supply_high", "供给高分位", 50.0, 100.0, 5.0),
            ("demand_low", "需求低分位", 0.0, 50.0, 5.0), ("demand_high", "需求高分位", 50.0, 100.0, 5.0),
            ("spread_low", "地方债利差低分位", 0.0, 50.0, 5.0), ("spread_high", "地方债利差高分位", 50.0, 100.0, 5.0),
            ("ncd_low", "NCD利差低分位", 0.0, 50.0, 5.0), ("ncd_high", "NCD利差高分位", 50.0, 100.0, 5.0),
            ("spread_change_bp", "利差5日变化阈值（BP）" if signal_frequency == "daily" else "利差周变化阈值（BP）", 0.5, 10.0, 0.5),
        ]
        threshold_values = {
            key: columns[index % 3].number_input(
                label, min_value=minimum, max_value=maximum, value=float(getattr(base.thresholds, key)), step=step,
                key=f"{key_prefix}_threshold_{key}",
            )
            for index, (key, label, minimum, maximum, step) in enumerate(threshold_specs)
        }

    with st.expander("仓位与执行规则", expanded=True):
        columns = st.columns(3)
        bullish_threshold = columns[0].number_input("看多分数阈值", 50.0, 100.0, float(base.positions.bullish_threshold), 5.0, key=f"{key_prefix}_bullish_threshold")
        bearish_threshold = columns[1].number_input("看空分数阈值", 0.0, 50.0, float(base.positions.bearish_threshold), 5.0, key=f"{key_prefix}_bearish_threshold")
        bullish_position = columns[0].number_input("看多仓位", -1.0, 1.5, float(base.positions.bullish_position), 0.1, key=f"{key_prefix}_bullish_position")
        neutral_position = columns[1].number_input("中性仓位", -1.0, 1.5, float(base.positions.neutral_position), 0.1, key=f"{key_prefix}_neutral_position")
        bearish_position = columns[2].number_input("看空仓位", -1.5, 1.0, float(base.positions.bearish_position), 0.1, key=f"{key_prefix}_bearish_position")
        min_modules = columns[0].selectbox("看空所需核心利空模块数", [0, 1, 2, 3, 4], index=int(base.positions.bearish_min_core_factors), key=f"{key_prefix}_min_modules")
        require_supply = columns[1].toggle("看空必须包含供给或需求利空", value=bool(base.positions.bearish_require_supply_or_demand), key=f"{key_prefix}_require_supply")
        confirmations = columns[2].selectbox("看空连续确认天数" if signal_frequency == "daily" else "看空连续确认周数", [1, 2, 3], index=int(base.positions.bearish_confirmation_periods) - 1, key=f"{key_prefix}_confirmations")
        take_profit_enabled = columns[0].toggle("启用止盈", value=float(base.positions.take_profit_bp) > 0.0, key=f"{key_prefix}_take_profit_enabled")
        stop_loss_enabled = columns[1].toggle("启用止损", value=float(base.positions.stop_loss_bp) > 0.0, key=f"{key_prefix}_stop_loss_enabled")
        take_profit_bp = columns[0].number_input("止盈阈值（BP）", 0.5, 100.0, max(float(base.positions.take_profit_bp), 3.0), 0.5, disabled=not take_profit_enabled, key=f"{key_prefix}_take_profit_bp") if take_profit_enabled else 0.0
        stop_loss_bp = columns[1].number_input("止损阈值（BP）", 0.5, 100.0, max(float(base.positions.stop_loss_bp), 3.0), 0.5, disabled=not stop_loss_enabled, key=f"{key_prefix}_stop_loss_bp") if stop_loss_enabled else 0.0
        position_values = {
            "bullish_threshold": bullish_threshold, "bearish_threshold": bearish_threshold,
            "bullish_position": bullish_position, "neutral_position": neutral_position, "bearish_position": bearish_position,
            "bearish_min_core_factors": min_modules, "bearish_require_supply_or_demand": int(require_supply),
            "bearish_confirmation_periods": confirmations,
            "take_profit_bp": take_profit_bp,
            "stop_loss_bp": stop_loss_bp,
        }
    source_label = _strategy_config_source_label(selected_path)
    config = DashboardStrategyConfig(
        name=f"{source_label}·{base.name}", weights=DashboardWeights(**weight_values),
        thresholds=DashboardThresholds(**threshold_values), positions=DashboardPositionPolicy(**position_values),
        objective=base.objective, factor_windows=base.factor_windows, backtest_start=None, backtest_end=None, benchmark_id=GOV_10Y,
        signal_frequency=signal_frequency,
    )
    return record_manual_changes(base, config, ROOT)


def _seed_search_draft(base: DashboardStrategyConfig) -> None:
    st.session_state["search_draft_signal_frequency"] = "日频" if base.signal_frequency == "daily" else "周频"
    for key, value in base.weights.as_dict().items():
        st.session_state[f"search_draft_weight_{key}"] = float(value)
    for key, value in base.thresholds.as_dict().items():
        st.session_state[f"search_draft_threshold_{key}"] = float(value)
    position_key_map = {
        "bullish_threshold": "bullish_threshold",
        "bearish_threshold": "bearish_threshold",
        "bullish_position": "bullish_position",
        "neutral_position": "neutral_position",
        "bearish_position": "bearish_position",
        "bearish_min_core_factors": "min_modules",
        "bearish_require_supply_or_demand": "require_supply",
        "bearish_confirmation_periods": "confirmations",
        "take_profit_bp": "take_profit_bp",
        "stop_loss_bp": "stop_loss_bp",
    }
    for field, widget_name in position_key_map.items():
        value = getattr(base.positions, field)
        if field == "bearish_require_supply_or_demand":
            value = bool(value)
        st.session_state[f"search_draft_{widget_name}"] = value
    st.session_state["search_draft_take_profit_enabled"] = float(base.positions.take_profit_bp) > 0.0
    st.session_state["search_draft_stop_loss_enabled"] = float(base.positions.stop_loss_bp) > 0.0
    for key, value in base.objective.as_dict().items():
        st.session_state[f"search_{key}"] = float(value)


def _persist_search_draft(config: DashboardStrategyConfig, source_path: object) -> None:
    path = ROOT / SEARCH_DRAFT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source_path": str(source_path), "config": config.as_dict()}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != serialized:
        path.write_text(serialized, encoding="utf-8")


def _load_search_draft() -> tuple[str, DashboardStrategyConfig | None]:
    path = ROOT / SEARCH_DRAFT_FILE
    if not path.exists():
        return "", None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("source_path", "")), strategy_config_from_dict(payload["config"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "", None


def _search_objective_controls(defaults: ObjectiveConfig | None = None) -> ObjectiveConfig:
    defaults = defaults or ObjectiveConfig()
    st.markdown("#### 搜索目标函数")
    st.caption("下列参数只影响两个批量搜索的候选排序，不影响单次回测。正权重奖励指标，惩罚项通常设为正数后乘以负向风险指标。")
    st.caption("收益导向更重视累计资本利得；胜率导向更重视已平仓交易中盈利交易的比例；综合导向同时考虑收益、胜率与资本利得回撤。当前页面保留完整系数，便于自定义三者取舍。")
    capital_col, risk_col, traditional_col = st.columns(3)
    with capital_col:
        capital_gain_bp_weight = st.number_input("累计收益率资本利得BP权重", value=float(defaults.capital_gain_bp_weight), step=0.1, key="search_capital_gain_bp_weight")
        capital_gain_excess_bp_weight = st.number_input("收益率资本利得超额BP权重", value=float(defaults.capital_gain_excess_bp_weight), step=0.05, key="search_capital_gain_excess_bp_weight")
        capital_trade_win_rate_weight = st.number_input("已平仓交易胜率权重", value=float(defaults.capital_trade_win_rate_weight), step=1.0, key="search_capital_trade_win_rate_weight")
        capital_gain_avg_win_bp_weight = st.number_input("平均每笔盈利BP权重", value=float(defaults.capital_gain_avg_win_bp_weight), step=0.1, key="search_capital_gain_avg_win_bp_weight", help="只对盈利交易求平均，奖励有效盈利的幅度；与胜率配合，避免候选只产生大量零点几个BP的微小盈利。")
    with risk_col:
        capital_gain_drawdown_bp_penalty = st.number_input("资本利得回撤BP系数", value=float(defaults.capital_gain_drawdown_bp_penalty), step=0.1, key="search_capital_gain_drawdown_bp_penalty", help="回撤指标本身为负数；正系数会形成惩罚。")
        signal_win_rate_weight = st.number_input("调仓周期胜率权重（辅助）", value=float(defaults.signal_win_rate_weight), step=0.1, key="search_signal_win_rate_weight")
    with traditional_col:
        total_return_weight = st.number_input("累计收益权重", value=float(defaults.total_return_weight), step=0.1, key="search_total_return_weight")
        excess_return_weight = st.number_input("超额收益权重", value=float(defaults.excess_return_weight), step=0.05, key="search_excess_return_weight")
        sharpe_weight = st.number_input("夏普权重", value=float(defaults.sharpe_weight), step=0.05, key="search_sharpe_weight")
    st.code(
        f"目标分 = {capital_gain_bp_weight:g}×累计资本利得BP + {capital_gain_excess_bp_weight:g}×资本利得超额BP "
        f"+ {capital_trade_win_rate_weight:g}×已平仓交易胜率 + {capital_gain_avg_win_bp_weight:g}×平均每笔盈利BP "
        f"+ {capital_gain_drawdown_bp_penalty:g}×资本利得回撤BP "
        f"+ {total_return_weight:g}×累计收益 + {excess_return_weight:g}×超额收益 + {sharpe_weight:g}×夏普 "
        f"+ {signal_win_rate_weight:g}×调仓周期胜率",
        language=None,
    )
    return ObjectiveConfig(
        total_return_weight=total_return_weight,
        excess_return_weight=excess_return_weight,
        sharpe_weight=sharpe_weight,
        max_drawdown_penalty=0.0,
        signal_win_rate_weight=signal_win_rate_weight,
        capital_gain_bp_weight=capital_gain_bp_weight,
        capital_gain_excess_bp_weight=capital_gain_excess_bp_weight,
        capital_trade_win_rate_weight=capital_trade_win_rate_weight,
        capital_gain_avg_win_bp_weight=capital_gain_avg_win_bp_weight,
        capital_gain_drawdown_bp_penalty=capital_gain_drawdown_bp_penalty,
    )


def _report_embed_height(report_html: str) -> int:
    row_count = report_html.count("<tr")
    chart_count = report_html.count("echarts.init")
    return min(12000, max(2200, 1250 + row_count * 34 + chart_count * 520))


def _render_nav_tab(daily: pd.DataFrame, strategy_metrics: dict[str, object], benchmark_metrics: dict[str, object]) -> None:
    strategy_return = float(strategy_metrics["total_return"])
    benchmark_return = float(benchmark_metrics["total_return"])
    st.markdown(
        f"""
        <section class="metric-grid">
            {_metric_cell('策略累计收益', _pct(strategy_return), '资本利得 + carry', strategy_return)}
            {_metric_cell('基准累计收益', _pct(benchmark_return), '100%长期持有', benchmark_return)}
            {_metric_cell('累计超额收益', _pct(strategy_return - benchmark_return), '策略相对基准', strategy_return - benchmark_return)}
            {_metric_cell('最大回撤', _pct(strategy_metrics['max_drawdown']), f"夏普 {strategy_metrics['sharpe']:.2f}", float(strategy_metrics['max_drawdown']))}
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 净值与仓位")
    st.caption("策略、基准和超额净值共用左轴；仓位使用右轴。拖动底部时间轴查看局部区间。")
    _render_interactive_chart(_nav_chart(daily), key="nav_and_position")
    st.markdown("#### 看板总分与执行仓位")
    _render_interactive_chart(_score_position_chart(daily), key="score_and_position")
    metrics = pd.DataFrame(
        [
            {"指标": "累计收益", "策略": _pct(strategy_metrics["total_return"]), "基准": _pct(benchmark_metrics["total_return"])},
            {"指标": "年化收益", "策略": _pct(strategy_metrics["annual_return"]), "基准": _pct(benchmark_metrics["annual_return"])},
            {"指标": "超额年化收益", "策略": _pct(strategy_metrics["excess_annual_return"]), "基准": _pct(benchmark_metrics["excess_annual_return"])},
            {"指标": "年化波动", "策略": _pct(strategy_metrics["annual_volatility"]), "基准": _pct(benchmark_metrics["annual_volatility"])},
            {"指标": "夏普（无风险利率 1.4%）", "策略": f"{strategy_metrics['sharpe']:.3f}", "基准": f"{benchmark_metrics['sharpe']:.3f}"},
            {"指标": "最大回撤", "策略": _pct(strategy_metrics["max_drawdown"]), "基准": _pct(benchmark_metrics["max_drawdown"])},
            {"指标": "调仓周期胜率", "策略": _pct(strategy_metrics["signal_period_win_rate"]), "基准": _pct(benchmark_metrics["signal_period_win_rate"])},
        ]
    )
    _render_theme_table(metrics, numeric_columns={"策略", "基准"})


def _render_attribution_tab(daily: pd.DataFrame) -> None:
    st.markdown("#### 资本利得与 Carry 辅助归因")
    st.caption("资本利得BP是策略捕获的收益率方向变动：-仓位 × YTM变化BP，不乘久期。久期折算价格收益与carry只用于解释传统净值。")
    _render_interactive_chart(_attribution_chart(daily), key="return_attribution")
    attribution = daily[
        [
            "date",
            "strategy_carry_cum",
            "strategy_capital_cum",
            "benchmark_carry_cum",
            "benchmark_capital_cum",
            "carry_excess_cum",
            "capital_excess_cum",
        ]
    ].copy()
    labels = {
        "strategy_carry_cum": "策略累计 Carry",
        "strategy_capital_cum": "策略累计久期价格收益",
        "benchmark_carry_cum": "基准累计 Carry",
        "benchmark_capital_cum": "基准累计久期价格收益",
        "carry_excess_cum": "Carry 超额",
        "capital_excess_cum": "久期价格收益超额",
    }
    latest = attribution.iloc[-1].drop(labels=["date"]).rename(index=labels).map(_pct).rename("累计贡献").reset_index()
    latest.columns = ["归因项目", "累计贡献"]
    _render_theme_table(latest, numeric_columns={"累计贡献"})


def _render_diagnostics_tab(diagnostics: pd.DataFrame, signal_frequency: str = "weekly") -> None:
    frequency_name = "日度" if signal_frequency == "daily" else "周度"
    st.markdown(f"#### {frequency_name}错判诊断")
    st.caption(f"这里按{frequency_name}信号周期做判断归因，不等同于开平仓交易笔数；逐笔交易请查看“交易表现”页。")
    summary = _capital_diagnostic_summary(diagnostics)
    _render_interactive_chart(_capital_diagnostics_bar(summary), key="period_diagnostics", allow_zoom=False)
    display_summary = summary.copy()
    for col in ["累计策略资本利得_BP", "累计基准资本利得_BP", "累计资本利得超额_BP", "平均每周期资本利得_BP"]:
        display_summary[col] = pd.to_numeric(display_summary[col], errors="coerce").map(lambda value: f"{value:.2f} BP")
    _render_theme_table(
        display_summary,
        numeric_columns={"周期数", "盈利周期数", "累计策略资本利得_BP", "累计基准资本利得_BP", "累计资本利得超额_BP", "平均每周期资本利得_BP"},
        qualitative_column="资本利得判断类型",
    )

    st.markdown("#### 周期明细")
    display = diagnostics.copy()
    for col in ["周期策略收益", "周期基准收益", "周期超额收益", "周期carry贡献", "周期资本利得贡献"]:
        display[col] = pd.to_numeric(display[col], errors="coerce").map(_pct)
    for col in ["周期策略资本利得_BP", "周期基准资本利得_BP", "周期资本利得超额_BP"]:
        display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda value: f"{value:.2f} BP")
    _render_theme_table(
        display,
        numeric_columns={"仓位", "总分", "周期策略资本利得_BP", "周期基准资本利得_BP", "周期资本利得超额_BP", "周期策略收益", "周期基准收益", "周期超额收益", "周期carry贡献", "周期资本利得贡献"},
        qualitative_column="资本利得判断类型",
        scrollable=True,
        wide=True,
    )


def _capital_diagnostic_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["周期策略资本利得_BP", "周期基准资本利得_BP", "周期资本利得超额_BP"]
    frame = diagnostics.copy()
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    summary = (
        frame.groupby("资本利得判断类型", dropna=False)
        .agg(
            周期数=("资本利得判断类型", "size"),
            盈利周期数=("资本利得交易是否盈利", lambda values: int((values == "是").sum())),
            累计策略资本利得_BP=("周期策略资本利得_BP", "sum"),
            累计基准资本利得_BP=("周期基准资本利得_BP", "sum"),
            累计资本利得超额_BP=("周期资本利得超额_BP", "sum"),
            平均每周期资本利得_BP=("周期策略资本利得_BP", "mean"),
        )
        .reset_index()
    )
    summary["_sort_order"] = summary["资本利得判断类型"].map({name: index for index, name in enumerate(DIAGNOSTIC_TYPE_ORDER)})
    summary = summary.sort_values(["_sort_order", "资本利得判断类型"], na_position="last").drop(columns="_sort_order")
    return summary


def _factor_threshold_display(factor_name: str, thresholds: dict[str, object] | None) -> str:
    """Describe the actual decision boundaries shown by each signal row."""
    thresholds = thresholds or {}

    def value(key: str, fallback: float) -> float:
        return float(thresholds.get(key, fallback) or fallback)

    supply_low, supply_high = value("supply_low", 25.0), value("supply_high", 75.0)
    demand_low, demand_high = value("demand_low", 25.0), value("demand_high", 75.0)
    spread_low, spread_high = value("spread_low", 20.0), value("spread_high", 80.0)
    ncd_low, ncd_high = value("ncd_low", 25.0), value("ncd_high", 75.0)
    change = value("spread_change_bp", 2.0)
    name = factor_name.replace(" ", "")
    if "供给×银行承接" in name or "供给×非银承接" in name:
        return "交互项：两项基础乘数均高时利多；任一项低时不加分"
    if name == "供给_发飞惩罚" or "供给|发飞惩罚" in name or "供给/发飞惩罚" in name:
        return "数据不足，固定为 0"
    if name in {"供给_发行量", "供给_发行占比", "供给_10Y以上发行"} or name.startswith(("供给|", "供给/")):
        if "银行承接" in name or "非银承接" in name:
            return "利多：两项乘数均高；利空：任一项低"
        return f"利多 ≤ {supply_low:.0f}%；利空 ≥ {supply_high:.0f}%"
    if name in {"银行需求", "需求_银行总净买入"} or "银行总净买入" in name:
        return f"利多 ≥ {demand_high:.0f}%；利空 ≤ {demand_low:.0f}%"
    if name in {"利差_地方债国债", "估值_地方债-国债利差", "估值|地方债-国债利差"}:
        return f"利多 ≥ {spread_high:.0f}%；利空 ≤ {spread_low:.0f}%"
    if name in {"利差_地方债NCD", "估值_地方债-NCD利差", "估值|地方债-NCD利差"}:
        return f"利多 ≥ {ncd_high:.0f}%；利空 ≤ {ncd_low:.0f}%"
    if name in {"利差_周度变化", "估值_地方债-国债利差变化", "估值|地方债-国债利差变化"} or "利差变化" in name:
        return f"利多 ≤ -{change:.1f} BP；利空 ≥ +{change:.1f} BP"
    if name in {"非银情绪", "情绪_非银情绪", "情绪_非银情绪变化", "情绪|非银情绪", "情绪|非银情绪变化"}:
        return "利多 > 0；利空 < 0"
    if "发行量变化" in name:
        return f"利多 ≤ {supply_low:.0f}%；利空 ≥ {supply_high:.0f}%"
    if "银行总净买入变化" in name:
        return f"利多 ≥ {demand_high:.0f}%；利空 ≤ {demand_low:.0f}%"
    return "未单独设定阈值"


def _signal_source_row(signal_frequency: str, signal_date: object) -> pd.Series:
    """Return the current signal-grid row for display-only backfills.

    Early factor-research archives preserved scores and observed-at fields but
    omitted several raw inputs and execution dates.  Those values are present
    in the canonical signal grid; use it only to fill a missing display field,
    never to overwrite an archived score or performance result.
    """
    date = pd.to_datetime(signal_date, errors="coerce")
    if pd.isna(date):
        return pd.Series(dtype=object)
    try:
        frame = pd.read_csv(ROOT / signal_file_for_frequency(signal_frequency), encoding="utf-8-sig")
    except (OSError, ValueError, UnicodeDecodeError):
        return pd.Series(dtype=object)
    if "信号日期" not in frame:
        return pd.Series(dtype=object)
    dates = pd.to_datetime(frame["信号日期"], errors="coerce")
    matched = frame.loc[dates == date.normalize()]
    return matched.iloc[-1] if not matched.empty else pd.Series(dtype=object)


def _render_latest_signal(
    signals: pd.DataFrame,
    factor_weights: dict[str, object] | None = None,
    factor_thresholds: dict[str, object] | None = None,
    signal_frequency: str = "weekly",
) -> None:
    latest = signals.iloc[-1]
    source_latest = _signal_source_row(signal_frequency, latest.get("signal_date"))

    def value_from(*names: str) -> object:
        for name in names:
            value = latest.get(name)
            if value is not None and not pd.isna(value):
                return value
        return None

    def value_from_source(*names: str) -> object:
        value = value_from(*names)
        if value is not None:
            return value
        for name in names:
            source_value = source_latest.get(name)
            if source_value is not None and not pd.isna(source_value):
                return source_value
        return None

    market_observed = [
        value_from_source("银行需求取数日期", "bank_observed_at"),
        value_from_source("利差取数日期", "valuation_observed_at"),
        value_from_source("非银情绪取数日期", "sentiment_observed_at"),
    ]
    valid_market_observed = [value for value in market_observed if value is not None and not pd.isna(value)]
    supply_observed = value_from_source("供给信息截至日期", "供给取数日期", "supply_observed_at")
    generated_at = value_from_source("信号产生日期", "signal_generated_at")
    applicable_from = value_from_source("信号适用开始日期")
    applicable_to = value_from_source("信号适用结束日期")
    core = pd.DataFrame(
        [
            {"项目": "信号日期", "数值": _latest_signal_date(latest)},
            {"项目": "信号产生日期", "数值": _format_date(generated_at)},
            {"项目": "信号适用日期", "数值": f"{_format_date(applicable_from)} 至 {_format_date(applicable_to)}"},
            {"项目": "供给信息截至", "数值": _format_date(supply_observed)},
            {"项目": "市场数据截至", "数值": _format_date(max(valid_market_observed, key=pd.Timestamp) if valid_market_observed else None)},
            {"项目": "定性结论", "数值": latest.get("结论", "")},
            {"项目": "总分", "数值": f"{float(latest.get('总分', 0)):.1f}"},
            {"项目": "目标仓位", "数值": f"{float(latest.get('仓位', 0)):.1f}"},
        ]
    )
    st.markdown("#### 最新执行信号")
    _render_theme_table(core, numeric_columns={"数值"})
    st.markdown("#### 因子判断与得分")
    observed_detail = " · ".join(
        (
            f"供给截至 {_format_date(supply_observed)}",
            f"银行截至 {_format_date(market_observed[0])}",
            f"估值截至 {_format_date(market_observed[1])}",
            f"情绪截至 {_format_date(market_observed[2])}",
        )
    )
    st.markdown(
        f"<div class='signal-asof-banner'><strong>本期看板 · 信号日 {_latest_signal_date(latest)}</strong>"
        f"<span>{observed_detail}</span></div>",
        unsafe_allow_html=True,
    )
    input_labels = {
        "供给_发行量": ("亿元", "1年滚动分位"),
        "供给_发行占比": ("比例", "1年滚动分位"),
        "供给_10Y以上发行": ("亿元", "1年滚动分位"),
        "供给_发飞惩罚": ("", ""),
        "银行需求": ("", "1年滚动分位"),
        "利差_地方债国债": ("BP", "1年滚动分位"),
        "利差_周度变化": ("BP", "判断阈值"),
        "利差_地方债NCD": ("BP", "1年滚动分位"),
        "非银情绪": ("", ""),
        "供给|发行量变化": ("亿元", "1年滚动分位"),
        "需求|银行总净买入变化": ("", "1年滚动分位"),
        "情绪|非银情绪变化": ("", ""),
        "估值|地方债-NCD利差变化": ("BP", "判断阈值"),
        "估值|地方债-国债利差变化加速度": ("BP", "判断阈值"),
        "供需交互|供给×银行承接": ("", ""),
        "供需交互|供给×非银承接": ("", ""),
    }

    def display_input(value: object, unit: str, secondary: bool = False) -> str:
        if value is None or pd.isna(value):
            return "-"
        if isinstance(value, str):
            return value
        number = float(value)
        if secondary and unit == "1年滚动分位":
            return f"{number / 100.0:.1%}"
        if unit == "比例":
            return f"{number:.1%}"
        if unit:
            return f"{number:.2f} {unit}"
        return f"{number:.2f}"

    factor_rows = []
    weight_keys = {str(label): str(key) for key, label in FACTOR_LABELS.items()}
    weight_keys.update({str(label).replace("|", "/"): str(key) for key, label in FACTOR_LABELS.items()})
    weight_keys.update({
        "供给_发行量": "supply_amount",
        "供给_发行占比": "supply_ratio",
        "供给_10Y以上发行": "supply_long",
        "供给_发飞惩罚": "fly_penalty",
        "银行需求": "bank_demand",
        "利差_地方债国债": "spread_gov",
        "利差_周度变化": "spread_change",
        "利差_地方债NCD": "spread_ncd",
        "非银情绪": "nonbank_sentiment",
    })
    for score_column in [column for column in latest.index if str(column).endswith("_得分")]:
        factor_name = str(score_column)[: -len("_得分")]
        canonical_name = factor_name.replace("/", "|")
        factor_key = weight_keys.get(factor_name, weight_keys.get(canonical_name, ""))
        score = pd.to_numeric(latest[score_column], errors="coerce")
        weight = None if factor_weights is None else pd.to_numeric(factor_weights.get(factor_key, 0.0), errors="coerce")
        raw_unit, secondary_label = input_labels.get(canonical_name, input_labels.get(factor_name, ("", "")))
        raw_value = latest.get(f"{factor_name}_原始数据")
        secondary_value = latest.get(f"{factor_name}_二级数据")
        # Expanded-factor signals retain auditable raw inputs under canonical
        # columns and their multiplier beside the score. Resolve these aliases
        # so single-module factor results show the same score/weight detail.
        expanded_raw_columns = {
            "supply_amount": "supply_amount_raw", "supply_ratio": "supply_ratio_raw", "supply_long": "supply_long_raw",
            "bank_demand": "bank_demand_raw", "spread_gov": "spread_gov_raw",
            "nonbank_sentiment": "nonbank_sentiment_raw", "spread_ncd": "spread_ncd_raw",
            "spread_change": "spread_change_raw", "supply_amount_change": "supply_amount_change_raw",
            "bank_demand_change": "bank_demand_change_raw", "nonbank_sentiment_change": "nonbank_sentiment_change_raw",
            "spread_ncd_change": "spread_ncd_change_raw", "spread_change_acceleration": "spread_change_acceleration_raw",
        }
        if (raw_value is None or pd.isna(raw_value)) and factor_key:
            raw_value = latest.get(expanded_raw_columns.get(factor_key, ""))
        if (raw_value is None or pd.isna(raw_value)) and factor_key:
            source_raw_columns = {
                "supply_amount": "未来一周地方债发行量",
                "supply_ratio": "未来一周地方债发行量/（国债发行量+地方债发行量）",
                "supply_long": "地方债发行10年以上绝对发行量",
                "bank_demand": "银行过去一周净买入金额",
                "spread_gov": "10年好地区一般债-10年国债活跃券利差",
                "spread_change": "上述利差周度变化情况",
                "spread_ncd": "10年好地区一般债-1年国股行NCD利差",
                "nonbank_sentiment": "基煜纯债基金周度净申购情况",
            }
            raw_value = source_latest.get(source_raw_columns.get(factor_key, ""))
        if secondary_value is None or pd.isna(secondary_value):
            secondary_value = latest.get(f"{factor_name}_乘数")
        qualitative = latest.get(f"{factor_name}_定性", None)
        if qualitative is None or pd.isna(qualitative):
            multiplier = pd.to_numeric(latest.get(f"{factor_name}_乘数"), errors="coerce")
            qualitative = "利多" if pd.notna(multiplier) and float(multiplier) >= 0.75 else "利空" if pd.notna(multiplier) and float(multiplier) <= 0.25 else "中性"
        if secondary_value is None or pd.isna(secondary_value):
            secondary_display = "-"
        else:
            secondary_display = display_input(secondary_value, secondary_label, secondary=True)
        factor_rows.append(
            {
                "因子": FACTOR_LABELS.get(factor_key, canonical_name.replace("_", " | ")),
                "原始数据": display_input(raw_value, raw_unit),
                "二级数据": secondary_display,
                "得分 / 权重": "" if pd.isna(score) else (
                    f"<strong>{float(score):.1f}</strong> / {float(weight):.1f}" if weight is not None and not pd.isna(weight) else f"<strong>{float(score):.1f}</strong>"
                ),
                "利多 / 利空阈值": _factor_threshold_display(factor_name, factor_thresholds),
                "定性": qualitative,
                "_row_class": (
                    "factor-inactive" if weight is not None and (pd.isna(weight) or float(weight) == 0.0) else "factor-active"
                ),
            }
        )
    # Some older/rolling archives only persisted total scores.  Keep the
    # configured factor universe visible in that case instead of silently
    # dropping the weight view; the score is explicitly marked unavailable.
    if not factor_rows and factor_weights:
        ordered_weight_keys = [key for key in FACTOR_DISPLAY_COLUMNS if key in factor_weights]
        ordered_weight_keys.extend(key for key in factor_weights if key not in ordered_weight_keys)
        for factor_key in ordered_weight_keys:
            weight = pd.to_numeric(factor_weights.get(factor_key), errors="coerce")
            factor_rows.append(
                {
                    "因子": FACTOR_LABELS.get(factor_key, factor_key),
                    "原始数据": "-",
                    "二级数据": "-",
                    "得分 / 权重": f"未保存 / {float(weight):.1f}" if pd.notna(weight) else "未保存",
                    "利多 / 利空阈值": _factor_threshold_display(FACTOR_LABELS.get(factor_key, factor_key), factor_thresholds),
                    "定性": "未保存",
                    "_row_class": "factor-inactive" if pd.isna(weight) or float(weight) == 0.0 else "factor-active",
                }
            )
    if factor_rows:
        # Archived extended-factor result columns were historically written as
        # "original 8" followed by "new 7".  Keep the table in economic
        # groups instead: e.g. supply amount is immediately followed by its
        # change factor, without affecting the archived scores or weights.
        display_rank = {FACTOR_LABELS[key]: index for index, key in enumerate(FACTOR_DISPLAY_COLUMNS)}
        factor_rows.sort(key=lambda row: display_rank.get(str(row["因子"]), len(display_rank)))
        st.caption("得分以“实际得分 / 搜索权重”显示。权重为 0 的因子已弱化：保留其状态和得分供核对，但不参与总分。")
        _render_theme_table(
            pd.DataFrame(factor_rows),
            qualitative_column="定性",
            html_columns={"得分 / 权重"},
            row_class_column="_row_class",
        )
    else:
        st.caption("该结果仅保存了实际执行的总分、结论和仓位；因子权重请在“研究记录与参数”查看。")

    st.markdown("#### 历史信号")
    history_columns = [
        "signal_date",
        "总分",
        "结论",
        "仓位",
    ]
    history = signals[[column for column in history_columns if column in signals.columns]].copy()
    history = history.rename(
        columns={
            "signal_date": "信号日期",
            "总分": "总分",
            "结论": "结论",
            "仓位": "仓位",
        }
    )
    for column in ["信号日期"]:
        if column in history:
            history[column] = pd.to_datetime(history[column], errors="coerce").dt.strftime("%Y-%m-%d")
    if "总分" in history:
        history["总分"] = pd.to_numeric(history["总分"], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.1f}")
    if "仓位" in history:
        history["仓位"] = pd.to_numeric(history["仓位"], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.1f}")
    _render_theme_table(history.sort_values("信号日期", ascending=False).reset_index(drop=True), qualitative_column="结论")


def _render_config_snapshot(config: DashboardStrategyConfig) -> None:
    groups = {
        "因子权重": _display_factor_weights(config),
        "定性阈值": config.thresholds.as_dict(),
        "仓位规则": config.positions.as_dict(),
        "目标函数": config.objective.as_dict(),
    }
    rows = []
    for group, values in groups.items():
        for key, value in values.items():
            rows.append({"参数组": group, "字段": key, "数值": value})
    st.markdown("#### 本次回测参数")
    st.caption("左侧继续调整不会改变这里的快照；再次运行后才会更新结果。需要复现时，请保存为 JSON。")
    _render_theme_table(pd.DataFrame(rows), numeric_columns={"数值"})
    with st.expander("查看原始 JSON"):
        st.json(config.as_dict())


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_date(value: object) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "未保存"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return "未保存" if str(value).strip().lower() in {"", "none", "nan", "nat"} else str(value)


def _latest_signal_date(latest: pd.Series) -> str:
    for key in ("signal_date", "信号日期", "date", "日期"):
        if key in latest.index and pd.notna(latest[key]):
            return _format_date(latest[key])
    if isinstance(latest.name, (pd.Timestamp, datetime)):
        return _format_date(latest.name)
    return "未识别"


def _render_theme_table(
    frame: pd.DataFrame,
    numeric_columns: set[str] | None = None,
    qualitative_column: str | None = None,
    scrollable: bool = False,
    wide: bool = False,
    html_columns: set[str] | None = None,
    row_class_column: str | None = None,
) -> None:
    numeric_columns = numeric_columns or set()
    html_columns = html_columns or set()
    visible_columns = [column for column in frame.columns if column != row_class_column]
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in visible_columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in visible_columns:
            value = "" if pd.isna(row[column]) else str(row[column])
            classes = ["numeric"] if column in numeric_columns else []
            if column in html_columns:
                cell_value = value
            elif qualitative_column == column:
                tone = _semantic_tone(value)
                cell_value = f'<span class="qualitative {tone}">{escape(value)}</span>'
            else:
                cell_value = escape(value)
            cells.append(f'<td class="{" ".join(classes)}">{cell_value}</td>')
        row_class = "" if row_class_column is None or row_class_column not in row.index else escape(str(row[row_class_column]))
        body_rows.append(f'<tr class="{row_class}">{"".join(cells)}</tr>')
    wrapper_classes = "research-table-wrap" + (" scrollable" if scrollable else "")
    table_classes = "research-table" + (" wide" if wide else "")
    st.markdown(
        f'<div class="{wrapper_classes}"><table class="{table_classes}"><thead><tr>{headers}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _semantic_tone(value: str) -> str:
    if value in {"利多", "看多", *DIAGNOSTIC_GOOD_TYPES}:
        return "bullish"
    if value in {"利空", "看空", *DIAGNOSTIC_BAD_TYPES}:
        return "bearish"
    return ""


def _render_interactive_chart(figure: go.Figure, key: str, allow_zoom: bool = True) -> None:
    st.plotly_chart(
        figure,
        use_container_width=True,
        key=key,
        config={
            "displaylogo": False,
            "scrollZoom": allow_zoom,
            "responsive": True,
            "displayModeBar": allow_zoom,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def _apply_chart_theme(figure: go.Figure, title: str, height: int) -> go.Figure:
    figure.update_layout(
        title={"text": title, "font": {"size": 17, "color": INK}, "x": 0, "xanchor": "left"},
        height=height,
        margin={"l": 58, "r": 58, "t": 78, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(251,250,246,0.45)",
        font={"family": "Geist, Microsoft YaHei, sans-serif", "color": MUTED, "size": 12},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#fbfaf6", "bordercolor": GRID, "font": {"color": INK}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "xanchor": "left", "x": 0},
        modebar={"bgcolor": "rgba(243,242,237,.8)", "color": MUTED, "activecolor": GREEN},
        hoverdistance=50,
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor=GRID,
        tickfont={"color": MUTED},
        rangeslider={"visible": True, "thickness": 0.06, "bgcolor": "#e7e9e4", "bordercolor": GRID, "borderwidth": 1},
    )
    figure.update_yaxes(
        gridcolor=GRID,
        griddash="dot",
        zerolinecolor="#bfc7c1",
        linecolor=GRID,
        tickfont={"color": MUTED},
    )
    return figure


def _lock_date_extent(figure: go.Figure, dates: pd.Series) -> None:
    date_min = pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
    date_max = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
    figure.update_xaxes(
        range=[date_min, date_max],
        minallowed=date_min,
        maxallowed=date_max,
    )


def _nav_chart(daily: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    series = [
        ("策略净值", "strategy_nav", GREEN, 2.8, None),
        ("基准净值", "benchmark_nav_rebased", INK, 1.8, None),
        ("超额净值", "excess_nav", CORAL, 1.8, "dash"),
    ]
    for name, column, color, width, dash in series:
        figure.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily[column],
                name=name,
                mode="lines",
                line={"color": color, "width": width, "dash": dash} if dash else {"color": color, "width": width},
                hovertemplate=f"{name} %{{y:.4f}}<extra></extra>",
            ),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["仓位"],
            name="仓位",
            mode="lines",
            line={"color": GOLD, "width": 1.5, "dash": "dot", "shape": "hv"},
            hovertemplate="仓位 %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    _apply_chart_theme(figure, "策略 / 基准 / 超额净值与仓位", 520)
    _lock_date_extent(figure, daily["date"])
    figure.update_yaxes(title_text="净值", secondary_y=False)
    figure.update_yaxes(title_text="仓位", range=[-1.1, 1.1], showgrid=False, secondary_y=True)
    return figure


def _capital_bp_chart(daily: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    for name, column, color, dash in [
        ("策略累计资本利得", "strategy_capital_cum_bp", CORAL, None),
        ("条件基准累计资本利得", "benchmark_capital_cum_bp", GREEN, None),
        ("资本利得超额", "capital_excess_cum_bp", GOLD, "dash"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily[column],
                name=name,
                mode="lines",
                line={"color": color, "width": 2.3, **({"dash": dash} if dash else {})},
                hovertemplate=f"{name} %{{y:.2f}} BP<extra></extra>",
            ),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["仓位"],
            name="仓位",
            mode="lines",
            line={"color": INK, "width": 1.3, "dash": "dot", "shape": "hv"},
            hovertemplate="仓位 %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    _apply_chart_theme(figure, "累计资本利得与执行仓位", 500)
    _lock_date_extent(figure, daily["date"])
    figure.update_yaxes(title_text="累计资本利得 BP", ticksuffix=" BP", secondary_y=False)
    figure.update_yaxes(title_text="仓位", range=[-1.1, 1.1], showgrid=False, secondary_y=True)
    return figure


def _yield_trade_signal_chart(daily: pd.DataFrame) -> go.Figure:
    frame = daily.sort_values("date").copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if "asset_yield_pct" not in frame.columns:
        try:
            asset_curve = load_market_data(ROOT, GOV_10Y)[["date", "asset_yield_pct"]].copy()
            asset_curve["date"] = pd.to_datetime(asset_curve["date"], errors="coerce")
            frame = frame.merge(asset_curve, on="date", how="left", validate="many_to_one")
        except (OSError, ValueError, KeyError):
            frame["asset_yield_pct"] = pd.NA
    frame["asset_yield_pct"] = pd.to_numeric(frame["asset_yield_pct"], errors="coerce")
    frame = frame.dropna(subset=["date", "asset_yield_pct"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["asset_yield_pct"],
            name="10Y地方债YTM",
            mode="lines",
            line={"color": INK, "width": 2.2},
            hovertemplate="%{x|%Y-%m-%d}<br>YTM %{y:.4f}%<extra></extra>",
        )
    )
    if frame.empty:
        figure.add_annotation(
            text="该历史实验缺少可匹配的10Y地方债收益率数据",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED, "size": 14},
        )
    trades = capital_gain_trade_table(daily)
    if not trades.empty and not frame.empty:
        yield_by_date = frame.drop_duplicates("date", keep="last").set_index("date")["asset_yield_pct"]
        trades = trades.copy()
        capital_gain_bp = pd.to_numeric(trades["strategy_capital_bp"], errors="coerce")
        trades["entry_marker_date"] = _trade_entry_valuation_dates(daily, trades)
        trades["exit_marker_date"] = _trade_exit_marker_dates(daily, trades)
        trades["connection_end_date"] = trades["exit_marker_date"].where(
            trades["is_closed"], pd.to_datetime(trades["mark_date"], errors="coerce")
        )

        # A line is a valuation interval, not a literal order ticket.  Daily
        # signals are produced the prior day and applied on the next trading
        # day; the period P&L therefore starts at the prior close.
        for _, trade in trades.iterrows():
            entry_date = trade["entry_marker_date"]
            exit_date = trade["connection_end_date"]
            entry_yield = yield_by_date.get(entry_date)
            exit_yield = yield_by_date.get(exit_date)
            if pd.isna(entry_date) or pd.isna(exit_date) or pd.isna(entry_yield) or pd.isna(exit_yield):
                continue
            is_profit = float(trade["strategy_capital_bp"]) > 0.0
            figure.add_trace(
                go.Scatter(
                    x=[entry_date, exit_date],
                    y=[entry_yield, exit_yield],
                    mode="lines",
                    line={
                        "color": "rgba(187,101,79,0.92)" if is_profit else "rgba(23,107,91,0.92)",
                        "width": 1.6,
                    },
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        signal_specs = [
            ("多头计价起点", trades["direction"].eq("多头"), "entry_marker_date", CORAL, "triangle-up", "上一交易日收盘估值"),
            ("空头计价起点", trades["direction"].eq("空头"), "entry_marker_date", GREEN, "triangle-down", "上一交易日收盘估值"),
            ("多头计价终点（正资本利得）", trades["direction"].eq("多头") & capital_gain_bp.ge(0.0), "exit_marker_date", CORAL, "circle", "仓位生效日收盘估值"),
            ("多头计价终点（负资本利得）", trades["direction"].eq("多头") & capital_gain_bp.lt(0.0), "exit_marker_date", GREEN, "circle", "仓位生效日收盘估值"),
            ("空头计价终点", trades["direction"].eq("空头"), "exit_marker_date", GREEN, "circle", "仓位生效日收盘估值"),
        ]
        for name, mask, date_column, color, symbol, date_role in signal_specs:
            points = trades.loc[mask].copy()
            if points.empty:
                continue
            points["signal_date"] = pd.to_datetime(points[date_column], errors="coerce")
            points = points.dropna(subset=["signal_date"])
            points["yield_pct"] = points["signal_date"].map(yield_by_date)
            points = points.dropna(subset=["yield_pct"])
            points["effective_date"] = pd.to_datetime(points["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            points["mark_date_text"] = pd.to_datetime(points["mark_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            figure.add_trace(
                go.Scatter(
                    x=points["signal_date"],
                    y=points["yield_pct"],
                    name=name,
                    mode="markers",
                    marker={"color": color, "size": 10, "symbol": symbol, "line": {"color": PAPER, "width": 1}},
                    customdata=points[["trade_id", "direction", "strategy_capital_bp", "effective_date", "mark_date_text"]],
                    hovertemplate=(
                        f"{name}<br>%{{x|%Y-%m-%d}}<br>YTM %{{y:.4f}}%"
                        "<br>交易 #%{customdata[0]} · %{customdata[1]}"
                        f"<br>{date_role}"
                        "<br>仓位生效日 %{customdata[3]} · 计价终点 %{customdata[4]}"
                        "<br>该笔资本利得 %{customdata[2]:.2f} BP"
                        "<extra></extra>"
                    ),
                )
            )
    _apply_chart_theme(figure, "10Y地方债收益率与实际持仓计价区间", 520)
    # A unified hover snaps to the nearest date from every trace.  On a sparse
    # daily trade path that made an 8/19 entry appear while the pointer was on
    # 8/18.  Trade markers must be exact-date interactions.
    figure.update_layout(hovermode="closest")
    if not frame.empty:
        _lock_date_extent(figure, frame["date"])
    figure.update_yaxes(title_text="到期收益率（%）", ticksuffix="%")
    return figure


def _trade_exit_marker_dates(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    """Use the date on which the trade P&L is actually accounted for.

    The engine applies the day's yield change to that day's executed position.
    Moving a close marker to the next signal date (for example 8/18) visually
    claimed that the 8/15 -> 8/18 yield move produced the trade's +2.85 BP,
    although the archived trade table correctly records one holding day on
    8/15.  Keep chart markers and trade accounting on the same date.
    """
    if trades.empty:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(trades["mark_date"], errors="coerce").astype("datetime64[ns]")


def _trade_entry_valuation_dates(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    """Map each active trade to the prior close that begins its first P&L day."""
    timeline = pd.to_datetime(daily.sort_values("date")["date"], errors="coerce").reset_index(drop=True)
    positions = {date: index for index, date in timeline.items() if pd.notna(date)}
    marker_dates: list[object] = []
    for entry in pd.to_datetime(trades["entry_date"], errors="coerce"):
        index = positions.get(entry)
        marker_dates.append(timeline.iloc[index - 1] if index is not None and index > 0 else entry)
    return pd.Series(marker_dates, index=trades.index, dtype="datetime64[ns]")


def _capital_trade_chart(daily: pd.DataFrame) -> go.Figure:
    trades = capital_gain_trade_table(daily)
    colors = [CORAL if value > 0 else GREEN for value in trades["strategy_capital_bp"]]
    figure = go.Figure(
        go.Bar(
            x=trades["entry_date"],
            y=trades["strategy_capital_bp"],
            marker={"color": colors},
            customdata=trades[["direction", "status", "holding_days"]],
            hovertemplate="开仓 %{x|%Y-%m-%d}<br>资本利得 %{y:.2f} BP<br>%{customdata[0]} / %{customdata[1]}<br>持有 %{customdata[2]} 日<extra></extra>",
        )
    )
    _apply_chart_theme(figure, "开平仓交易资本利得：红色盈利，绿色亏损", 430)
    _lock_date_extent(figure, trades["entry_date"])
    figure.update_yaxes(title_text="单笔资本利得（BP）", ticksuffix=" BP")
    return figure


def _score_position_chart(daily: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["总分"],
            name="总分",
            mode="lines",
            line={"color": GREEN, "width": 2.6},
            hovertemplate="总分 %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["仓位"],
            name="仓位",
            mode="lines",
            line={"color": CORAL, "width": 1.6, "dash": "dot", "shape": "hv"},
            hovertemplate="仓位 %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    _apply_chart_theme(figure, "看板总分与执行仓位", 430)
    _lock_date_extent(figure, daily["date"])
    figure.update_yaxes(title_text="总分", range=[0, 100], secondary_y=False)
    figure.update_yaxes(title_text="仓位", range=[-1.1, 1.1], showgrid=False, secondary_y=True)
    return figure


def _attribution_chart(daily: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    series = [
        ("策略 Carry", "strategy_carry_cum", GREEN, 2.5, None),
        ("策略久期价格收益", "strategy_capital_cum", CORAL, 2.5, None),
        ("基准 Carry", "benchmark_carry_cum", GREEN_LIGHT, 1.7, None),
        ("基准久期价格收益", "benchmark_capital_cum", INK, 1.7, None),
        ("Carry 超额", "carry_excess_cum", GOLD, 1.6, "dash"),
        ("久期价格收益超额", "capital_excess_cum", "#7b8180", 1.6, "dash"),
    ]
    for name, column, color, width, dash in series:
        line = {"color": color, "width": width}
        if dash:
            line["dash"] = dash
        figure.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily[column] * 100.0,
                name=name,
                mode="lines",
                line=line,
                hovertemplate=f"{name} %{{y:.2f}}%<extra></extra>",
            )
        )
    _apply_chart_theme(figure, "传统净值归因：Carry 与久期价格收益", 500)
    _lock_date_extent(figure, daily["date"])
    figure.update_yaxes(title_text="累计贡献（%）", ticksuffix="%")
    return figure


def _capital_diagnostics_bar(summary: pd.DataFrame) -> go.Figure:
    categories = summary["资本利得判断类型"].astype(str).tolist()
    values = summary["累计策略资本利得_BP"].astype(float).tolist()
    bad_values = [value if category in DIAGNOSTIC_BAD_TYPES else None for category, value in zip(categories, values)]
    good_values = [value if category in DIAGNOSTIC_GOOD_TYPES else None for category, value in zip(categories, values)]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=categories,
            y=bad_values,
            name="错判 / 拖累",
            marker={"color": GREEN, "line": {"color": "#fbfaf6", "width": 1}},
            hovertemplate="%{x}<br>累计资本利得 %{y:.2f} BP<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=categories,
            y=good_values,
            name="有效 / 改善",
            marker={"color": CORAL, "line": {"color": "#fbfaf6", "width": 1}},
            hovertemplate="%{x}<br>累计资本利得 %{y:.2f} BP<extra></extra>",
        )
    )
    _apply_chart_theme(figure, "各类交易的累计资本利得", 370)
    figure.update_layout(barmode="group", margin={"l": 58, "r": 28, "t": 72, "b": 48})
    figure.update_xaxes(rangeslider={"visible": False}, fixedrange=True, categoryorder="array", categoryarray=DIAGNOSTIC_TYPE_ORDER)
    figure.update_yaxes(title_text="累计资本利得（BP）", ticksuffix=" BP", fixedrange=True)
    return figure


if __name__ == "__main__":
    main()
