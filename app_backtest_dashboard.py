from __future__ import annotations

from datetime import datetime
from html import escape
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from common.config import (
    CONFIG_DIR,
    DashboardStrategyConfig,
    ObjectiveConfig,
    load_strategy_config,
    save_strategy_config,
    strategy_config_from_dict,
)
from common.experiments import archive_dashboard_experiment, list_experiments, load_experiment_result
from common.reporting import _build_period_diagnostics
from common.runner import run_dashboard_config
from common.runner import run_dashboard_weight_search_v1
from common.threshold_research import run_threshold_research
from common.trade_metrics import capital_gain_trade_table
from strategies.dashboard_signal_v1 import DashboardThresholds, DashboardWeights
from strategies.position_policy import DashboardPositionPolicy


ROOT = Path(__file__).resolve().parent
RESULT_STATE_KEY = "dashboard_backtest_result"
SEARCH_DRAFT_FILE = Path("backtest_outputs") / "search_page_draft.json"

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
    "supply_amount": "供给 / 发行量",
    "supply_ratio": "供给 / 发行占比",
    "supply_long": "供给 / 10Y以上发行",
    "fly_penalty": "发行结果 / 发飞惩罚",
    "bank_demand": "需求 / 银行需求",
    "spread_gov": "估值 / 地方债-国债利差",
    "spread_change": "估值 / 利差周度变化",
    "spread_ncd": "估值 / 地方债-NCD利差",
    "nonbank_sentiment": "情绪 / 非银情绪",
}


def main() -> None:
    st.set_page_config(page_title="10Y地方债策略工作台", layout="wide")
    _inject_styles()

    if st.query_params.get("home") == "1":
        st.session_state.pop(RESULT_STATE_KEY, None)
        st.query_params.clear()
        st.rerun()

    view = str(st.query_params.get("view", "home"))
    if view.startswith("history_result__"):
        _render_historical_result_page(view.removeprefix("history_result__"))
        return
    if view == "history":
        experiment_id = str(st.query_params.get("experiment", "")).strip()
        if experiment_id:
            _render_historical_result_page(experiment_id)
        else:
            _render_header(None, None, view)
            _render_history_page()
        return
    if view == "search":
        _render_header(None, None, view)
        _render_search_page()
        return

    config_path = _select_config()
    base_config = load_strategy_config(config_path)
    config, run_clicked, save_clicked = _sidebar_config(base_config, key_prefix=f"main_{config_path.stem}")

    if save_clicked:
        _save_config(config)
    if run_clicked:
        experiment_dir = _run_backtest(config)
        if experiment_dir is not None:
            st.query_params["view"] = f"history_result__{experiment_dir.name}"
            st.rerun()

    result = st.session_state.get(RESULT_STATE_KEY)
    _render_header(config, result, "home")

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


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {
            --ink: #18201d;
            --muted: #66716c;
            --paper: #f3f2ed;
            --surface: #fbfaf6;
            --line: #d9ddd8;
            --green: #176b5b;
            --coral: #bb654f;
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
        [data-testid="stMainBlockContainer"] { max-width: 92rem; padding-top: 2.1rem; padding-bottom: 5rem; }
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

        .app-header-integration { height: 0; margin: 0; padding: 0; }
        .brand-lockup {
            position: fixed;
            top: .78rem;
            left: 4.25rem;
            z-index: 1000001;
            display: flex;
            align-items: center;
            gap: .72rem;
            min-width: 0;
            color: var(--ink) !important;
            text-decoration: none !important;
            border-bottom: 0 !important;
            transition: left .28s ease, color .2s ease, transform .2s ease;
        }
        body:has([data-testid="stSidebar"][aria-expanded="true"]) .brand-lockup {
            left: calc(300px + 1.25rem);
        }
        body:has([data-testid="stSidebar"][aria-expanded="false"]) .brand-lockup {
            left: 4.25rem;
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
            position: fixed;
            top: .62rem;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000001;
            display: flex;
            align-items: center;
            gap: 1.25rem;
            height: 2.25rem;
        }
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
        .aside-value { font-family: "IBM Plex Mono", monospace; font-size: clamp(1.8rem, 3vw, 3.1rem); font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1; }
        .aside-note { color: var(--muted); font-size: .82rem; margin-top: .75rem; line-height: 1.5; }

        .metric-grid { display: grid; grid-template-columns: repeat(12, 1fr); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); margin: 0 0 2.3rem; animation: rise .55s .12s ease both; }
        .metric-cell { grid-column: span 3; padding: 1.35rem 1.2rem 1.5rem 0; min-width: 0; }
        .metric-cell + .metric-cell { border-left: 1px solid var(--line); padding-left: 1.2rem; }
        .metric-label { color: var(--muted); font-size: .78rem; margin-bottom: .65rem; }
        .metric-value { font-family: "IBM Plex Mono", monospace; font-size: clamp(1.55rem, 2.3vw, 2.35rem); line-height: 1; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
        .metric-detail { color: var(--muted); font-size: .76rem; margin-top: .65rem; }
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
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.7rem; border-bottom: 1px solid var(--line); }
        [data-testid="stTabs"] button[role="tab"] { padding: .8rem 0; color: var(--muted); font-weight: 500; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: var(--ink); }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--green); }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }
        iframe { background: transparent; border-radius: 5px; }
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
        .archive-breadcrumb { display: flex; align-items: center; gap: .55rem; margin: 2.2rem 0 -1.2rem; color: var(--muted); font-size: .78rem; }
        .archive-breadcrumb a { color: var(--green) !important; text-decoration: none !important; }
        .archive-breadcrumb strong { color: var(--ink); font-weight: 600; }
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

        .featured-carousel { position: relative; min-height: 15.8rem; margin-bottom: 2.4rem; overflow: hidden; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
        .featured-slide { position: absolute; inset: 0; display: grid; grid-template-rows: auto 1fr; opacity: 0; pointer-events: none; animation: featuredCycle 12s infinite; }
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
        .featured-metric.highlight { background: rgba(187,101,79,.075); box-shadow: inset 0 3px 0 var(--coral); padding-left: 1.1rem; }
        .featured-metric-label { color: var(--muted); font-size: .76rem; margin-bottom: .65rem; }
        .featured-metric-value { color: var(--ink); font-family: "IBM Plex Mono", monospace; font-size: clamp(1.35rem, 2vw, 2.05rem); font-weight: 600; white-space: nowrap; }
        .featured-metric.highlight .featured-metric-value { color: var(--coral); }
        .featured-metric-benchmark { color: var(--muted); font-size: .72rem; margin-top: .65rem; }

        @keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes featuredCycle { 0%, 29% { opacity: 1; pointer-events: auto; transform: translateY(0); } 33%, 96% { opacity: 0; pointer-events: none; transform: translateY(8px); } 100% { opacity: 1; pointer-events: auto; transform: translateY(0); } }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
        @media (max-width: 900px) {
            [data-testid="stHeader"] { min-height: 5.2rem; }
            .top-navigation { top: 2.65rem; left: 4.25rem; right: auto; transform: none; gap: 1rem; }
            body:has([data-testid="stSidebar"][aria-expanded="true"]) .top-navigation { left: calc(300px + 1.25rem); }
            body:has([data-testid="stSidebar"][aria-expanded="false"]) .top-navigation { left: 4.25rem; }
            .hero-shell, .launch-grid { grid-template-columns: 1fr; }
            .hero-shell { gap: 2rem; padding-top: 2.6rem; }
            .hero-title { font-size: 2.55rem; }
            .hero-aside { border-left: 0; border-top: 1px solid var(--line); padding: 1.3rem 0 0; }
            .metric-cell { grid-column: span 6; }
            .metric-cell:nth-child(3) { border-left: 0; padding-left: 0; }
            .logic-flow { grid-template-columns: 1fr; }
            .featured-carousel { min-height: 25rem; }
            .featured-metrics { grid-template-columns: 1fr 1fr; }
            .featured-metric:nth-child(3) { border-left: 0; padding-left: 0; border-top: 1px solid var(--line); }
            .featured-metric:nth-child(4) { border-top: 1px solid var(--line); }
            .history-quick-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 560px) {
            [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
            .brand-sub { display: none; }
            .brand-lockup { left: 3.6rem; top: .82rem; }
            .brand-name { white-space: nowrap; }
            .hero-title { font-size: 2.55rem; }
            .metric-cell { grid-column: span 12; border-left: 0 !important; padding-left: 0 !important; border-bottom: 1px solid var(--line); }
            .metric-cell:last-child { border-bottom: 0; }
            .featured-carousel { min-height: 42rem; }
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
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        st.sidebar.error("configs 目录中没有可加载的 JSON 配置。")
        st.stop()
    default = config_dir / "策略02_最佳权重_中性0.5.json"
    if not default.exists():
        default = config_dir / "dashboard_signal_v1_default.json"
    default_index = configs.index(default) if default in configs else 0
    return st.sidebar.selectbox(
        "基线配置",
        configs,
        index=default_index,
        format_func=lambda path: path.stem,
        help="选择一个历史配置作为本次调参起点。",
    )


@st.cache_data(show_spinner=False)
def _benchmark_date_bounds() -> tuple[object, object]:
    path = ROOT / "benchmark_data" / "地方政府债到期收益率_10年_2024至最新.csv"
    dates = pd.to_datetime(pd.read_csv(path, encoding="utf-8-sig", usecols=["日期"])["日期"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("基准收益率文件没有可用日期")
    return dates.min().date(), dates.max().date()


@st.cache_data(show_spinner=False)
def _full_strategy_date_bounds() -> tuple[str, str]:
    benchmark_path = ROOT / "benchmark_data" / "地方政府债到期收益率_10年_2024至最新.csv"
    signal_path = ROOT / "data_processed" / "图表指标_周度宽表_统一日期.csv"
    benchmark_dates = pd.to_datetime(
        pd.read_csv(benchmark_path, encoding="utf-8-sig", usecols=["日期"])["日期"], errors="coerce"
    ).dropna().sort_values()
    signal_dates = pd.to_datetime(
        pd.read_csv(signal_path, encoding="utf-8-sig", usecols=["信号日期"])["信号日期"], errors="coerce"
    ).dropna()
    if benchmark_dates.empty or signal_dates.empty:
        raise ValueError("基准或周度信号文件没有可用日期")
    usable = benchmark_dates.loc[benchmark_dates >= signal_dates.min()]
    if usable.empty:
        raise ValueError("基准与周度信号没有重叠日期")
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
        st.caption("先命名，再保存；运行使用当前页面参数。")

    available_start, available_end = _benchmark_date_bounds()
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
            "spread_change_bp": st.slider("利差周变化阈值（BP）", 0.5, 10.0, float(base.thresholds.spread_change_bp), 0.5, key=widget_key("threshold_spread_change")),
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
                "看空连续确认周数",
                options=[1, 2, 3],
                value=int(base.positions.bearish_confirmation_periods),
                key=widget_key("position_confirmation"),
            ),
        }

    with st.sidebar.expander("搜索目标函数（只读）"):
        st.caption("单次回测不使用目标函数。这里仅展示该配置保存时的搜索口径；请在顶部“搜索研究”页调整。")
        objective_values = base.objective.as_dict()
        objective_display = pd.DataFrame(
            [
                {"项目": "累计资本利得BP", "权重": objective_values["capital_gain_bp_weight"]},
                {"项目": "资本利得超额BP", "权重": objective_values["capital_gain_excess_bp_weight"]},
                {"项目": "已平仓交易胜率", "权重": objective_values["capital_trade_win_rate_weight"]},
                {"项目": "资本利得回撤BP", "权重": objective_values["capital_gain_drawdown_bp_penalty"]},
                {"项目": "累计收益", "权重": objective_values["total_return_weight"]},
                {"项目": "超额收益", "权重": objective_values["excess_return_weight"]},
                {"项目": "夏普", "权重": objective_values["sharpe_weight"]},
                {"项目": "传统最大回撤", "权重": objective_values["max_drawdown_penalty"]},
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
    if effective_name == base.name and changed_groups:
        effective_name = _suggest_variant_name(base, weight_values, threshold_values, position_values, changed_groups, selected_window)
        st.sidebar.info(f"保存时自动更名：{effective_name}")

    config = DashboardStrategyConfig(
        name=effective_name,
        weights=DashboardWeights(**weight_values),
        thresholds=DashboardThresholds(**threshold_values),
        positions=DashboardPositionPolicy(**position_values),
        objective=ObjectiveConfig(**objective_values),
        backtest_start=selected_window["start_date"],
        backtest_end=selected_window["end_date"],
    )
    return config, run_clicked, save_clicked


def _suggest_variant_name(
    base: DashboardStrategyConfig,
    weights: dict[str, float],
    thresholds: dict[str, float],
    positions: dict[str, float | int],
    changed_groups: list[str],
    backtest: dict[str, str],
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
        if int(positions["bearish_min_core_factors"]) > 0:
            tags.append(f"确认{int(positions['bearish_min_core_factors'])}模块")
        if int(positions["bearish_require_supply_or_demand"]):
            tags.append("含供需")
        if int(positions["bearish_confirmation_periods"]) > 1:
            tags.append(f"连续{int(positions['bearish_confirmation_periods'])}周")
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
            daily, signals, strategy_metrics, benchmark_metrics = run_dashboard_config(ROOT, config)
            diagnostics = _build_period_diagnostics(daily, signals)
            experiment_dir = archive_dashboard_experiment(
                ROOT,
                config,
                daily,
                signals,
                strategy_metrics,
                benchmark_metrics,
                source="网页",
            )
        st.session_state[RESULT_STATE_KEY] = (
            daily,
            signals,
            strategy_metrics,
            benchmark_metrics,
            diagnostics,
            config.name,
            experiment_dir,
            config,
        )
        st.toast(f"实验已归档：{experiment_dir.name}")
        return experiment_dir
    except Exception as exc:
        st.error(f"回测运行失败：{exc}")
        return None


def _render_header(config: DashboardStrategyConfig | None, result: object, view: str) -> None:
    nav_links = "".join(
        f'<a class="top-nav-link {"active" if view == key else ""}" href="?view={key}" target="_self">{label}</a>'
        for key, label in [("home", "首页"), ("history", "历史实验"), ("search", "搜索研究")]
    )
    st.markdown(
        f"""
        <nav class="app-header-integration">
            <a class="brand-lockup" href="?view=home" target="_self" aria-label="返回首页">
                <span class="brand-mark"></span>
                <span class="brand-copy"><span class="brand-name">地方债策略实验室</span><span class="brand-sub">10Y Local Government Bond</span></span>
            </a>
            <div class="top-navigation">{nav_links}</div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def _render_launch_state(config: DashboardStrategyConfig) -> None:
    st.markdown(
        f"""
        <section class="hero-shell">
            <div>
                <p class="eyebrow">10Y 地方债方向与仓位研究</p>
                <h1 class="hero-title">把看板判断，转成可复现的持仓路径。</h1>
                <p class="hero-copy">以周度看板信号决定交易仓位，主要评价10Y地方债久期交易产生的资本利得BP、逐笔胜率和亏损控制；carry与传统收益率作为辅助。</p>
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
    st.markdown(
        f"""
        <section class="launch-grid">
            <article class="launch-panel">
                <h3>策略计算路径</h3>
                <div class="logic-flow">
                    <div class="logic-step"><strong>看板因子</strong><span>供给、银行需求、估值利差与非银情绪按周更新定性结论。</span></div>
                    <div class="logic-step"><strong>分数与仓位</strong><span>因子权重合成总分，再映射到看多、中性和看空三档仓位。</span></div>
                    <div class="logic-step"><strong>交易与诊断</strong><span>非零仓位开仓、归零平仓，反向时先平后开；统计逐笔资本利得BP、胜率与盈亏比。</span></div>
                </div>
            </article>
            <article class="launch-panel">
                <h3>本次参数快照</h3>
                <div class="config-line"><span>看多触发</span><strong>总分 ≥ {config.positions.bullish_threshold:.0f}</strong></div>
                <div class="config-line"><span>看空触发</span><strong>总分 &lt; {config.positions.bearish_threshold:.0f}</strong></div>
                <div class="config-line"><span>仓位档位</span><strong>{config.positions.bullish_position:.1f} / {config.positions.neutral_position:.1f} / {config.positions.bearish_position:.1f}</strong></div>
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
    capital_win_rate = strategy_metrics.get("capital_gain_trade_win_rate")
    average_trade_bp = strategy_metrics.get("capital_gain_avg_trade_bp")
    capital_drawdown_bp = strategy_metrics.get("capital_gain_max_drawdown_bp")
    win_rate_text = "暂无已平仓" if capital_win_rate is None else _pct(capital_win_rate)
    average_trade_text = "暂无已平仓" if average_trade_bp is None else f"{float(average_trade_bp):.2f} BP"
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
                <p class="hero-copy">总分 {float(latest.get('总分', 0)):.1f}，目标仓位 {float(latest['仓位']):.1f}。本项目以资本利得BP和逐笔交易胜率为主评价口径。</p>
            </div>
            <aside class="hero-aside">
                <div class="aside-label">最新仓位</div>
                <div class="aside-value">{float(latest['仓位']):.1f}</div>
                <div class="aside-note">信号日期 {escape(signal_date)}<br>策略夏普 {sharpe:.2f}</div>
            </aside>
        </section>
        <section class="metric-grid">
            {_metric_cell('累计资本利得', f'{capital_bp:.2f} BP', f'满仓基准 {benchmark_capital_bp:.2f} BP', capital_bp)}
            {_metric_cell('已平仓交易胜率', win_rate_text, f"盈利 {strategy_metrics['capital_gain_winning_trades']:.0f} / 已平仓 {strategy_metrics['capital_gain_closed_trade_count']:.0f} 笔", 0.0 if capital_win_rate is None else float(capital_win_rate) - 0.5)}
            {_metric_cell('平均单笔资本利得', average_trade_text, '盈亏比暂无' if payoff is None else f"盈亏比 {float(payoff):.2f}", 0.0 if average_trade_bp is None else float(average_trade_bp))}
            {_metric_cell('资本利得最大回撤', capital_drawdown_text, f"满仓基准 {float(benchmark_metrics.get('capital_gain_max_drawdown_bp', 0.0)):.2f} BP", 0.0 if capital_drawdown_bp is None else float(capital_drawdown_bp))}
        </section>
        """,
        unsafe_allow_html=True,
    )


def _metric_cell(label: str, value: str, detail: str, direction: float) -> str:
    color_class = "positive" if direction > 0 else "negative" if direction < 0 else ""
    return f"<div class='metric-cell'><div class='metric-label'>{escape(label)}</div><div class='metric-value {color_class}'>{escape(value)}</div><div class='metric-detail'>{escape(detail)}</div></div>"


def _render_workspace(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
    diagnostics: pd.DataFrame,
    result_name: str,
    config: DashboardStrategyConfig,
    experiment_dir: Path,
) -> None:
    st.markdown(
        f"<div class='section-head'><h2>研究工作区</h2><p>结果参数：{escape(result_name)} · 图表支持框选与滚轮缩放</p></div>",
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["交易表现", "传统净值", "收益归因", "错判诊断", "最新信号", "参数快照"])
    with tabs[0]:
        _render_trading_tab(daily, strategy_metrics, benchmark_metrics)
    with tabs[1]:
        _render_nav_tab(daily, strategy_metrics, benchmark_metrics)
    with tabs[2]:
        _render_attribution_tab(daily)
    with tabs[3]:
        _render_diagnostics_tab(diagnostics)
    with tabs[4]:
        _render_latest_signal(signals)
    with tabs[5]:
        _render_config_snapshot(config)
        st.caption(f"本次实验目录：{experiment_dir}")


def _render_trading_tab(
    daily: pd.DataFrame,
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
) -> None:
    st.markdown("#### 资本利得交易曲线")
    st.caption("交易从非零仓位开始，归零或反向时结束；同方向加减仓仍属于同一笔。BP仅统计久期资本利得，不包含carry。")
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
        elif unit in {"笔"}:
            strategy_text = "" if strategy_value is None else f"{int(strategy_value)} {unit}"
            benchmark_text = "" if benchmark_value is None else f"{int(benchmark_value)} {unit}"
        elif unit == "x":
            strategy_text = "" if strategy_value is None else f"{float(strategy_value):.2f}"
            benchmark_text = "" if benchmark_value is None else f"{float(benchmark_value):.2f}"
        else:
            strategy_text = "" if strategy_value is None else f"{float(strategy_value):.2f} {unit}"
            benchmark_text = "" if benchmark_value is None else f"{float(benchmark_value):.2f} {unit}"
        rows.append({"交易指标": label, "策略": strategy_text, "满仓基准": benchmark_text})
    _render_theme_table(pd.DataFrame(rows), numeric_columns={"策略", "满仓基准"})
    trades = capital_gain_trade_table(daily)
    if not trades.empty:
        trade_display = trades.rename(
            columns={
                "trade_id": "交易编号", "entry_date": "开仓日期", "exit_date": "平仓日期", "mark_date": "估值日期",
                "direction": "方向", "entry_position": "开仓仓位", "average_abs_position": "平均绝对仓位",
                "holding_days": "持有交易日", "strategy_capital_bp": "策略资本利得_BP",
                "benchmark_capital_bp": "同期满仓基准资本利得_BP", "capital_excess_bp": "资本利得超额_BP", "status": "状态",
            }
        )
        trade_display = trade_display[["交易编号", "开仓日期", "平仓日期", "估值日期", "方向", "开仓仓位", "平均绝对仓位", "持有交易日", "策略资本利得_BP", "同期满仓基准资本利得_BP", "资本利得超额_BP", "状态"]]
        for column in ["开仓日期", "平仓日期", "估值日期"]:
            trade_display[column] = pd.to_datetime(trade_display[column]).dt.strftime("%Y-%m-%d").fillna("")
        for column in ["策略资本利得_BP", "同期满仓基准资本利得_BP", "资本利得超额_BP"]:
            trade_display[column] = pd.to_numeric(trade_display[column], errors="coerce").map(lambda value: f"{value:.2f} BP")
        st.markdown("#### 开平仓交易明细")
        _render_theme_table(trade_display, numeric_columns={"开仓仓位", "平均绝对仓位", "持有交易日", "策略资本利得_BP", "同期满仓基准资本利得_BP", "资本利得超额_BP"}, scrollable=True, wide=True)


def _render_history_page() -> None:
    st.markdown(
        "<section class='hero-shell'><div><p class='eyebrow'>归档结果 / 不重新计算</p><h1 class='hero-title'>历史实验</h1><p class='hero-copy'>每条记录保留当时的配置、数据指纹、资本利得交易指标和完整HTML。查看归档不会使用当前数据重算。</p></div></section>",
        unsafe_allow_html=True,
    )
    _render_experiment_history(show_report=True)


def _render_historical_result_page(experiment_id: str) -> None:
    experiment_root = (ROOT / "backtest_outputs" / "experiments").resolve()
    experiment_dir = (experiment_root / Path(experiment_id).name).resolve()
    if experiment_dir.parent != experiment_root or not (experiment_dir / "run_manifest.json").exists():
        _render_header(None, None, "history")
        st.error("未找到该历史实验，可能已移动或归档不完整。")
        st.markdown('<a class="table-link" href="?view=history" target="_self">返回历史实验</a>', unsafe_allow_html=True)
        return
    try:
        daily, signals, strategy_metrics, benchmark_metrics, config = load_experiment_result(experiment_dir)
    except Exception as exc:
        _render_header(None, None, "history")
        st.error(f"历史实验加载失败：{exc}")
        return
    diagnostics = _build_period_diagnostics(daily, signals)
    st.sidebar.markdown("## 策略控制台")
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
            st.query_params["view"] = f"history_result__{new_experiment_dir.name}"
            st.rerun()
    _render_header(config, True, "history")
    st.markdown(
        f'<div class="archive-breadcrumb"><a href="?view=history" target="_self">历史实验</a><span>/</span><strong>{escape(config.name)}</strong></div>',
        unsafe_allow_html=True,
    )
    _render_summary(strategy_metrics, benchmark_metrics, signals)
    _render_workspace(
        daily,
        signals,
        strategy_metrics,
        benchmark_metrics,
        diagnostics,
        config.name,
        config,
        experiment_dir,
    )


def _render_historical_sidebar(config: DashboardStrategyConfig, experiment_dir: Path) -> None:
    objective = config.objective.as_dict()
    st.sidebar.markdown("## 历史结果参数")
    st.sidebar.caption("参数来自该次实验归档，只读展示，不会被当前首页配置替换。")
    st.sidebar.markdown(f"**策略名称**  \n{config.name}")
    st.sidebar.caption(f"归档：{experiment_dir.name}")
    with st.sidebar.expander("搜索目标函数（归档）", expanded=True):
        rows = [
            {"项目": "累计资本利得BP", "系数": objective["capital_gain_bp_weight"]},
            {"项目": "资本利得超额BP", "系数": objective["capital_gain_excess_bp_weight"]},
            {"项目": "已平仓交易胜率", "系数": objective["capital_trade_win_rate_weight"]},
            {"项目": "资本利得回撤BP", "系数": objective["capital_gain_drawdown_bp_penalty"]},
            {"项目": "累计收益", "系数": objective["total_return_weight"]},
            {"项目": "超额收益", "系数": objective["excess_return_weight"]},
            {"项目": "夏普", "系数": objective["sharpe_weight"]},
            {"项目": "传统最大回撤", "系数": objective["max_drawdown_penalty"]},
            {"项目": "调仓周期胜率", "系数": objective["signal_win_rate_weight"]},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.code(
            f"目标分 = {objective['capital_gain_bp_weight']:g}×资本BP + "
            f"{objective['capital_gain_excess_bp_weight']:g}×超额BP + "
            f"{objective['capital_trade_win_rate_weight']:g}×交易胜率 + "
            f"{objective['capital_gain_drawdown_bp_penalty']:g}×资本回撤BP + "
            f"{objective['total_return_weight']:g}×收益 + "
            f"{objective['excess_return_weight']:g}×超额收益 + "
            f"{objective['sharpe_weight']:g}×夏普 + "
            f"{objective['max_drawdown_penalty']:g}×最大回撤 + "
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
    experiments = list_experiments(ROOT)
    full_start, full_end = _full_strategy_date_bounds()
    st.markdown(
        f"<div class='section-head'><h2>研究快照</h2><p>仅比较全历史 {full_start} 至 {full_end} · 点击策略名称查看归档结果</p></div>",
        unsafe_allow_html=True,
    )
    valid = experiments.dropna(subset=["累计资本利得_BP"]) if not experiments.empty else pd.DataFrame()
    if not valid.empty:
        starts = pd.to_datetime(valid["回测起始日期"], errors="coerce").dt.strftime("%Y-%m-%d")
        ends = pd.to_datetime(valid["回测结束日期"], errors="coerce").dt.strftime("%Y-%m-%d")
        valid = valid.loc[(starts == full_start) & (ends == full_end)].copy()
    if valid.empty:
        st.caption("尚无与当前数据源完整区间一致的归档实验。请用完整区间运行策略后再进行横向比较。")
        return
    slides = []
    for index, (advantage, highlight_keys, row) in enumerate(_pick_featured_experiments(valid)):
        experiment_id = quote(Path(str(row["实验目录"])).name)
        metrics = [
            ("capital", "累计资本利得", row.get("累计资本利得_BP"), row.get("基准累计资本利得_BP"), "bp"),
            ("win", "已平仓交易胜率", row.get("资本利得交易胜率"), None, "pct"),
            ("average", "平均单笔资本利得", row.get("平均单笔资本利得_BP"), None, "bp"),
            ("capital_drawdown", "资本利得最大回撤", row.get("资本利得最大回撤_BP"), row.get("基准资本利得最大回撤_BP"), "bp"),
        ]
        metric_html = "".join(
            _featured_metric_html(label, value, benchmark, unit, key in highlight_keys)
            for key, label, value, benchmark, unit in metrics
        )
        slides.append(
            f'<article class="featured-slide" style="animation-delay:-{index * 4}s">'
            f'<div class="featured-slide-header"><div class="featured-strategy"><a href="?view=history_result__{experiment_id}" target="_self">{escape(str(row["策略名称"]))}</a></div>'
            f'<div class="featured-badge">{escape(advantage)}</div></div><div class="featured-metrics">{metric_html}</div></article>'
        )
    carousel_class = "featured-carousel" if len(slides) == 3 else "featured-carousel single"
    st.markdown(f'<section class="{carousel_class}">{"".join(slides)}</section>', unsafe_allow_html=True)


def _pick_featured_experiments(frame: pd.DataFrame) -> list[tuple[str, set[str], pd.Series]]:
    latest = frame.sort_values("运行时间", ascending=False).drop_duplicates("策略名称").copy()
    for column in ["累计资本利得_BP", "资本利得交易胜率", "平均单笔资本利得_BP", "资本利得最大回撤_BP", "已平仓交易数"]:
        latest[column] = pd.to_numeric(latest[column], errors="coerce")
    eligible = latest.loc[latest["已平仓交易数"].fillna(0) >= 3].copy()
    if eligible.empty:
        eligible = latest
    specifications = [
        ("累计资本利得领先", "capital", "累计资本利得_BP", False),
        ("已平仓胜率领先", "win", "资本利得交易胜率", False),
        ("资本回撤绝对值最小", "capital_drawdown", "资本利得最大回撤_BP", True),
    ]
    selected: list[tuple[list[str], set[str], pd.Series]] = []
    selected_index: dict[str, int] = {}
    for advantage, highlight_key, column, absolute_minimum in specifications:
        candidates = eligible.dropna(subset=[column]).copy()
        if absolute_minimum:
            candidates["_rank"] = candidates[column].abs()
            candidates = candidates.sort_values("_rank", ascending=True)
        else:
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
    if len(selected) < 3:
        for _, row in eligible.sort_values("累计资本利得_BP", ascending=False).iterrows():
            experiment_dir = str(row["实验目录"])
            if experiment_dir in selected_index:
                continue
            selected_index[experiment_dir] = len(selected)
            selected.append((["资本利得表现靠前"], {"capital"}, row))
            if len(selected) == 3:
                break
    return [(" · ".join(advantages), highlights, row) for advantages, highlights, row in selected[:3]]


def _featured_metric_html(label: str, value: object, benchmark: object, unit: str, highlighted: bool) -> str:
    if value is None or pd.isna(value):
        value_text = "暂无"
    elif unit == "pct":
        value_text = _pct(value)
    else:
        value_text = f"{float(value):.2f} BP"
    if label == "已平仓交易胜率":
        benchmark_text = "基准持续持有，无已平仓胜率"
    elif label == "平均单笔资本利得":
        benchmark_text = "基准持续持有，无已平仓平均单笔"
    elif benchmark is None or pd.isna(benchmark):
        benchmark_text = "基准暂无可比数据"
    elif unit == "pct":
        benchmark_text = f"满仓基准 {_pct(benchmark)}"
    else:
        benchmark_text = f"满仓基准 {float(benchmark):.2f} BP"
    highlight_class = " highlight" if highlighted else ""
    return (
        f'<div class="featured-metric{highlight_class}"><div class="featured-metric-label">{escape(label)}</div>'
        f'<div class="featured-metric-value">{escape(value_text)}</div>'
        f'<div class="featured-metric-benchmark">{escape(benchmark_text)}</div></div>'
    )


def _render_experiment_history(show_report: bool = False) -> None:
    experiments = list_experiments(ROOT)
    if experiments.empty:
        st.caption("尚无已归档实验。运行一次回测后，这里会保留配置、数据版本、指标、诊断和HTML报告。")
        return

    display = experiments.copy()
    display.insert(
        0,
        "打开结果",
        [f"?view=history_result__{quote(Path(str(path)).name)}" for path in display["实验目录"]],
    )
    display = display.drop(columns=["实验目录"])
    display = display[
        [
            "打开结果", "运行时间", "策略名称", "运行来源", "回测区间", "累计资本利得_BP", "资本利得超额_BP",
            "资本利得交易胜率", "已平仓交易数", "平均单笔资本利得_BP", "资本利得最大回撤_BP",
            "策略累计收益率", "最大回撤", "夏普比率",
        ]
    ]
    for column in ["策略累计收益率", "最大回撤", "资本利得交易胜率"]:
        display[column] = display[column].map(_pct)
    for column in ["累计资本利得_BP", "资本利得超额_BP", "平均单笔资本利得_BP", "资本利得最大回撤_BP"]:
        display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.2f} BP")
    display["夏普比率"] = pd.to_numeric(display["夏普比率"], errors="coerce").map(
        lambda value: "" if pd.isna(value) else f"{value:.3f}"
    )
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=min(860, 42 + len(display) * 36),
        column_config={
            "打开结果": st.column_config.LinkColumn("打开", display_text="查看", width="small"),
            "策略名称": st.column_config.TextColumn("策略名称", width="large"),
        },
    )
    st.caption("每一行点击“查看”可直接打开该次历史回测；下方选择框仅用于下载归档文件。")

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
    baseline = _search_baseline_controls()
    objective = _search_objective_controls(baseline.objective)
    baseline = DashboardStrategyConfig(
        name=baseline.name,
        weights=baseline.weights,
        thresholds=baseline.thresholds,
        positions=baseline.positions,
        objective=objective,
        backtest_start=None,
        backtest_end=None,
    )
    _persist_search_draft(baseline, st.session_state.get("search_draft_source_path", ""))
    left, right = st.columns(2)
    with left:
        st.markdown("#### 因子权重搜索")
        st.caption("遍历因子权重；使用上方设置的定性阈值和仓位制度。基线权重只用于对照，不限制候选空间。")
        if st.button("运行权重搜索", type="primary", use_container_width=True, key="run_weight_search_web"):
            with st.spinner("正在搜索权重并生成报告..."):
                metrics = run_dashboard_weight_search_v1(ROOT, objective_config=objective, base_config=baseline)
            st.success(f"权重搜索完成：累计资本利得 {metrics['capital_gain_total_bp']:.2f} BP，逐笔胜率 {metrics['capital_gain_trade_win_rate']:.2%}")
        _render_latest_search_result_link("权重向量化搜索", "查看最近权重搜索最优策略")
    with right:
        st.markdown("#### 定性阈值与看空规则搜索")
        st.caption("使用上方设置的因子权重和仓位作为基线，搜索定性阈值、看空总分和确认条件。")
        if st.button("运行阈值搜索", type="primary", use_container_width=True, key="run_threshold_search_web"):
            with st.spinner("正在搜索阈值并生成报告..."):
                metrics = run_threshold_research(ROOT, objective_config=objective, base_config_override=baseline)
            st.success(f"阈值搜索完成：累计资本利得 {metrics['capital_gain_total_bp']:.2f} BP，逐笔胜率 {metrics['capital_gain_trade_win_rate']:.2%}")
        _render_latest_search_result_link("阈值向量化搜索", "查看最近阈值搜索最优策略")

    report_choice = st.segmented_control("查看搜索报告", ["权重搜索", "阈值搜索"], default="权重搜索")
    report_path = (
        ROOT / "backtest_outputs" / "dashboard_weight_search_v1" / "权重搜索报告.html"
        if report_choice == "权重搜索"
        else ROOT / "backtest_outputs" / "阈值调参实验_v1" / "阈值调参报告.html"
    )
    if report_path.exists():
        st.caption(f"报告更新时间：{datetime.fromtimestamp(report_path.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
        report_html = report_path.read_text(encoding="utf-8")
        components.html(report_html, height=_report_embed_height(report_html), scrolling=False)
    else:
        st.info("尚未生成该搜索报告，请先运行对应搜索。")


def _render_latest_search_result_link(source: str, label: str) -> None:
    experiments = list_experiments(ROOT)
    if experiments.empty:
        return
    matches = experiments.loc[experiments["运行来源"] == source]
    if matches.empty:
        st.caption("尚无可查看的搜索最优策略归档。")
        return
    latest = matches.iloc[0]
    experiment_id = quote(Path(str(latest["实验目录"])).name)
    st.markdown(
        f'<a class="search-result-link" href="?view=history_result__{experiment_id}" target="_self"><strong>{escape(label)}</strong><span>{escape(str(latest["策略名称"]))}</span></a>',
        unsafe_allow_html=True,
    )


def _search_baseline_controls() -> DashboardStrategyConfig:
    config_paths = sorted((ROOT / CONFIG_DIR).glob("*.json"))
    config_paths += sorted((ROOT / CONFIG_DIR / "experiments").glob("*.json"))
    if not config_paths:
        st.error("没有可用的搜索基线 JSON。")
        st.stop()
    persisted_source, persisted_config = _load_search_draft()
    default_path = Path(persisted_source) if persisted_source else ROOT / CONFIG_DIR / "策略02_最佳权重_中性0.5.json"
    default_index = config_paths.index(default_path) if default_path in config_paths else 0
    selected_path = st.selectbox(
        "搜索基线配置",
        config_paths,
        index=default_index,
        format_func=lambda path: (
            f"{'搜索结果' if path.parent.name == 'experiments' else '基础配置'} · {path.stem}"
        ),
        key="search_baseline_config",
        help="权重搜索继承其阈值和仓位；阈值搜索继承其权重和仓位。",
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
        "可选项包括 configs 根目录中的初始/手动保存配置，以及 configs/experiments 中的搜索最优配置。"
        "所有搜索强制使用完整历史；普通历史回测归档不会自动加入此下拉框。"
    )

    with st.expander("因子权重", expanded=True):
        columns = st.columns(3)
        weight_values: dict[str, float] = {}
        for index, (key, label) in enumerate(WEIGHT_LABELS.items()):
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
            ("spread_change_bp", "利差周变化阈值（BP）", 0.5, 10.0, 0.5),
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
        confirmations = columns[2].selectbox("看空连续确认周数", [1, 2, 3], index=int(base.positions.bearish_confirmation_periods) - 1, key=f"{key_prefix}_confirmations")
        position_values = {
            "bullish_threshold": bullish_threshold, "bearish_threshold": bearish_threshold,
            "bullish_position": bullish_position, "neutral_position": neutral_position, "bearish_position": bearish_position,
            "bearish_min_core_factors": min_modules, "bearish_require_supply_or_demand": int(require_supply),
            "bearish_confirmation_periods": confirmations,
        }
    return DashboardStrategyConfig(
        name=f"搜索基线_{base.name}", weights=DashboardWeights(**weight_values),
        thresholds=DashboardThresholds(**threshold_values), positions=DashboardPositionPolicy(**position_values),
        objective=base.objective, backtest_start=None, backtest_end=None,
    )


def _seed_search_draft(base: DashboardStrategyConfig) -> None:
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
    }
    for field, widget_name in position_key_map.items():
        value = getattr(base.positions, field)
        if field == "bearish_require_supply_or_demand":
            value = bool(value)
        st.session_state[f"search_draft_{widget_name}"] = value
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
    capital_col, risk_col, traditional_col = st.columns(3)
    with capital_col:
        capital_gain_bp_weight = st.number_input("累计资本利得BP权重", value=float(defaults.capital_gain_bp_weight), step=0.1, key="search_capital_gain_bp_weight")
        capital_gain_excess_bp_weight = st.number_input("资本利得超额BP权重", value=float(defaults.capital_gain_excess_bp_weight), step=0.05, key="search_capital_gain_excess_bp_weight")
        capital_trade_win_rate_weight = st.number_input("已平仓交易胜率权重", value=float(defaults.capital_trade_win_rate_weight), step=1.0, key="search_capital_trade_win_rate_weight")
    with risk_col:
        capital_gain_drawdown_bp_penalty = st.number_input("资本利得回撤BP系数", value=float(defaults.capital_gain_drawdown_bp_penalty), step=0.1, key="search_capital_gain_drawdown_bp_penalty", help="回撤指标本身为负数；正系数会形成惩罚。")
        max_drawdown_penalty = st.number_input("传统最大回撤系数", value=float(defaults.max_drawdown_penalty), step=0.1, key="search_max_drawdown_penalty", help="最大回撤本身为负数；正系数会形成惩罚。")
        signal_win_rate_weight = st.number_input("调仓周期胜率权重（辅助）", value=float(defaults.signal_win_rate_weight), step=0.1, key="search_signal_win_rate_weight")
    with traditional_col:
        total_return_weight = st.number_input("累计收益权重", value=float(defaults.total_return_weight), step=0.1, key="search_total_return_weight")
        excess_return_weight = st.number_input("超额收益权重", value=float(defaults.excess_return_weight), step=0.05, key="search_excess_return_weight")
        sharpe_weight = st.number_input("夏普权重", value=float(defaults.sharpe_weight), step=0.05, key="search_sharpe_weight")
    st.code(
        f"目标分 = {capital_gain_bp_weight:g}×累计资本利得BP + {capital_gain_excess_bp_weight:g}×资本利得超额BP "
        f"+ {capital_trade_win_rate_weight:g}×已平仓交易胜率 + {capital_gain_drawdown_bp_penalty:g}×资本利得回撤BP "
        f"+ {total_return_weight:g}×累计收益 + {excess_return_weight:g}×超额收益 + {sharpe_weight:g}×夏普 "
        f"+ {max_drawdown_penalty:g}×最大回撤 + {signal_win_rate_weight:g}×调仓周期胜率",
        language=None,
    )
    return ObjectiveConfig(
        total_return_weight=total_return_weight,
        excess_return_weight=excess_return_weight,
        sharpe_weight=sharpe_weight,
        max_drawdown_penalty=max_drawdown_penalty,
        signal_win_rate_weight=signal_win_rate_weight,
        capital_gain_bp_weight=capital_gain_bp_weight,
        capital_gain_excess_bp_weight=capital_gain_excess_bp_weight,
        capital_trade_win_rate_weight=capital_trade_win_rate_weight,
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
    st.caption("资本利得是主口径；carry只用于解释传统总收益。资本利得BP是价格收益的基点数，不是收益率曲线变动BP。")
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
        "strategy_capital_cum": "策略累计资本利得",
        "benchmark_carry_cum": "基准累计 Carry",
        "benchmark_capital_cum": "基准累计资本利得",
        "carry_excess_cum": "Carry 超额",
        "capital_excess_cum": "资本利得超额",
    }
    latest = attribution.iloc[-1].drop(labels=["date"]).rename(index=labels).map(_pct).rename("累计贡献").reset_index()
    latest.columns = ["归因项目", "累计贡献"]
    _render_theme_table(latest, numeric_columns={"累计贡献"})


def _render_diagnostics_tab(diagnostics: pd.DataFrame) -> None:
    st.markdown("#### 周度错判诊断")
    st.caption("这里按周度信号周期做判断归因，不等同于开平仓交易笔数；逐笔交易请查看“交易表现”页。")
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


def _render_latest_signal(signals: pd.DataFrame) -> None:
    latest = signals.iloc[-1]
    core = pd.DataFrame(
        [
            {"项目": "信号日期", "数值": _latest_signal_date(latest)},
            {"项目": "定性结论", "数值": latest.get("结论", "")},
            {"项目": "总分", "数值": f"{float(latest.get('总分', 0)):.1f}"},
            {"项目": "目标仓位", "数值": f"{float(latest.get('仓位', 0)):.1f}"},
        ]
    )
    st.markdown("#### 最新执行信号")
    _render_theme_table(core, numeric_columns={"数值"})
    st.markdown("#### 因子判断与得分")
    factor_rows = []
    for score_column in [column for column in latest.index if str(column).endswith("_得分")]:
        factor_name = str(score_column)[: -len("_得分")]
        score = pd.to_numeric(latest[score_column], errors="coerce")
        factor_rows.append(
            {
                "因子": factor_name.replace("_", " / "),
                "数值": "" if pd.isna(score) else f"{float(score):.1f}",
                "定性": latest.get(f"{factor_name}_定性", "中性"),
            }
        )
    _render_theme_table(pd.DataFrame(factor_rows), numeric_columns={"数值"}, qualitative_column="定性")


def _render_config_snapshot(config: DashboardStrategyConfig) -> None:
    groups = {
        "因子权重": config.weights.as_dict(),
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
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


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
) -> None:
    numeric_columns = numeric_columns or set()
    html_columns = html_columns or set()
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
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
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    wrapper_classes = "research-table-wrap" + (" scrollable" if scrollable else "")
    table_classes = "research-table" + (" wide" if wide else "")
    st.markdown(
        f'<div class="{wrapper_classes}"><table class="{table_classes}"><thead><tr>{headers}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _semantic_tone(value: str) -> str:
    if value in {"利多", *DIAGNOSTIC_GOOD_TYPES}:
        return "bullish"
    if value in {"利空", *DIAGNOSTIC_BAD_TYPES}:
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
        ("满仓基准累计资本利得", "benchmark_capital_cum_bp", GREEN, None),
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
    figure.update_yaxes(title_text="累计资本利得（BP）", ticksuffix=" BP", secondary_y=False)
    figure.update_yaxes(title_text="仓位", range=[-1.1, 1.1], showgrid=False, secondary_y=True)
    return figure


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
        ("策略资本利得", "strategy_capital_cum", CORAL, 2.5, None),
        ("基准 Carry", "benchmark_carry_cum", GREEN_LIGHT, 1.7, None),
        ("基准资本利得", "benchmark_capital_cum", INK, 1.7, None),
        ("Carry 超额", "carry_excess_cum", GOLD, 1.6, "dash"),
        ("资本利得超额", "capital_excess_cum", "#7b8180", 1.6, "dash"),
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
    _apply_chart_theme(figure, "收益归因：Carry 与资本利得累计贡献", 500)
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
