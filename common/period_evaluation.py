from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from common.market_data import CONDITIONAL_BENCHMARK_ID, CONDITIONAL_BENCHMARK_NAME
from common.performance import performance_metrics
from common.trade_metrics import capital_gain_trade_metrics


DEFAULT_TRAINING_END = "2025-06-30"
RECENT_START = "2026-01-01"


def period_ranges(training_end: str, data_start: object, data_end: object) -> OrderedDict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(data_start).normalize()
    end = pd.Timestamp(data_end).normalize()
    requested_cutoff = pd.Timestamp(training_end).normalize()
    # A legacy rolling archive may contain only the stitched OOS path.  Do
    # not clamp a training cutoff before that path to its first OOS row: that
    # would fabricate a one-day "搜索期" and exclude the first OOS day.
    if requested_cutoff < start:
        return OrderedDict(
            [
                ("搜索期", (start, start - pd.Timedelta(days=1))),
                ("样本外", (start, end)),
                ("2026年以来", (max(pd.Timestamp(RECENT_START), start), end)),
                ("全区间", (start, end)),
            ]
        )
    cutoff = min(requested_cutoff, end)
    out_of_sample_start = cutoff + pd.Timedelta(days=1)
    recent_start = max(pd.Timestamp(RECENT_START), out_of_sample_start)
    return OrderedDict(
        [
            ("搜索期", (start, cutoff)),
            ("样本外", (out_of_sample_start, end)),
            ("2026年以来", (recent_start, end)),
            ("全区间", (start, end)),
        ]
    )


def evaluate_periods(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    training_end: str = DEFAULT_TRAINING_END,
    benchmark_name: str = CONDITIONAL_BENCHMARK_NAME,
    include_shortcuts: bool = False,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    ranges = period_ranges(training_end, daily["date"].min(), daily["date"].max())
    if include_shortcuts:
        data_start = pd.Timestamp(daily["date"].min()).normalize()
        data_end = pd.Timestamp(daily["date"].max()).normalize()
        ranges = OrderedDict(
            [(key, ranges[key]) for key in ("搜索期", "样本外") if key in ranges]
            + [
                ("2024年", (max(data_start, pd.Timestamp("2024-01-01")), min(data_end, pd.Timestamp("2024-12-31")))),
                ("2025年", (max(data_start, pd.Timestamp("2025-01-01")), min(data_end, pd.Timestamp("2025-12-31")))),
                ("2025年以来", (max(data_start, pd.Timestamp("2025-01-01")), data_end)),
                ("2025年下半年以来", (max(data_start, pd.Timestamp("2025-07-01")), data_end)),
                ("2026年", (max(data_start, pd.Timestamp("2026-01-01")), min(data_end, pd.Timestamp("2026-12-31")))),
                ("2026年以来", ranges["2026年以来"]),
                ("全区间", (data_start, data_end)),
            ]
        )
    for label, (start, end) in ranges.items():
        sliced_daily, sliced_signals, strategy_metrics, benchmark_metrics = evaluate_period(
            daily, signals, start, end, benchmark_name
        )
        results[label] = {
            "daily": sliced_daily,
            "signals": sliced_signals,
            "strategy_metrics": strategy_metrics,
            "benchmark_metrics": benchmark_metrics,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
        }
    return results


def evaluate_period(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    start: object,
    end: object,
    benchmark_name: str = CONDITIONAL_BENCHMARK_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    frame = daily.loc[(daily["date"] >= start_ts) & (daily["date"] <= end_ts)].copy().reset_index(drop=True)
    if frame.empty:
        return frame, signals.iloc[:0].copy(), {}, {}

    anchor_columns = [
        "strategy_return", "total_return", "strategy_carry_return", "strategy_capital_return",
        "benchmark_carry_return", "benchmark_capital_return", "carry_excess_return", "capital_excess_return",
        "strategy_capital_bp", "benchmark_capital_bp", "capital_excess_bp",
    ]
    for column in anchor_columns:
        if column in frame:
            frame.loc[0, column] = 0.0
    _rebuild_paths(frame)

    used_signal_dates = pd.to_datetime(frame["signal_date"], errors="coerce").dropna().unique()
    signal_frame = signals.loc[pd.to_datetime(signals["signal_date"], errors="coerce").isin(used_signal_dates)].copy()
    strategy_metrics = performance_metrics(frame, return_col="strategy_return", nav_col="strategy_nav")
    benchmark_metrics = performance_metrics(frame, return_col="total_return", nav_col="benchmark_nav_rebased")
    strategy_metrics.update(_signal_period_metrics(frame, "strategy_return"))
    benchmark_metrics.update(_signal_period_metrics(frame, "total_return"))
    strategy_metrics.update(capital_gain_trade_metrics(frame, "strategy_capital_bp", position_col="仓位"))
    comparison_col = "comparison_position" if "comparison_position" in frame else None
    benchmark_metrics.update(capital_gain_trade_metrics(frame, "benchmark_capital_bp", position_col=comparison_col))
    benchmark_metrics["benchmark_id"] = CONDITIONAL_BENCHMARK_ID
    benchmark_metrics["benchmark_name"] = benchmark_name
    strategy_metrics["capital_gain_excess_bp"] = (
        float(strategy_metrics["capital_gain_total_bp"]) - float(benchmark_metrics["capital_gain_total_bp"])
    )
    periods = max(len(frame) - 1, 1)
    strategy_metrics["capital_gain_excess_annualized_bp"] = strategy_metrics["capital_gain_excess_bp"] * 252.0 / periods
    return frame, signal_frame, strategy_metrics, benchmark_metrics


def period_summary_frame(periods: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for label, result in periods.items():
        metrics = result.get("strategy_metrics", {})
        benchmark = result.get("benchmark_metrics", {})
        if not metrics:
            continue
        rows.append(
            {
                "区间": label,
                "起始日期": result["start"],
                "结束日期": result["end"],
                "累计资本利得_BP": metrics.get("capital_gain_total_bp"),
                "年化资本利得_BP": metrics.get("capital_gain_annualized_bp"),
                "条件基准资本利得_BP": benchmark.get("capital_gain_total_bp"),
                "资本利得超额_BP": metrics.get("capital_gain_excess_bp"),
                "年化资本利得超额_BP": metrics.get("capital_gain_excess_annualized_bp"),
                "已平仓胜率": metrics.get("capital_gain_trade_win_rate"),
                "已平仓交易数": metrics.get("capital_gain_closed_trade_count"),
                "平均单笔_BP": metrics.get("capital_gain_avg_trade_bp"),
                "平均亏损_BP": metrics.get("capital_gain_avg_loss_bp"),
                "资本利得最大回撤_BP": metrics.get("capital_gain_max_drawdown_bp"),
                "平均持有交易日": metrics.get("capital_gain_avg_holding_days"),
            }
        )
    return pd.DataFrame(rows)


def generalization_summary(periods: dict[str, dict[str, object]]) -> dict[str, object]:
    train = periods.get("搜索期", {}).get("strategy_metrics", {})
    test = periods.get("样本外", {}).get("strategy_metrics", {})
    if not train or not test:
        return {}
    train_annual = _number(train.get("capital_gain_annualized_bp"))
    test_annual = _number(test.get("capital_gain_annualized_bp"))
    train_excess = _number(train.get("capital_gain_excess_annualized_bp"))
    test_excess = _number(test.get("capital_gain_excess_annualized_bp"))
    return {
        "年化资本利得衰减_BP": test_annual - train_annual,
        "年化超额衰减_BP": test_excess - train_excess,
        "样本外年化保留率": None if abs(train_annual) < 1e-12 else test_annual / abs(train_annual),
        "样本外胜率变化": _optional_difference(test.get("capital_gain_trade_win_rate"), train.get("capital_gain_trade_win_rate")),
        "样本外回撤变化_BP": _optional_difference(test.get("capital_gain_max_drawdown_bp"), train.get("capital_gain_max_drawdown_bp")),
    }


def top_stability_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    oos_capital = pd.to_numeric(frame["oos_capital_gain_total_bp"], errors="coerce")
    oos_excess = pd.to_numeric(frame["oos_capital_gain_excess_bp"], errors="coerce")
    train_rank = pd.Series(range(1, len(frame) + 1), index=frame.index, dtype=float)
    oos_rank = oos_capital.rank(ascending=False, method="average")
    rank_correlation = (
        float(train_rank.corr(oos_rank, method="spearman"))
        if len(frame) > 1 and oos_rank.nunique(dropna=True) > 1
        else None
    )
    return {
        "Top候选数": int(len(frame)),
        "样本外资本利得为正占比": float((oos_capital > 0).mean()),
        "样本外超额为正占比": float((oos_excess > 0).mean()),
        "样本外资本利得中位数_BP": float(oos_capital.median()),
        "训练与样本外排名相关性": rank_correlation,
        "训练期第一名的样本外排名": int(oos_rank.iloc[0]) if oos_rank.notna().iloc[0] else None,
    }


def append_search_evaluation_html(
    path: object,
    period_frame: pd.DataFrame,
    stability_frame: pd.DataFrame,
    generalization: dict[str, object],
    stability: dict[str, object],
) -> None:
    report_path = pd.io.common.stringify_path(path)
    html = open(report_path, encoding="utf-8").read()
    summary = pd.DataFrame(
        [{"指标": key, "数值": value} for key, value in {**generalization, **stability}.items()]
    )
    section = (
        "<section class='section'><h2>样本内外检验</h2>"
        "<p>候选参数只按搜索期目标函数排序；样本外、2026年以来和全区间结果均不参与参数选择。</p>"
        f"{period_frame.to_html(index=False, border=0)}"
        "<h2>Top候选稳定性</h2>"
        f"{summary.to_html(index=False, border=0)}"
        f"{stability_frame.head(20).to_html(index=False, border=0)}"
        "</section>"
    )
    html = html.replace("</body>", section + "</body>")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def _rebuild_paths(frame: pd.DataFrame) -> None:
    cumulative_pairs = [
        ("strategy_carry_return", "strategy_carry_cum"),
        ("strategy_capital_return", "strategy_capital_cum"),
        ("benchmark_carry_return", "benchmark_carry_cum"),
        ("benchmark_capital_return", "benchmark_capital_cum"),
        ("carry_excess_return", "carry_excess_cum"),
        ("capital_excess_return", "capital_excess_cum"),
        ("strategy_capital_bp", "strategy_capital_cum_bp"),
        ("benchmark_capital_bp", "benchmark_capital_cum_bp"),
        ("capital_excess_bp", "capital_excess_cum_bp"),
    ]
    for source, target in cumulative_pairs:
        if source in frame:
            frame[target] = pd.to_numeric(frame[source], errors="coerce").fillna(0.0).cumsum()
    frame["strategy_nav"] = (1.0 + pd.to_numeric(frame["strategy_return"], errors="coerce").fillna(0.0)).cumprod()
    frame["benchmark_nav_rebased"] = (1.0 + pd.to_numeric(frame["total_return"], errors="coerce").fillna(0.0)).cumprod()
    frame["excess_nav"] = frame["strategy_nav"] / frame["benchmark_nav_rebased"]


def _signal_period_metrics(daily: pd.DataFrame, return_col: str) -> dict[str, object]:
    period_returns = daily.groupby("signal_date")[return_col].apply(
        lambda values: (1.0 + pd.to_numeric(values, errors="coerce").fillna(0.0)).prod() - 1.0
    )
    winning = int((period_returns > 0).sum())
    return {
        "signal_period_count": int(len(period_returns)),
        "winning_signal_periods": winning,
        "signal_period_win_rate": float(winning / len(period_returns)) if len(period_returns) else None,
        "avg_signal_period_return": float(period_returns.mean()) if len(period_returns) else None,
    }


def _number(value: object) -> float:
    return 0.0 if value is None or pd.isna(value) else float(value)


def _optional_difference(left: object, right: object) -> float | None:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return None
    return float(left) - float(right)
