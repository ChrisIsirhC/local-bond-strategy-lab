from __future__ import annotations

from pathlib import Path
import base64
import html as html_lib
import tempfile
from io import BytesIO

import pandas as pd

from common.trade_metrics import capital_gain_trade_table


METRIC_LABELS = {
    "start_date": "起始日期",
    "end_date": "结束日期",
    "trading_days": "交易日数",
    "final_nav": "期末净值",
    "total_return": "累计收益率",
    "annual_return": "年化收益率",
    "risk_free_rate_annual": "无风险利率_年化",
    "excess_annual_return": "超额年化收益率",
    "annual_volatility": "年化波动率",
    "sharpe": "夏普比率",
    "max_drawdown": "最大回撤",
    "max_drawdown_start": "最大回撤起点",
    "max_drawdown_end": "最大回撤终点",
    "calmar": "Calmar比率",
    "win_rate": "日胜率_参考",
    "signal_period_count": "调仓周期数",
    "winning_signal_periods": "盈利调仓周期数",
    "signal_period_win_rate": "调仓周期胜率",
    "avg_signal_period_return": "单周期平均收益",
    "capital_gain_total_bp": "累计资本利得_BP",
    "capital_gain_annualized_bp": "年化资本利得_BP",
    "capital_gain_trade_count": "资本利得交易笔数",
    "capital_gain_closed_trade_count": "已平仓交易笔数",
    "capital_gain_winning_trades": "资本利得盈利笔数",
    "capital_gain_losing_trades": "资本利得亏损笔数",
    "capital_gain_flat_trades": "资本利得持平笔数",
    "capital_gain_trade_win_rate": "资本利得交易胜率",
    "capital_gain_avg_trade_bp": "平均单笔资本利得_BP",
    "capital_gain_median_trade_bp": "资本利得中位数_BP",
    "capital_gain_avg_win_bp": "平均每笔盈利_BP",
    "capital_gain_avg_loss_bp": "平均每笔亏损_BP",
    "capital_gain_profit_loss_ratio": "资本利得盈亏比",
    "capital_gain_best_trade_bp": "最佳交易_BP",
    "capital_gain_worst_trade_bp": "最差交易_BP",
    "capital_gain_avg_holding_days": "平均每笔持有交易日",
    "capital_gain_max_holding_days": "最长单笔持有交易日",
    "capital_gain_max_drawdown_bp": "资本利得最大回撤_BP",
    "capital_gain_max_drawdown_start": "资本利得最大回撤起点",
    "capital_gain_max_drawdown_end": "资本利得最大回撤终点",
    "capital_gain_longest_losing_streak": "最长连续亏损笔数",
    "capital_gain_open_trade_count": "当前未平仓交易数",
    "capital_gain_open_trade_bp": "当前未平仓资本利得_BP",
}


def write_outputs(frame: pd.DataFrame, metrics: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_nav(frame, output_dir / "benchmark_nav.csv")
    _write_components(frame, output_dir / "return_components.csv")
    _write_metrics(metrics, output_dir / "performance_metrics.csv")
    _write_report(metrics, output_dir / "performance_report.md")
    _write_benchmark_html_report(frame, metrics, output_dir / "performance_report.html")
    _write_nav_chart(frame, output_dir / "benchmark_nav_chart.html")


def write_strategy_outputs(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
    output_dir: Path,
    *,
    persist_html_report: bool = True,
    persist_primary_csv: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if persist_primary_csv:
        _write_signal_score(signals, output_dir / "signal_score.csv")
        _write_strategy_nav(daily, output_dir / "strategy_nav.csv")
    diagnostics = _build_period_diagnostics(daily, signals)
    diagnostics.to_csv(output_dir / "period_diagnostics.csv", index=False, encoding="utf-8-sig")
    _write_capital_gain_trades(daily, output_dir / "capital_gain_trades.csv")
    _write_strategy_metrics(strategy_metrics, benchmark_metrics, output_dir / "performance_metrics.csv")
    if persist_html_report:
        _write_strategy_html_report(daily, signals, strategy_metrics, benchmark_metrics, output_dir / "performance_report.html", diagnostics)


def build_strategy_html_report(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
) -> bytes:
    """Build a downloadable report from archived numerical results on demand."""
    diagnostics = _build_period_diagnostics(daily, signals)
    with tempfile.TemporaryDirectory(prefix="local_bond_report_") as temporary_dir:
        path = Path(temporary_dir) / "performance_report.html"
        _write_strategy_html_report(daily, signals, strategy_metrics, benchmark_metrics, path, diagnostics)
        return path.read_bytes()


def _write_nav(frame: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "日期": frame["date"].dt.strftime("%Y-%m-%d"),
            "基准净值": frame["benchmark_nav"],
            "日收益率": frame["total_return"],
        }
    )
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _write_components(frame: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "日期": frame["date"].dt.strftime("%Y-%m-%d"),
            "到期收益率_百分比": frame["yield_pct"],
            "到期收益率_小数": frame["yield_decimal"],
            "收益率日变化_BP": frame["yield_change_bp"],
            "修正久期": frame["modified_duration"],
            "票息Carry收益": frame["carry_return"],
            "久期资本利得": frame["duration_pnl"],
            "合成总收益": frame["total_return"],
            "基准净值": frame["benchmark_nav"],
        }
    )
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _write_metrics(metrics: dict[str, object], path: Path) -> None:
    rows = [
        {"指标": METRIC_LABELS.get(key, key), "字段": key, "数值": value}
        for key, value in metrics.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_report(metrics: dict[str, object], path: Path) -> None:
    def fmt(value: object, pct: bool = False) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.2%}" if pct else f"{value:.4f}"
        return str(value)

    lines = [
        "# 10Y地方债长期持有基准回测",
        "",
        "## 口径",
        "",
        "- 标的：地方政府债到期收益率:10年。",
        "- 收益合成：票息 carry + 久期资本利得。",
        "- 久期：按10年平价债、年付息、当期收益率估算修正久期。",
        "- 基准：100%长期持有10Y地方债久期敞口。",
        "",
        "## 绩效",
        "",
        f"- 区间：{metrics['start_date']} 至 {metrics['end_date']}",
        f"- 交易日数：{metrics['trading_days']}",
        f"- 期末净值：{fmt(metrics['final_nav'])}",
        f"- 累计收益率：{fmt(metrics['total_return'], pct=True)}",
        f"- 年化收益率：{fmt(metrics['annual_return'], pct=True)}",
        f"- 无风险利率：{fmt(metrics['risk_free_rate_annual'], pct=True)}",
        f"- 超额年化收益率：{fmt(metrics['excess_annual_return'], pct=True)}",
        f"- 年化波动率：{fmt(metrics['annual_volatility'], pct=True)}",
        f"- 最大回撤：{fmt(metrics['max_drawdown'], pct=True)}",
        f"- 最大回撤区间：{metrics['max_drawdown_start']} 至 {metrics['max_drawdown_end']}",
        f"- 夏普比率：{fmt(metrics['sharpe'])}",
        f"- Calmar比率：{fmt(metrics['calmar'])}",
        f"- 日胜率（参考）：{fmt(metrics['win_rate'], pct=True)}",
        "",
        "## 后续",
        "",
        "本报告只验证长期持有基准。后续接入看板信号后，再输出策略净值、基准净值和超额净值。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_signal_score(signals: pd.DataFrame, path: Path) -> None:
    out = signals.copy()
    out["signal_date"] = out["signal_date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _write_strategy_nav(daily: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "日期": daily["date"].dt.strftime("%Y-%m-%d"),
            "信号日期": daily["signal_date"].dt.strftime("%Y-%m-%d"),
            "总分": daily["总分"],
            "结论": daily["结论"],
            "目标仓位": daily.get("目标仓位", daily["仓位"]),
            "仓位": daily["仓位"],
            "止盈止损事件": daily.get("止盈止损事件", pd.Series("", index=daily.index, dtype=object)),
            "条件基准仓位": daily.get("comparison_position", pd.Series(index=daily.index, dtype=float)),
            "交易标的到期收益率_百分比": daily.get("asset_yield_pct", pd.Series(index=daily.index, dtype=float)),
            "比较基准到期收益率_百分比": daily.get("yield_pct", pd.Series(index=daily.index, dtype=float)),
            "策略日收益率": daily["strategy_return"],
            "基准日收益率": daily["total_return"],
            "策略票息Carry收益": daily.get("strategy_carry_return", pd.Series(index=daily.index, dtype=float)),
            "策略资本利得收益": daily.get("strategy_capital_return", pd.Series(index=daily.index, dtype=float)),
            "基准票息Carry收益": daily.get("benchmark_carry_return", daily.get("carry_return", pd.Series(index=daily.index, dtype=float))),
            "基准资本利得收益": daily.get("benchmark_capital_return", daily.get("duration_pnl", pd.Series(index=daily.index, dtype=float))),
            "票息Carry超额": daily.get("carry_excess_return", pd.Series(index=daily.index, dtype=float)),
            "资本利得超额": daily.get("capital_excess_return", pd.Series(index=daily.index, dtype=float)),
            "策略资本利得_BP": daily.get("strategy_capital_bp", pd.Series(index=daily.index, dtype=float)),
            "基准资本利得_BP": daily.get("benchmark_capital_bp", pd.Series(index=daily.index, dtype=float)),
            "资本利得超额_BP": daily.get("capital_excess_bp", pd.Series(index=daily.index, dtype=float)),
            "策略累计资本利得_BP": daily.get("strategy_capital_cum_bp", pd.Series(index=daily.index, dtype=float)),
            "基准累计资本利得_BP": daily.get("benchmark_capital_cum_bp", pd.Series(index=daily.index, dtype=float)),
            "累计资本利得超额_BP": daily.get("capital_excess_cum_bp", pd.Series(index=daily.index, dtype=float)),
            "策略累计票息Carry": daily.get("strategy_carry_cum", pd.Series(index=daily.index, dtype=float)),
            "策略累计资本利得": daily.get("strategy_capital_cum", pd.Series(index=daily.index, dtype=float)),
            "基准累计票息Carry": daily.get("benchmark_carry_cum", pd.Series(index=daily.index, dtype=float)),
            "基准累计资本利得": daily.get("benchmark_capital_cum", pd.Series(index=daily.index, dtype=float)),
            "策略净值": daily["strategy_nav"],
            "基准净值": daily["benchmark_nav_rebased"],
            "超额净值": daily["excess_nav"],
        }
    )
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _write_capital_gain_trades(daily: pd.DataFrame, path: Path) -> None:
    trades = capital_gain_trade_table(daily)
    if trades.empty:
        pd.DataFrame(columns=["交易编号", "开仓日期", "平仓日期", "估值日期", "方向", "开仓仓位", "平均绝对仓位", "持有交易日", "策略资本利得_BP", "同期条件基准资本利得_BP", "资本利得超额_BP", "是否盈利", "是否已平仓", "状态"]).to_csv(path, index=False, encoding="utf-8-sig")
        return
    out = trades.rename(
        columns={
            "trade_id": "交易编号", "entry_date": "开仓日期", "exit_date": "平仓日期", "mark_date": "估值日期",
            "direction": "方向", "entry_position": "开仓仓位", "average_abs_position": "平均绝对仓位",
            "holding_days": "持有交易日", "strategy_capital_bp": "策略资本利得_BP",
            "benchmark_capital_bp": "同期条件基准资本利得_BP", "capital_excess_bp": "资本利得超额_BP",
            "is_win": "是否盈利", "is_closed": "是否已平仓", "status": "状态",
        }
    )
    for column in ["开仓日期", "平仓日期", "估值日期"]:
        out[column] = pd.to_datetime(out[column]).dt.strftime("%Y-%m-%d").fillna("")
    out["是否盈利"] = out["是否盈利"].map({True: "是", False: "否"})
    out["是否已平仓"] = out["是否已平仓"].map({True: "是", False: "否"})
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _classify_period(position: float, strategy_return: float, benchmark_return: float, excess_return: float) -> str:
    if position < 0 and strategy_return < 0:
        return "做空做反"
    if 0 <= position < 1 and benchmark_return > 0 and excess_return < 0:
        return "低仓少吃"
    if position >= 0 and benchmark_return < 0 and excess_return <= 0:
        return "该空没空"
    if position < 1 and benchmark_return < 0 and excess_return > 0:
        return "有效防守"
    if position > 0 and benchmark_return > 0 and strategy_return > 0:
        return "有效进攻"
    return "其他"


def _build_period_diagnostics(daily: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    signal_base = signals.copy()
    signal_base["signal_date"] = pd.to_datetime(signal_base["signal_date"], errors="coerce").dt.normalize()
    signal_base = (
        signal_base.dropna(subset=["signal_date"])
        .sort_values("signal_date")
        .drop_duplicates("signal_date", keep="last")
        .reset_index(drop=True)
    )
    if signal_base.empty:
        raise ValueError("无法生成周期诊断：没有可用策略信号")
    signal_base["下一信号日期"] = signal_base["signal_date"].shift(-1)
    factor_cols = [c for c in signal_base.columns if c.endswith("_定性") or c.endswith("_得分")]

    rows: list[dict[str, object]] = []
    for signal_date, group in daily.groupby("signal_date", sort=True):
        signal_date = pd.Timestamp(signal_date).normalize()
        strategy_return = (1.0 + pd.to_numeric(group["strategy_return"], errors="coerce").fillna(0.0)).prod() - 1.0
        benchmark_return = (1.0 + pd.to_numeric(group["total_return"], errors="coerce").fillna(0.0)).prod() - 1.0
        excess_return = strategy_return - benchmark_return
        strategy_capital_bp = pd.to_numeric(group["strategy_capital_bp"], errors="coerce").fillna(0.0).sum()
        benchmark_capital_bp = pd.to_numeric(group["benchmark_capital_bp"], errors="coerce").fillna(0.0).sum()
        capital_excess_bp = strategy_capital_bp - benchmark_capital_bp
        position = float(group["仓位"].iloc[0])
        # The rolling executor may add a final, synthetic execution boundary
        # at the last market day so the previous weekly signal covers the
        # remaining trading days.  That date is intentionally not persisted
        # as a newly generated signal.  Use the latest actual signal at or
        # before the execution boundary instead of assuming an exact index
        # match (which formerly raised KeyError for e.g. 2026-09-18).
        # Do not use ``.loc`` here.  Cloud deployments can deserialize the
        # archived datetime index with a different resolution; plain boolean
        # filtering keeps the selected signal as a row and never asks pandas
        # to resolve a timestamp as an exact index label.
        eligible_signals = signal_base[signal_base["signal_date"].le(signal_date)]
        if eligible_signals.empty:
            # A malformed result should be explicit rather than silently
            # borrowing a future signal and introducing look-ahead bias.
            raise ValueError(f"周期诊断缺少 {signal_date:%Y-%m-%d} 当日或此前的策略信号")
        # Keep this positional rather than indexing the timestamp again.
        # ``daily.signal_date`` can originate from a platform-normalized
        # calendar while persisted signals may carry a subtly different
        # datetime representation.  We have already chosen the latest valid
        # row by value; a second exact DatetimeIndex lookup can fail on
        # Streamlit Cloud even though that row exists.
        signal_row = eligible_signals.iloc[-1]
        row = {
            "信号日期": pd.to_datetime(signal_date).strftime("%Y-%m-%d"),
            "下一信号日期": pd.to_datetime(signal_row["下一信号日期"]).strftime("%Y-%m-%d")
            if pd.notna(signal_row["下一信号日期"])
            else "",
            "仓位": position,
            "结论": group["结论"].iloc[0],
            "周期策略收益": strategy_return,
            "周期基准收益": benchmark_return,
            "周期超额收益": excess_return,
            "周期carry贡献": pd.to_numeric(group["strategy_carry_return"], errors="coerce").fillna(0.0).sum(),
            "周期资本利得贡献": pd.to_numeric(group["strategy_capital_return"], errors="coerce").fillna(0.0).sum(),
            "周期策略资本利得_BP": strategy_capital_bp,
            "周期基准资本利得_BP": benchmark_capital_bp,
            "周期资本利得超额_BP": capital_excess_bp,
            "资本利得交易是否盈利": "是" if strategy_capital_bp > 0 else "否",
            "资本利得判断类型": _classify_period(position, strategy_capital_bp, benchmark_capital_bp, capital_excess_bp),
            "基准是否赚钱": "是" if benchmark_return > 0 else "否",
            "策略是否亏钱": "是" if strategy_return < 0 else "否",
            "错判类型": _classify_period(position, strategy_return, benchmark_return, excess_return),
        }
        for col in factor_cols:
            row[col] = signal_row[col]
        rows.append(row)

    return pd.DataFrame(rows)


def _write_strategy_metrics(
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
    path: Path,
) -> None:
    rows = []
    for key in strategy_metrics:
        rows.append(
            {
                "指标": METRIC_LABELS.get(key, key),
                "字段": key,
                "策略": strategy_metrics.get(key),
                "基准": benchmark_metrics.get(key),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _format_metric(value: object, pct: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2%}" if pct else f"{value:.4f}"
    return str(value)


def _metric_table(strategy_metrics: dict[str, object], benchmark_metrics: dict[str, object]) -> str:
    pct_keys = {
        "total_return",
        "annual_return",
        "risk_free_rate_annual",
        "excess_annual_return",
        "annual_volatility",
        "max_drawdown",
        "win_rate",
        "signal_period_win_rate",
        "avg_signal_period_return",
    }
    rows = []
    for key in strategy_metrics:
        rows.append(
            "<tr>"
            f"<td>{METRIC_LABELS.get(key, key)}</td>"
            f"<td>{_format_metric(strategy_metrics.get(key), key in pct_keys)}</td>"
            f"<td>{_format_metric(benchmark_metrics.get(key), key in pct_keys)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _capital_trade_metric_table(strategy_metrics: dict[str, object], benchmark_metrics: dict[str, object]) -> str:
    keys = [
        "capital_gain_total_bp",
        "capital_gain_trade_count",
        "capital_gain_closed_trade_count",
        "capital_gain_winning_trades",
        "capital_gain_losing_trades",
        "capital_gain_trade_win_rate",
        "capital_gain_avg_trade_bp",
        "capital_gain_median_trade_bp",
        "capital_gain_avg_win_bp",
        "capital_gain_avg_loss_bp",
        "capital_gain_profit_loss_ratio",
        "capital_gain_best_trade_bp",
        "capital_gain_worst_trade_bp",
        "capital_gain_avg_holding_days",
        "capital_gain_max_holding_days",
        "capital_gain_max_drawdown_bp",
        "capital_gain_longest_losing_streak",
        "capital_gain_open_trade_count",
        "capital_gain_open_trade_bp",
    ]
    rows = []
    for key in keys:
        strategy = strategy_metrics.get(key)
        benchmark = benchmark_metrics.get(key)
        if key == "capital_gain_trade_win_rate":
            strategy_text = _format_metric(strategy, True)
            benchmark_text = _format_metric(benchmark, True)
        elif key in {"capital_gain_trade_count", "capital_gain_closed_trade_count", "capital_gain_winning_trades", "capital_gain_losing_trades", "capital_gain_longest_losing_streak", "capital_gain_open_trade_count", "capital_gain_max_holding_days"}:
            strategy_text = "" if strategy is None else str(int(strategy))
            benchmark_text = "" if benchmark is None else str(int(benchmark))
        elif key == "capital_gain_avg_holding_days":
            strategy_text = "" if strategy is None else f"{float(strategy):.1f} 交易日"
            benchmark_text = "" if benchmark is None else f"{float(benchmark):.1f} 交易日"
        elif key == "capital_gain_profit_loss_ratio":
            strategy_text = _format_metric(strategy)
            benchmark_text = _format_metric(benchmark)
        else:
            strategy_text = "" if strategy is None else f"{float(strategy):.2f} BP"
            benchmark_text = "" if benchmark is None else f"{float(benchmark):.2f} BP"
        rows.append(
            f"<tr><td>{METRIC_LABELS[key]}</td><td>{strategy_text}</td><td>{benchmark_text}</td></tr>"
        )
    return "\n".join(rows)


def _attribution_table(daily: pd.DataFrame, benchmark_name: str = "10Y地方政府债") -> str:
    last = daily.iloc[-1]
    rows = [
        ("策略票息 carry", last.get("strategy_carry_cum", 0.0), "策略仓位 × 基准 carry，低仓位会少吃票息，负仓位会反向承担 carry。"),
        ("策略久期折算价格收益", last.get("strategy_capital_cum", 0.0), "传统净值辅助口径：策略仓位 × (-修正久期 × 收益率变化)。"),
        ("基准票息 carry", last.get("benchmark_carry_cum", 0.0), f"100% 长期持有 {benchmark_name} 久期敞口的 carry。"),
        ("基准久期折算价格收益", last.get("benchmark_capital_cum", 0.0), f"传统净值辅助口径：100%长期持有 {benchmark_name} 的久期折算价格收益。"),
        ("票息 carry 超额", last.get("carry_excess_cum", 0.0), "策略相对满仓持有少吃或多吃的 carry。"),
        ("资本利得超额", last.get("capital_excess_cum", 0.0), "策略相对满仓持有通过择时获得或损失的资本利得。"),
    ]
    return "\n".join(
        "<tr>"
        f"<td>{name}</td>"
        f"<td>{_format_metric(float(value), True)}</td>"
        f"<td>{desc}</td>"
        "</tr>"
        for name, value, desc in rows
    )


def _diagnostics_html(diagnostics: pd.DataFrame) -> str:
    if diagnostics.empty:
        return "<p>暂无调仓周期诊断数据。</p>"

    display_cols = [
        "信号日期",
        "下一信号日期",
        "仓位",
        "结论",
        "周期策略收益",
        "周期基准收益",
        "周期超额收益",
        "周期carry贡献",
        "周期资本利得贡献",
        "周期策略资本利得_BP",
        "周期基准资本利得_BP",
        "周期资本利得超额_BP",
        "资本利得交易是否盈利",
        "资本利得判断类型",
        "基准是否赚钱",
        "策略是否亏钱",
        "错判类型",
    ]
    display = diagnostics[display_cols].copy()
    for col in ["周期策略收益", "周期基准收益", "周期超额收益", "周期carry贡献", "周期资本利得贡献"]:
        display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    for col in ["周期策略资本利得_BP", "周期基准资本利得_BP", "周期资本利得超额_BP"]:
        display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["仓位"] = pd.to_numeric(display["仓位"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    type_counts = diagnostics["资本利得判断类型"].value_counts().rename_axis("资本利得判断类型").reset_index(name="周期数")
    counts_html = type_counts.to_html(index=False, escape=False)

    drag = diagnostics.copy()
    drag["周期策略资本利得_BP"] = pd.to_numeric(drag["周期策略资本利得_BP"], errors="coerce")
    drag = drag.sort_values("周期策略资本利得_BP", ascending=True).head(12)
    drag_display = drag[display_cols].copy()
    for col in ["周期策略收益", "周期基准收益", "周期超额收益", "周期carry贡献", "周期资本利得贡献"]:
        drag_display[col] = pd.to_numeric(drag_display[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    for col in ["周期策略资本利得_BP", "周期基准资本利得_BP", "周期资本利得超额_BP"]:
        drag_display[col] = pd.to_numeric(drag_display[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    drag_display["仓位"] = pd.to_numeric(drag_display["仓位"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    return "\n".join(
        [
            "<h3>信号周期资本利得判断统计</h3>",
            counts_html,
            "<h3>资本利得亏损最大的信号周期</h3>",
            drag_display.to_html(index=False, escape=False),
        ]
    )


def _weights_rows(weights: dict[str, object]) -> str:
    items = [
        ("供给：未来一周发行量", "supply_amount", "发行压力低利多，高利空"),
        ("供给：地方债发行占比", "supply_ratio", "地方债相对供给低利多，高利空"),
        ("供给：10Y以上发行量", "supply_long", "长端供给低利多，高利空"),
        ("供给：发飞（停用）", "fly_penalty", "历史覆盖不足，当前固定为0，不参与回测"),
        ("银行需求", "bank_demand", "银行净买入强利多，弱利空"),
        ("地方债-国债利差", "spread_gov", "利差高代表补偿较厚，利多；低则利空"),
        ("利差一周变化", "spread_change", "利差明显收窄利多，明显走阔利空"),
        ("地方债-NCD利差", "spread_ncd", "相对资金/存单收益补偿高利多，低利空"),
        ("非银情绪", "nonbank_sentiment", "纯债基金净申购利多，净赎回利空"),
    ]
    rows = []
    for label, key, desc in items:
        value = weights.get(key, "")
        rows.append(f"<tr><td>{label}</td><td>{_format_metric(value)}</td><td>{desc}</td></tr>")
    return "\n".join(rows)


def _chart_base64(daily: pd.DataFrame) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(daily["date"], daily["strategy_nav"], label="Strategy NAV", linewidth=2)
    ax.plot(daily["date"], daily["benchmark_nav_rebased"], label="Benchmark NAV", linewidth=2)
    ax.plot(daily["date"], daily["excess_nav"], label="Excess NAV", linewidth=1.6)
    ax.set_title("Dashboard Signal V1 NAV")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _benchmark_chart_base64(frame: pd.DataFrame) -> str:
    temp = frame.copy()
    temp["strategy_nav"] = temp["benchmark_nav"]
    temp["benchmark_nav_rebased"] = temp["benchmark_nav"]
    temp["excess_nav"] = 1.0
    return _chart_base64(temp)


def _write_strategy_html_report(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
    path: Path,
    diagnostics: pd.DataFrame,
) -> None:
    charts_html = _strategy_charts_html(daily)
    latest = signals.iloc[-1]
    latest_factor_cols = [c for c in signals.columns if c.endswith("_定性")]
    factor_rows = "\n".join(
        f"<tr><td>{html_lib.escape(col.removesuffix('_定性'))}</td><td>{html_lib.escape(str(latest[col]))}</td><td>{_format_metric(latest.get(col.replace('_定性', '_得分'), ''))}</td></tr>"
        for col in latest_factor_cols
    )
    policy = signals.attrs.get("position_policy", {})
    bullish_threshold = policy.get("bullish_threshold", 70.0)
    bearish_threshold = policy.get("bearish_threshold", 30.0)
    bullish_position = policy.get("bullish_position", 1.0)
    neutral_position = policy.get("neutral_position", 0.5)
    bearish_position = policy.get("bearish_position", -1.0)
    take_profit_bp = float(policy.get("take_profit_bp", 0.0) or 0.0)
    stop_loss_bp = float(policy.get("stop_loss_bp", 0.0) or 0.0)
    stop_rule_text = (
        f"止盈 {take_profit_bp:g}BP" if take_profit_bp > 0 else "止盈停用"
    ) + "；" + (
        f"止损 {stop_loss_bp:g}BP" if stop_loss_bp > 0 else "止损停用"
    )
    weights_rows = _weights_rows(signals.attrs.get("weights", {}))
    benchmark_name = str(benchmark_metrics.get("benchmark_name", "10Y地方政府债"))
    signal_frequency = str(signals.attrs.get("signal_frequency", "weekly"))
    frequency_name = "日频" if signal_frequency == "daily" else "周频"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Dashboard Signal V1 回测报告</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ margin-bottom: 10px; }}
    h3 {{ margin-top: 22px; margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; }}
    th {{ background: #f0f4f8; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; }}
    .note {{ color: #52606d; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 16px 0 22px; }}
    .card {{ border: 1px solid #d8dee9; padding: 12px 14px; background: #f8fafc; }}
    .card .label {{ color: #52606d; font-size: 13px; }}
    .card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .section {{ margin-top: 28px; }}
    ul {{ margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Dashboard Signal V1 回测报告</h1>
  <p class="note">本报告用于复盘规则版 v1 的策略逻辑、收益口径、信号状态和回测表现；不需要再打开代码或中间 CSV 才能理解本轮结果。</p>

  <div class="grid">
    <div class="card"><div class="label">区间</div><div class="value">{strategy_metrics['start_date']} 至 {strategy_metrics['end_date']}</div></div>
    <div class="card"><div class="label">累计资本利得</div><div class="value">{strategy_metrics['capital_gain_total_bp']:.2f} BP</div></div>
    <div class="card"><div class="label">已平仓交易胜率</div><div class="value">{_format_metric(strategy_metrics['capital_gain_trade_win_rate'], True) or '暂无已平仓'}</div></div>
    <div class="card"><div class="label">平均每笔盈利 / 平均每笔亏损</div><div class="value">{_format_metric(strategy_metrics['capital_gain_avg_win_bp']) or '-'} / {_format_metric(strategy_metrics['capital_gain_avg_loss_bp']) or '-'} BP</div></div>
  </div>

  <h2>策略设计</h2>
  <ul>
    <li>策略目标：以交易盘视角检验地方债看板对10Y地方债资本利得的择时能力。</li>
    <li>交易定义：仓位从0变为非0时开仓，回到0时平仓；多空反向视为先平旧仓再开新仓，同方向加减仓不拆分交易。</li>
    <li>信号频率：{frequency_name}。{'每个交易日更新信号和目标仓位，利差变化采用5个交易日口径。' if signal_frequency == 'daily' else '每周更新看板信号和目标仓位。'}</li>
    <li>交易标的：用 <code>地方政府债到期收益率:10年</code> 构造的 10Y 地方债合成总收益。</li>
    <li>条件基准：{html_lib.escape(benchmark_name)}。策略仓位为正时，基准按相同仓位买入10Y国债；仓位为0或负时，基准持有现金，资本利得为0。</li>
    <li>仓位规则：总分 ≥ {bullish_threshold:g} 看多，仓位 {bullish_position:g}；{bearish_threshold:g} 至 {bullish_threshold:g} 中性，仓位 {neutral_position:g}；低于 {bearish_threshold:g} 看空，仓位 {bearish_position:g}。</li>
    <li>止盈止损：{stop_rule_text}。触发日资本利得计入该笔交易，随后清仓；同方向信号持续禁开，直到信号先转为中性或反向。</li>
    <li>当前版本定位：规则版 v1，重点是可解释和可调整，不做参数优化。</li>
  </ul>

  <h2>资本利得交易绩效</h2>
  <p>资本利得BP按每日 <code>-仓位 × YTM变化_BP</code> 累加，不乘久期。多头超额为同仓位地方债资本利得减同仓位国债资本利得；空仓或空头的比较基准为现金，因此空头超额等于做空地方债的资本利得。胜率只统计已平仓交易。</p>
  <table><thead><tr><th>指标</th><th>策略</th><th>基准</th></tr></thead><tbody>{_capital_trade_metric_table(strategy_metrics, benchmark_metrics)}</tbody></table>

  <h2>交互图表</h2>
  {charts_html}

  <h2>传统收益率绩效（辅助）</h2>
  <table>
    <thead><tr><th>指标</th><th>策略</th><th>基准</th></tr></thead>
    <tbody>{_metric_table(strategy_metrics, benchmark_metrics)}</tbody>
  </table>

  <h2>收益归因：资本利得 vs carry</h2>
  <p>本项目以资本利得为主，carry仅作为辅助解释。传统总收益也使用同一条件基准：多头同仓位持有国债，非多头持有现金。</p>
  <table>
    <thead><tr><th>项目</th><th>区间累计贡献</th><th>解释</th></tr></thead>
    <tbody>{_attribution_table(daily, benchmark_name)}</tbody>
  </table>

  <h2>逐笔交易诊断</h2>
  <p>逐笔交易按开仓到平仓拆分；<code>period_diagnostics.csv</code> 按当前信号频率定位错判周期，两者口径不同。</p>
  {_diagnostics_html(diagnostics)}

  <h2>信号与权重</h2>
  <p>看板因子先判断利多、利空或中性，再按权重折算为总分。发飞字段目前仅有2026年少数周的有效观测，因此本轮固定为0，不参与评分或供给模块利空判断。</p>
  <table>
    <thead><tr><th>模块/因子</th><th>权重</th><th>判断逻辑</th></tr></thead>
    <tbody>{weights_rows}</tbody>
  </table>

  <h2>最新信号</h2>
  <p>信号日期：{latest['signal_date'].date()}；总分：{latest['总分']:.2f}；结论：{latest['结论']}；仓位：{latest['仓位']:.2f}</p>
  <table>
    <thead><tr><th>因子</th><th>定性</th><th>得分</th></tr></thead>
    <tbody>{factor_rows}</tbody>
  </table>

  <h2>收益口径说明</h2>
  <p>当前回测不是具体个券的票息/净价/全价回测，而是 10Y 地方债收益率曲线方向回测。资本利得BP直接使用 <code>-仓位 × YTM变化_BP</code>；传统净值为了保留价格收益参考，仍使用 carry 与 -修正久期 × 收益率变化近似。</p>
  <p>这个口径适合检验“看板是否能择时 10Y 地方债久期敞口”，但不能等同于某一只地方债的真实持有收益。后续如果接入地方债指数净值、真实久期或个券全价数据，可以替换当前收益合成模块。</p>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _strategy_charts_html(daily: pd.DataFrame) -> str:
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Bar, Line
    except Exception:
        chart = _chart_base64(daily)
        return f'<img src="data:image/png;base64,{chart}" alt="nav chart">' if chart else "<p>当前 Python 环境未安装 pyecharts 或 matplotlib，未生成图表。</p>"

    dates = daily["date"].dt.strftime("%Y-%m-%d").tolist()
    strategy_nav = [round(float(x), 6) for x in daily["strategy_nav"]]
    benchmark_nav = [round(float(x), 6) for x in daily["benchmark_nav_rebased"]]
    excess_nav = [round(float(x), 6) for x in daily["excess_nav"]]
    position = [round(float(x), 4) for x in daily["仓位"]]
    score = [round(float(x), 2) for x in daily["总分"]]
    strategy_carry_cum = [round(float(x) * 100.0, 4) for x in daily.get("strategy_carry_cum", pd.Series(0.0, index=daily.index))]
    strategy_capital_cum = [round(float(x) * 100.0, 4) for x in daily.get("strategy_capital_cum", pd.Series(0.0, index=daily.index))]
    benchmark_carry_cum = [round(float(x) * 100.0, 4) for x in daily.get("benchmark_carry_cum", pd.Series(0.0, index=daily.index))]
    benchmark_capital_cum = [round(float(x) * 100.0, 4) for x in daily.get("benchmark_capital_cum", pd.Series(0.0, index=daily.index))]
    carry_excess_cum = [round(float(x) * 100.0, 4) for x in daily.get("carry_excess_cum", pd.Series(0.0, index=daily.index))]
    capital_excess_cum = [round(float(x) * 100.0, 4) for x in daily.get("capital_excess_cum", pd.Series(0.0, index=daily.index))]
    strategy_capital_cum_bp = [round(float(x), 3) for x in daily.get("strategy_capital_cum_bp", pd.Series(0.0, index=daily.index))]
    benchmark_capital_cum_bp = [round(float(x), 3) for x in daily.get("benchmark_capital_cum_bp", pd.Series(0.0, index=daily.index))]
    capital_excess_cum_bp = [round(float(x), 3) for x in daily.get("capital_excess_cum_bp", pd.Series(0.0, index=daily.index))]

    capital_bp_line = (
        Line(init_opts=opts.InitOpts(width="100%", height="500px"))
        .add_xaxis(dates)
        .add_yaxis("策略累计资本利得", strategy_capital_cum_bp, is_symbol_show=False, color="#bb654f")
        .add_yaxis("基准累计资本利得", benchmark_capital_cum_bp, is_symbol_show=False, color="#176b5b")
        .add_yaxis("资本利得超额", capital_excess_cum_bp, is_symbol_show=False, color="#9a7b38", linestyle_opts=opts.LineStyleOpts(type_="dashed"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="收益率资本利得交易曲线（BP，不乘久期）"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts(type_="slider", pos_bottom="2%")],
            yaxis_opts=opts.AxisOpts(name="累计BP", type_="value", is_scale=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )

    trades = capital_gain_trade_table(daily)
    trade_dates = pd.to_datetime(trades["entry_date"]).dt.strftime("%Y-%m-%d").tolist()
    trade_bp = trades["strategy_capital_bp"]
    positive_trade_bp = [round(float(value), 3) if value > 0 else None for value in trade_bp]
    negative_trade_bp = [round(float(value), 3) if value <= 0 else None for value in trade_bp]
    trade_bar = (
        Bar(init_opts=opts.InitOpts(width="100%", height="430px"))
        .add_xaxis(trade_dates)
        .add_yaxis("盈利交易", positive_trade_bp, color="#bb654f", stack="trade")
        .add_yaxis("亏损/持平交易", negative_trade_bp, color="#176b5b", stack="trade")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="开平仓逐笔资本利得"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts(type_="slider", pos_bottom="2%")],
            yaxis_opts=opts.AxisOpts(name="单笔BP", type_="value", is_scale=True),
            xaxis_opts=opts.AxisOpts(type_="category"),
        )
    )

    nav_line = (
        Line(init_opts=opts.InitOpts(width="100%", height="520px"))
        .add_xaxis(dates)
        .add_yaxis("策略净值", strategy_nav, is_symbol_show=False, is_smooth=False)
        .add_yaxis("基准净值", benchmark_nav, is_symbol_show=False, is_smooth=False)
        .add_yaxis("超额净值", excess_nav, is_symbol_show=False, is_smooth=False)
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="仓位",
                type_="value",
                min_=-1.1,
                max_=1.1,
                position="right",
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
            )
        )
        .add_yaxis(
            "仓位",
            position,
            yaxis_index=1,
            is_step=True,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=1.5, type_="dashed"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="策略/基准/超额净值与仓位"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside"),
                opts.DataZoomOpts(type_="slider", pos_bottom="2%"),
            ],
            yaxis_opts=opts.AxisOpts(name="净值", type_="value", is_scale=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )

    score_line = (
        Line(init_opts=opts.InitOpts(width="100%", height="420px"))
        .add_xaxis(dates)
        .add_yaxis("总分", score, is_symbol_show=False)
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="仓位",
                type_="value",
                min_=-1.1,
                max_=1.1,
                position="right",
            )
        )
        .add_yaxis(
            "仓位",
            position,
            yaxis_index=1,
            is_step=True,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=1.5, type_="dashed"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="看板总分与仓位"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside"),
                opts.DataZoomOpts(type_="slider", pos_bottom="2%"),
            ],
            yaxis_opts=opts.AxisOpts(name="总分", min_=0, max_=100),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )

    attribution_line = (
        Line(init_opts=opts.InitOpts(width="100%", height="500px"))
        .add_xaxis(dates)
        .add_yaxis("策略累计票息Carry", strategy_carry_cum, is_symbol_show=False)
        .add_yaxis("策略累计资本利得", strategy_capital_cum, is_symbol_show=False)
        .add_yaxis("基准累计票息Carry", benchmark_carry_cum, is_symbol_show=False)
        .add_yaxis("基准累计资本利得", benchmark_capital_cum, is_symbol_show=False)
        .add_yaxis("票息Carry超额", carry_excess_cum, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(type_="dashed"))
        .add_yaxis("资本利得超额", capital_excess_cum, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(type_="dashed"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="收益归因：票息 Carry 与资本利得累计贡献"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside"),
                opts.DataZoomOpts(type_="slider", pos_bottom="2%"),
            ],
            yaxis_opts=opts.AxisOpts(name="累计收益贡献(%)", type_="value", is_scale=True, axislabel_opts=opts.LabelOpts(formatter="{value}%")),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )

    weekly = daily.drop_duplicates(subset=["signal_date"], keep="first").copy()
    signal_dates = weekly["signal_date"].dt.strftime("%Y-%m-%d").tolist()
    signal_positions = [round(float(x), 4) for x in weekly["仓位"]]
    position_bar = (
        Bar(init_opts=opts.InitOpts(width="100%", height="360px"))
        .add_xaxis(signal_dates)
        .add_yaxis("信号仓位", signal_positions)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="信号仓位变化"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside"),
                opts.DataZoomOpts(type_="slider", pos_bottom="2%"),
            ],
            yaxis_opts=opts.AxisOpts(name="仓位", min_=-1.1, max_=1.1),
        )
    )

    return "\n".join(
        [
            '<div class="section">',
            capital_bp_line.render_embed(),
            "</div>",
            '<div class="section">',
            trade_bar.render_embed(),
            "</div>",
            '<div class="section">',
            nav_line.render_embed(),
            "</div>",
            '<div class="section">',
            score_line.render_embed(),
            "</div>",
            '<div class="section">',
            attribution_line.render_embed(),
            "</div>",
            '<div class="section">',
            position_bar.render_embed(),
            "</div>",
        ]
    )


def _write_benchmark_html_report(frame: pd.DataFrame, metrics: dict[str, object], path: Path) -> None:
    charts_html = _benchmark_charts_html(frame)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>10Y地方债长期持有基准报告</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ margin-bottom: 10px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; }}
    th {{ background: #f0f4f8; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; }}
    .note {{ color: #52606d; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 16px 0 22px; }}
    .card {{ border: 1px solid #d8dee9; padding: 12px 14px; background: #f8fafc; }}
    .card .label {{ color: #52606d; font-size: 13px; }}
    .card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .section {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>10Y地方债长期持有基准报告</h1>
  <p class="note">本报告用于展示 100% 长期持有 10Y 地方债久期敞口的基准表现，并拆分票息 carry 与资本利得。</p>
  <div class="grid">
    <div class="card"><div class="label">区间</div><div class="value">{metrics['start_date']} 至 {metrics['end_date']}</div></div>
    <div class="card"><div class="label">累计收益</div><div class="value">{_format_metric(metrics['total_return'], True)}</div></div>
    <div class="card"><div class="label">最大回撤</div><div class="value">{_format_metric(metrics['max_drawdown'], True)}</div></div>
    <div class="card"><div class="label">期末净值</div><div class="value">{_format_metric(metrics['final_nav'])}</div></div>
  </div>
  <h2>交互图表</h2>
  {charts_html}
  <h2>绩效指标</h2>
  <table>
    <thead><tr><th>指标</th><th>数值</th></tr></thead>
    <tbody>{''.join(f'<tr><td>{METRIC_LABELS.get(k, k)}</td><td>{_format_metric(v, k in {"total_return", "annual_return", "risk_free_rate_annual", "excess_annual_return", "annual_volatility", "max_drawdown", "win_rate"})}</td></tr>' for k, v in metrics.items())}</tbody>
  </table>
  <h2>收益口径说明</h2>
  <p>收益合成：上一期收益率 / 252 作为票息 carry，久期资本利得为 -修正久期 × 收益率变化。该基准等价于 100% 长期持有 10Y 地方债收益率曲线敞口。</p>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _benchmark_charts_html(frame: pd.DataFrame) -> str:
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Line
    except Exception:
        chart = _benchmark_chart_base64(frame)
        return f'<img src="data:image/png;base64,{chart}" alt="benchmark nav chart">' if chart else "<p>当前 Python 环境未安装 pyecharts 或 matplotlib，未生成图表。</p>"

    dates = frame["date"].dt.strftime("%Y-%m-%d").tolist()
    benchmark_nav = [round(float(x), 6) for x in frame["benchmark_nav"]]
    yield_pct = [round(float(x), 4) for x in frame["yield_pct"]]
    carry_cum = [round(float(x) * 100.0, 4) for x in frame["carry_return"].fillna(0.0).cumsum()]
    capital_cum = [round(float(x) * 100.0, 4) for x in frame["duration_pnl"].fillna(0.0).cumsum()]
    total_cum = [round(float(x) * 100.0, 4) for x in frame["total_return"].fillna(0.0).cumsum()]

    nav_line = (
        Line(init_opts=opts.InitOpts(width="100%", height="500px"))
        .add_xaxis(dates)
        .add_yaxis("基准净值", benchmark_nav, is_symbol_show=False)
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="到期收益率(%)",
                type_="value",
                position="right",
                is_scale=True,
            )
        )
        .add_yaxis(
            "10Y地方债收益率",
            yield_pct,
            yaxis_index=1,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=1.4, type_="dashed"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="基准净值与10Y地方债收益率"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside"),
                opts.DataZoomOpts(type_="slider", pos_bottom="2%"),
            ],
            yaxis_opts=opts.AxisOpts(name="净值", type_="value", is_scale=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )

    attribution_line = (
        Line(init_opts=opts.InitOpts(width="100%", height="500px"))
        .add_xaxis(dates)
        .add_yaxis("累计票息Carry", carry_cum, is_symbol_show=False)
        .add_yaxis("累计资本利得", capital_cum, is_symbol_show=False)
        .add_yaxis("累计总收益", total_cum, is_symbol_show=False)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="基准收益归因：票息 Carry 与资本利得"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside"),
                opts.DataZoomOpts(type_="slider", pos_bottom="2%"),
            ],
            yaxis_opts=opts.AxisOpts(name="累计收益贡献(%)", type_="value", is_scale=True, axislabel_opts=opts.LabelOpts(formatter="{value}%")),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )

    return "\n".join(
        [
            '<div class="section">',
            nav_line.render_embed(),
            "</div>",
            '<div class="section">',
            attribution_line.render_embed(),
            "</div>",
        ]
    )


def _write_nav_chart(frame: pd.DataFrame, path: Path) -> None:
    data = frame[["date", "benchmark_nav"]].dropna().copy()
    if data.empty:
        path.write_text("<html><body>No data</body></html>", encoding="utf-8")
        return

    width, height, pad = 960, 420, 48
    values = data["benchmark_nav"].astype(float)
    min_v, max_v = values.min(), values.max()
    span = max(max_v - min_v, 1e-9)
    points = []
    for i, value in enumerate(values):
        x = pad + i * (width - 2 * pad) / max(len(values) - 1, 1)
        y = height - pad - (value - min_v) * (height - 2 * pad) / span
        points.append(f"{x:.2f},{y:.2f}")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>10Y地方债长期持有基准净值</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    svg {{ max-width: 100%; height: auto; border: 1px solid #d8dee9; }}
    .meta {{ color: #52606d; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>10Y地方债长期持有基准净值</h1>
  <p class="meta">{data['date'].iloc[0].date()} 至 {data['date'].iloc[-1].date()}，期末净值 {values.iloc[-1]:.4f}</p>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="benchmark nav chart">
    <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#9aa5b1" />
    <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#9aa5b1" />
    <polyline fill="none" stroke="#1f6feb" stroke-width="2.5" points="{' '.join(points)}" />
    <text x="{pad}" y="{pad-14}" font-size="12">{max_v:.4f}</text>
    <text x="{pad}" y="{height-pad+28}" font-size="12">{min_v:.4f}</text>
  </svg>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
