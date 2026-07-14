from __future__ import annotations

import numpy as np
import pandas as pd


def vectorized_capital_trade_metrics(
    weekly_positions: np.ndarray,
    weekly_capital_bp: np.ndarray,
) -> dict[str, np.ndarray]:
    candidate_count = weekly_positions.shape[0]
    current_direction = np.zeros(candidate_count, dtype=int)
    current_pnl = np.zeros(candidate_count, dtype=float)
    total_count = np.zeros(candidate_count, dtype=int)
    closed_count = np.zeros(candidate_count, dtype=int)
    winning_count = np.zeros(candidate_count, dtype=int)
    losing_count = np.zeros(candidate_count, dtype=int)
    closed_pnl_sum = np.zeros(candidate_count, dtype=float)
    best_trade = np.full(candidate_count, -np.inf)
    worst_trade = np.full(candidate_count, np.inf)

    for period in range(weekly_positions.shape[1]):
        direction = np.sign(weekly_positions[:, period]).astype(int)
        closing = (current_direction != 0) & ((direction == 0) | (direction != current_direction))
        if closing.any():
            pnl = current_pnl[closing]
            closed_count[closing] += 1
            winning_count[closing] += pnl > 0
            losing_count[closing] += pnl < 0
            closed_pnl_sum[closing] += pnl
            best_trade[closing] = np.maximum(best_trade[closing], pnl)
            worst_trade[closing] = np.minimum(worst_trade[closing], pnl)
            current_pnl[closing] = 0.0
            current_direction[closing] = 0

        opening = (direction != 0) & (current_direction == 0)
        total_count[opening] += 1
        current_direction[opening] = direction[opening]
        active = direction != 0
        current_pnl[active] += weekly_capital_bp[active, period]

    open_trade = current_direction != 0
    win_rate = np.divide(
        winning_count,
        closed_count,
        out=np.full(candidate_count, np.nan),
        where=closed_count != 0,
    )
    average_trade = np.divide(
        closed_pnl_sum,
        closed_count,
        out=np.full(candidate_count, np.nan),
        where=closed_count != 0,
    )
    best_trade[closed_count == 0] = np.nan
    worst_trade[closed_count == 0] = np.nan
    return {
        "trade_count": total_count,
        "closed_trade_count": closed_count,
        "winning_trade_count": winning_count,
        "losing_trade_count": losing_count,
        "trade_win_rate": win_rate,
        "average_trade_bp": average_trade,
        "best_trade_bp": best_trade,
        "worst_trade_bp": worst_trade,
        "open_trade_count": open_trade.astype(int),
        "open_trade_bp": np.where(open_trade, current_pnl, 0.0),
    }


def capital_gain_trade_metrics(
    daily: pd.DataFrame,
    capital_return_col: str,
    signal_col: str = "signal_date",
    position_col: str | None = "仓位",
) -> dict[str, object]:
    capital_bp = pd.to_numeric(daily[capital_return_col], errors="coerce").fillna(0.0) * 10000.0
    trades = capital_gain_trade_table(
        daily,
        strategy_col=capital_return_col,
        benchmark_col=capital_return_col,
        position_col=position_col,
    )
    closed = trades.loc[trades["is_closed"]].copy()
    trade_bp = closed["strategy_capital_bp"]
    winning = trade_bp[trade_bp > 0]
    losing = trade_bp[trade_bp < 0]
    cumulative = capital_bp.cumsum()
    drawdown = cumulative - cumulative.cummax()

    average_win = float(winning.mean()) if not winning.empty else None
    average_loss = float(losing.mean()) if not losing.empty else None
    profit_loss_ratio = (
        average_win / abs(average_loss)
        if average_win is not None and average_loss is not None and average_loss != 0
        else None
    )
    return {
        "capital_gain_total_bp": float(capital_bp.sum()),
        "capital_gain_trade_count": int(len(trades)),
        "capital_gain_closed_trade_count": int(len(trade_bp)),
        "capital_gain_winning_trades": int(len(winning)),
        "capital_gain_losing_trades": int(len(losing)),
        "capital_gain_flat_trades": int((trade_bp == 0).sum()),
        "capital_gain_trade_win_rate": float((trade_bp > 0).mean()) if len(trade_bp) else None,
        "capital_gain_avg_trade_bp": float(trade_bp.mean()) if len(trade_bp) else None,
        "capital_gain_median_trade_bp": float(trade_bp.median()) if len(trade_bp) else None,
        "capital_gain_avg_win_bp": average_win,
        "capital_gain_avg_loss_bp": average_loss,
        "capital_gain_profit_loss_ratio": profit_loss_ratio,
        "capital_gain_best_trade_bp": float(trade_bp.max()) if len(trade_bp) else None,
        "capital_gain_worst_trade_bp": float(trade_bp.min()) if len(trade_bp) else None,
        "capital_gain_max_drawdown_bp": float(drawdown.min()) if len(drawdown) else None,
        "capital_gain_longest_losing_streak": _longest_streak(trade_bp < 0),
        "capital_gain_open_trade_count": int((~trades["is_closed"]).sum()) if not trades.empty else 0,
        "capital_gain_open_trade_bp": float(trades.loc[~trades["is_closed"], "strategy_capital_bp"].sum()) if not trades.empty else 0.0,
    }


def capital_gain_trade_table(
    daily: pd.DataFrame,
    strategy_col: str = "strategy_capital_return",
    benchmark_col: str = "benchmark_capital_return",
    signal_col: str = "signal_date",
    position_col: str | None = "仓位",
) -> pd.DataFrame:
    frame = daily.sort_values("date").reset_index(drop=True).copy()
    position = (
        pd.to_numeric(frame[position_col], errors="coerce").fillna(0.0)
        if position_col is not None
        else pd.Series(1.0, index=frame.index)
    )
    active = position.ne(0.0)
    direction = position.apply(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
    previous_direction = direction.shift(1, fill_value=0)
    starts = active & ((previous_direction == 0) | (direction != previous_direction))
    trade_id = starts.cumsum().where(active)

    rows = []
    for identifier, group in frame.loc[active].groupby(trade_id.loc[active], sort=True):
        start_index = int(group.index.min())
        end_index = int(group.index.max())
        next_position = float(position.iloc[end_index + 1]) if end_index + 1 < len(position) else None
        trade_direction = int(direction.iloc[start_index])
        is_closed = next_position is not None and (next_position == 0 or (1 if next_position > 0 else -1) != trade_direction)
        strategy_bp = float(pd.to_numeric(group[strategy_col], errors="coerce").fillna(0.0).sum() * 10000.0)
        benchmark_bp = float(pd.to_numeric(group[benchmark_col], errors="coerce").fillna(0.0).sum() * 10000.0)
        rows.append(
            {
                "trade_id": int(identifier),
                "entry_date": pd.Timestamp(group["date"].iloc[0]),
                "exit_date": pd.Timestamp(group["date"].iloc[-1]) if is_closed else pd.NaT,
                "mark_date": pd.Timestamp(group["date"].iloc[-1]),
                "direction": "多头" if trade_direction > 0 else "空头",
                "entry_position": float(position.iloc[start_index]),
                "average_abs_position": float(position.loc[group.index].abs().mean()),
                "holding_days": int(len(group)),
                "strategy_capital_bp": strategy_bp,
                "benchmark_capital_bp": benchmark_bp,
                "capital_excess_bp": strategy_bp - benchmark_bp,
                "is_win": strategy_bp > 0,
                "is_closed": bool(is_closed),
                "status": "已平仓" if is_closed else "未平仓",
            }
        )
    columns = [
        "trade_id", "entry_date", "exit_date", "mark_date", "direction", "entry_position",
        "average_abs_position", "holding_days", "strategy_capital_bp", "benchmark_capital_bp",
        "capital_excess_bp", "is_win", "is_closed", "status",
    ]
    return pd.DataFrame(rows, columns=columns)


def _longest_streak(mask: pd.Series) -> int:
    longest = 0
    current = 0
    for value in mask.fillna(False).astype(bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)
