from __future__ import annotations

import numpy as np
import pandas as pd


def apply_capital_stop_rules(
    target_positions: np.ndarray,
    capital_bp_per_unit: np.ndarray,
    signal_ids: np.ndarray,
    take_profit_bp: float | np.ndarray = 0.0,
    stop_loss_bp: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(target_positions, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    unit_bp = np.asarray(capital_bp_per_unit, dtype=float)
    if unit_bp.ndim != 1 or unit_bp.shape[0] != targets.shape[1]:
        raise ValueError("止盈止损的BP序列长度与仓位路径不一致")
    signals = np.asarray(signal_ids)
    if signals.ndim != 1 or signals.shape[0] != targets.shape[1]:
        raise ValueError("止盈止损的信号序列长度与仓位路径不一致")

    candidate_count, period_count = targets.shape
    take_profit = np.broadcast_to(np.asarray(take_profit_bp, dtype=float).reshape(-1), (candidate_count,))
    stop_loss = np.broadcast_to(np.asarray(stop_loss_bp, dtype=float).reshape(-1), (candidate_count,))
    if np.all(take_profit <= 0.0) and np.all(stop_loss <= 0.0):
        return targets.copy(), np.zeros_like(targets, dtype=np.int8)

    executed = np.zeros_like(targets)
    stop_events = np.zeros_like(targets, dtype=np.int8)
    active_direction = np.zeros(candidate_count, dtype=int)
    trade_pnl = np.zeros(candidate_count, dtype=float)
    blocked_direction = np.zeros(candidate_count, dtype=int)

    for period in range(period_count):
        raw_target = targets[:, period]
        raw_direction = np.sign(raw_target).astype(int)
        direction_changed_since_stop = (blocked_direction != 0) & (raw_direction != blocked_direction)
        blocked_direction[direction_changed_since_stop] = 0
        target = np.where(blocked_direction != 0, 0.0, raw_target)
        direction = np.sign(target).astype(int)
        direction_changed = (active_direction != 0) & (direction != active_direction)
        trade_pnl[direction_changed] = 0.0
        active_direction[direction_changed] = 0
        opening = (direction != 0) & (active_direction == 0)
        active_direction[opening] = direction[opening]

        executed[:, period] = target
        active = direction != 0
        trade_pnl[active] += target[active] * unit_bp[period]
        take_profit_hit = active & (take_profit > 0.0) & (trade_pnl >= take_profit)
        stop_loss_hit = active & (stop_loss > 0.0) & (trade_pnl <= -stop_loss)
        triggered = take_profit_hit | stop_loss_hit
        stop_events[take_profit_hit, period] = 1
        stop_events[stop_loss_hit, period] = -1
        blocked_direction[triggered] = direction[triggered]
        active_direction[triggered] = 0
        trade_pnl[triggered] = 0.0

    return executed, stop_events


def select_executed_weekly_positions(
    weekly_positions: np.ndarray,
    daily_signal_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    signal_ids = np.unique(np.asarray(daily_signal_index, dtype=int))
    if signal_ids.size == 0:
        return weekly_positions[:, :0], signal_ids
    if signal_ids[0] < 0 or signal_ids[-1] >= weekly_positions.shape[1]:
        raise ValueError("日度收益序列引用了不存在的信号周期")
    return weekly_positions[:, signal_ids], signal_ids


def vectorized_capital_trade_metrics(
    weekly_positions: np.ndarray,
    weekly_capital_bp: np.ndarray,
    close_after_period: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    candidate_count = weekly_positions.shape[0]
    current_direction = np.zeros(candidate_count, dtype=int)
    current_pnl = np.zeros(candidate_count, dtype=float)
    total_count = np.zeros(candidate_count, dtype=int)
    closed_count = np.zeros(candidate_count, dtype=int)
    winning_count = np.zeros(candidate_count, dtype=int)
    losing_count = np.zeros(candidate_count, dtype=int)
    closed_pnl_sum = np.zeros(candidate_count, dtype=float)
    winning_pnl_sum = np.zeros(candidate_count, dtype=float)
    best_trade = np.full(candidate_count, -np.inf)
    worst_trade = np.full(candidate_count, np.inf)
    forced_closes = (
        np.asarray(close_after_period, dtype=bool)
        if close_after_period is not None
        else np.zeros_like(weekly_positions, dtype=bool)
    )

    for period in range(weekly_positions.shape[1]):
        direction = np.sign(weekly_positions[:, period]).astype(int)
        closing = (current_direction != 0) & ((direction == 0) | (direction != current_direction))
        if closing.any():
            pnl = current_pnl[closing]
            closed_count[closing] += 1
            winning_count[closing] += pnl > 0
            losing_count[closing] += pnl < 0
            closed_pnl_sum[closing] += pnl
            winning_pnl_sum[closing] += np.where(pnl > 0.0, pnl, 0.0)
            best_trade[closing] = np.maximum(best_trade[closing], pnl)
            worst_trade[closing] = np.minimum(worst_trade[closing], pnl)
            current_pnl[closing] = 0.0
            current_direction[closing] = 0

        opening = (direction != 0) & (current_direction == 0)
        total_count[opening] += 1
        current_direction[opening] = direction[opening]
        active = direction != 0
        current_pnl[active] += weekly_capital_bp[active, period]
        forced = forced_closes[:, period] & (current_direction != 0)
        if forced.any():
            pnl = current_pnl[forced]
            closed_count[forced] += 1
            winning_count[forced] += pnl > 0
            losing_count[forced] += pnl < 0
            closed_pnl_sum[forced] += pnl
            winning_pnl_sum[forced] += np.where(pnl > 0.0, pnl, 0.0)
            best_trade[forced] = np.maximum(best_trade[forced], pnl)
            worst_trade[forced] = np.minimum(worst_trade[forced], pnl)
            current_pnl[forced] = 0.0
            current_direction[forced] = 0

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
    average_win = np.divide(
        winning_pnl_sum,
        winning_count,
        out=np.full(candidate_count, np.nan),
        where=winning_count != 0,
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
        "average_win_bp": average_win,
        "best_trade_bp": best_trade,
        "worst_trade_bp": worst_trade,
        "open_trade_count": open_trade.astype(int),
        "open_trade_bp": np.where(open_trade, current_pnl, 0.0),
    }


def capital_gain_trade_metrics(
    daily: pd.DataFrame,
    capital_bp_col: str,
    signal_col: str = "signal_date",
    position_col: str | None = "仓位",
) -> dict[str, object]:
    frame = daily.sort_values("date").reset_index(drop=True)
    capital_bp = pd.to_numeric(frame[capital_bp_col], errors="coerce").fillna(0.0)
    trades = capital_gain_trade_table(
        frame,
        strategy_col=capital_bp_col,
        benchmark_col=capital_bp_col,
        position_col=position_col,
    )
    closed = trades.loc[trades["is_closed"]].copy()
    trade_bp = closed["strategy_capital_bp"]
    winning = trade_bp[trade_bp > 0]
    losing = trade_bp[trade_bp < 0]
    cumulative = capital_bp.cumsum()
    drawdown = cumulative - cumulative.cummax()
    periods = max(len(frame) - 1, 1)
    annualized_bp = float(capital_bp.sum() * 252.0 / periods)
    drawdown_end_index = int(drawdown.idxmin()) if len(drawdown) else None
    drawdown_start_index = (
        int(cumulative.loc[:drawdown_end_index].idxmax()) if drawdown_end_index is not None else None
    )
    drawdown_start = (
        pd.Timestamp(frame.loc[drawdown_start_index, "date"]).date().isoformat()
        if drawdown_start_index is not None
        else None
    )
    drawdown_end = (
        pd.Timestamp(frame.loc[drawdown_end_index, "date"]).date().isoformat()
        if drawdown_end_index is not None
        else None
    )

    average_win = float(winning.mean()) if not winning.empty else None
    average_loss = float(losing.mean()) if not losing.empty else None
    holding_days = pd.to_numeric(trades["holding_days"], errors="coerce").dropna()
    profit_loss_ratio = (
        average_win / abs(average_loss)
        if average_win is not None and average_loss is not None and average_loss != 0
        else None
    )
    return {
        "capital_gain_total_bp": float(capital_bp.sum()),
        "capital_gain_annualized_bp": annualized_bp,
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
        "capital_gain_avg_holding_days": float(holding_days.mean()) if len(holding_days) else None,
        "capital_gain_max_holding_days": int(holding_days.max()) if len(holding_days) else None,
        "capital_gain_max_drawdown_bp": float(drawdown.min()) if len(drawdown) else None,
        "capital_gain_max_drawdown_start": drawdown_start,
        "capital_gain_max_drawdown_end": drawdown_end,
        "capital_gain_longest_losing_streak": _longest_streak(trade_bp < 0),
        "capital_gain_open_trade_count": int((~trades["is_closed"]).sum()) if not trades.empty else 0,
        "capital_gain_open_trade_bp": float(trades.loc[~trades["is_closed"], "strategy_capital_bp"].sum()) if not trades.empty else 0.0,
    }


def capital_gain_trade_table(
    daily: pd.DataFrame,
    strategy_col: str = "strategy_capital_bp",
    benchmark_col: str = "benchmark_capital_bp",
    signal_col: str = "signal_date",
    position_col: str | None = "仓位",
    close_event_col: str | None = "止盈止损事件",
) -> pd.DataFrame:
    frame = daily.sort_values("date").reset_index(drop=True).copy()
    position = (
        pd.to_numeric(frame[position_col], errors="coerce").fillna(0.0)
        if position_col is not None
        else pd.Series(1.0, index=frame.index)
    )
    active = position.ne(0.0)
    close_event = (
        frame[close_event_col].fillna("").astype(str).ne("")
        if close_event_col is not None and close_event_col in frame.columns
        else pd.Series(False, index=frame.index)
    )
    direction = position.apply(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
    previous_direction = direction.shift(1, fill_value=0)
    starts = active & (
        (previous_direction == 0)
        | (direction != previous_direction)
        | close_event.shift(1, fill_value=False)
    )
    trade_id = starts.cumsum().where(active)

    rows = []
    for identifier, group in frame.loc[active].groupby(trade_id.loc[active], sort=True):
        start_index = int(group.index.min())
        end_index = int(group.index.max())
        next_position = float(position.iloc[end_index + 1]) if end_index + 1 < len(position) else None
        trade_direction = int(direction.iloc[start_index])
        is_closed = bool(close_event.iloc[end_index]) or (
            next_position is not None
            and (next_position == 0 or (1 if next_position > 0 else -1) != trade_direction)
        )
        strategy_bp = float(pd.to_numeric(group[strategy_col], errors="coerce").fillna(0.0).sum())
        benchmark_bp = float(pd.to_numeric(group[benchmark_col], errors="coerce").fillna(0.0).sum())
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
