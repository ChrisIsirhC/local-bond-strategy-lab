from __future__ import annotations

import math

import pandas as pd


def max_drawdown(nav: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    if nav.empty:
        return math.nan, None, None
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    end = drawdown.idxmin()
    start = nav.loc[:end].idxmax()
    return float(drawdown.loc[end]), start, end


def performance_metrics(
    frame: pd.DataFrame,
    return_col: str = "total_return",
    nav_col: str = "nav",
    date_col: str = "date",
    trading_days_per_year: int = 252,
    risk_free_rate_annual: float = 0.014,
) -> dict[str, object]:
    data = frame.dropna(subset=[date_col, nav_col]).copy()
    if data.empty:
        raise ValueError("performance input is empty")

    returns = pd.to_numeric(data[return_col], errors="coerce").fillna(0.0)
    nav = pd.to_numeric(data[nav_col], errors="coerce")
    nav.index = pd.to_datetime(data[date_col])
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    periods = max(len(data) - 1, 1)
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (trading_days_per_year / periods) - 1.0
    annual_volatility = returns.std(ddof=0) * math.sqrt(trading_days_per_year)
    excess_annual_return = annual_return - risk_free_rate_annual
    sharpe = excess_annual_return / annual_volatility if annual_volatility else math.nan
    mdd, mdd_start, mdd_end = max_drawdown(nav)
    calmar = annual_return / abs(mdd) if mdd and not math.isnan(mdd) else math.nan

    return {
        "start_date": data[date_col].iloc[0].date().isoformat(),
        "end_date": data[date_col].iloc[-1].date().isoformat(),
        "trading_days": int(len(data)),
        "final_nav": float(nav.iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "risk_free_rate_annual": float(risk_free_rate_annual),
        "excess_annual_return": float(excess_annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe) if not math.isnan(sharpe) else None,
        "max_drawdown": float(mdd),
        "max_drawdown_start": mdd_start.date().isoformat() if mdd_start is not None else None,
        "max_drawdown_end": mdd_end.date().isoformat() if mdd_end is not None else None,
        "calmar": float(calmar) if not math.isnan(calmar) else None,
        "win_rate": float((returns > 0).mean()),
    }
