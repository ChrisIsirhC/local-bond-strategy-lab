from __future__ import annotations

from typing import Any

import pandas as pd


MAX_CALENDAR_GAP_DAYS = 14


def long_date_gaps(
    frame: pd.DataFrame,
    date_col: str,
    max_gap_days: int = MAX_CALENDAR_GAP_DAYS,
) -> list[dict[str, Any]]:
    if date_col not in frame.columns:
        raise ValueError(f"数据质量检查缺少日期列: {date_col}")
    dates = pd.to_datetime(frame[date_col], errors="coerce").dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if len(dates) < 2:
        return []
    gaps = dates.diff().dt.days
    rows = []
    for index in gaps[gaps > max_gap_days].index:
        rows.append(
            {
                "前一有效日期": dates.iloc[index - 1].strftime("%Y-%m-%d"),
                "后一有效日期": dates.iloc[index].strftime("%Y-%m-%d"),
                "间隔自然日": int(gaps.iloc[index]),
            }
        )
    return rows


def assert_no_long_date_gaps(
    frame: pd.DataFrame,
    date_col: str,
    source: str,
    max_gap_days: int = MAX_CALENDAR_GAP_DAYS,
) -> None:
    gaps = long_date_gaps(frame, date_col, max_gap_days)
    if not gaps:
        return
    detail = "; ".join(
        f"{item['前一有效日期']} -> {item['后一有效日期']}（{item['间隔自然日']}天）" for item in gaps[:5]
    )
    suffix = f"；另有 {len(gaps) - 5} 处" if len(gaps) > 5 else ""
    raise ValueError(
        f"数据质量检查失败：{source} 存在超过 {max_gap_days} 个自然日的日期空缺：{detail}{suffix}。"
        "已中止更新，禁止使用缺口数据继续生成信号或回测。"
    )
