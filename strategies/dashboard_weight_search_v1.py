from __future__ import annotations

from itertools import product

from strategies.dashboard_signal_v1 import DashboardWeights


def generate_weight_candidates() -> list[DashboardWeights]:
    """Generate v1 dashboard weight candidates in 5-point steps."""
    step = 5
    candidates: list[DashboardWeights] = []
    values_0_20 = range(0, 21, step)
    values_0_25 = range(0, 26, step)

    for supply_total in range(20, 41, step):
        supply_splits = [x for x in product(values_0_20, repeat=3) if sum(x) == supply_total]
        for bank_demand in range(10, 26, step):
            for spread_total in range(25, 46, step):
                spread_splits = [x for x in product(values_0_25, repeat=3) if sum(x) == spread_total]
                for nonbank_sentiment in range(5, 26, step):
                    if supply_total + bank_demand + spread_total + nonbank_sentiment != 100:
                        continue
                    for supply_amount, supply_ratio, supply_long in supply_splits:
                        for spread_gov, spread_change, spread_ncd in spread_splits:
                            for fly_penalty in range(-30, 1, step):
                                candidates.append(
                                    DashboardWeights(
                                        supply_amount=float(supply_amount),
                                        supply_ratio=float(supply_ratio),
                                        supply_long=float(supply_long),
                                        fly_penalty=float(fly_penalty),
                                        bank_demand=float(bank_demand),
                                        spread_gov=float(spread_gov),
                                        spread_change=float(spread_change),
                                        spread_ncd=float(spread_ncd),
                                        nonbank_sentiment=float(nonbank_sentiment),
                                    )
                                )

    return candidates
