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
                            candidates.append(
                                DashboardWeights(
                                    supply_amount=float(supply_amount),
                                    supply_ratio=float(supply_ratio),
                                    supply_long=float(supply_long),
                                    fly_penalty=0.0,
                                    bank_demand=float(bank_demand),
                                    spread_gov=float(spread_gov),
                                    spread_change=float(spread_change),
                                    spread_ncd=float(spread_ncd),
                                    nonbank_sentiment=float(nonbank_sentiment),
                                )
                            )

    return candidates


def generate_weight_candidates_v2(module_max: int = 50) -> list[DashboardWeights]:
    """Search module budgets without minimum weights, in 5-point steps.

    Supply and valuation budgets are subsequently allocated across their three
    sub-factors. Demand and nonbank each have a single sub-factor. All module
    budgets may be zero, while the complete score budget remains 100.
    """
    step = 5
    if module_max <= 0 or module_max % step:
        raise ValueError("module_max must be a positive multiple of 5")
    module_values = range(0, module_max + 1, step)
    split_values = range(0, module_max + 1, step)
    candidates: list[DashboardWeights] = []

    for supply_total in module_values:
        supply_splits = [values for values in product(split_values, repeat=3) if sum(values) == supply_total]
        for bank_demand in module_values:
            for spread_total in module_values:
                nonbank_sentiment = 100 - supply_total - bank_demand - spread_total
                if nonbank_sentiment not in module_values:
                    continue
                spread_splits = [values for values in product(split_values, repeat=3) if sum(values) == spread_total]
                for supply_amount, supply_ratio, supply_long in supply_splits:
                    for spread_gov, spread_change, spread_ncd in spread_splits:
                        candidates.append(
                            DashboardWeights(
                                supply_amount=float(supply_amount),
                                supply_ratio=float(supply_ratio),
                                supply_long=float(supply_long),
                                fly_penalty=0.0,
                                bank_demand=float(bank_demand),
                                spread_gov=float(spread_gov),
                                spread_change=float(spread_change),
                                spread_ncd=float(spread_ncd),
                                nonbank_sentiment=float(nonbank_sentiment),
                            )
                        )
    return candidates
