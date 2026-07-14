from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DashboardPositionPolicy:
    bullish_threshold: float = 70.0
    bearish_threshold: float = 30.0
    bullish_position: float = 1.0
    neutral_position: float = 0.7
    bearish_position: float = -1.0
    bearish_min_core_factors: int = 0
    bearish_require_supply_or_demand: int = 0
    bearish_confirmation_periods: int = 1

    def conclusion(self, score: float) -> str:
        if score >= self.bullish_threshold:
            return "看多"
        if score < self.bearish_threshold:
            return "看空"
        return "中性"

    def position(self, score: float) -> float:
        if score >= self.bullish_threshold:
            return self.bullish_position
        if score < self.bearish_threshold:
            return self.bearish_position
        return self.neutral_position

    def bearish_mask(
        self,
        scores: np.ndarray,
        bearish_module_count: np.ndarray | None = None,
        supply_or_demand_bearish: np.ndarray | None = None,
    ) -> np.ndarray:
        mask = scores < self.bearish_threshold
        if self.bearish_min_core_factors > 0:
            if bearish_module_count is None:
                mask = np.zeros_like(mask, dtype=bool)
            else:
                mask &= bearish_module_count >= self.bearish_min_core_factors
        if self.bearish_require_supply_or_demand:
            if supply_or_demand_bearish is None:
                mask = np.zeros_like(mask, dtype=bool)
            else:
                mask &= supply_or_demand_bearish.astype(bool)

        periods = max(int(self.bearish_confirmation_periods), 1)
        if periods > 1:
            confirmed = np.zeros_like(mask, dtype=bool)
            if mask.shape[-1] >= periods:
                window = np.ones_like(mask[..., periods - 1 :], dtype=bool)
                for offset in range(periods):
                    window &= mask[..., offset : mask.shape[-1] - periods + offset + 1]
                confirmed[..., periods - 1 :] = window
            mask = confirmed
        return mask

    def vectorized_positions(
        self,
        scores: np.ndarray,
        bearish_module_count: np.ndarray | None = None,
        supply_or_demand_bearish: np.ndarray | None = None,
    ) -> np.ndarray:
        positions = np.where(
            scores >= self.bullish_threshold,
            self.bullish_position,
            self.neutral_position,
        )
        return np.where(
            self.bearish_mask(scores, bearish_module_count, supply_or_demand_bearish),
            self.bearish_position,
            positions,
        )

    def vectorized_conclusions(
        self,
        scores: np.ndarray,
        bearish_module_count: np.ndarray | None = None,
        supply_or_demand_bearish: np.ndarray | None = None,
    ) -> np.ndarray:
        conclusions = np.where(scores >= self.bullish_threshold, "看多", "中性")
        return np.where(
            self.bearish_mask(scores, bearish_module_count, supply_or_demand_bearish),
            "看空",
            conclusions,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "bullish_threshold": self.bullish_threshold,
            "bearish_threshold": self.bearish_threshold,
            "bullish_position": self.bullish_position,
            "neutral_position": self.neutral_position,
            "bearish_position": self.bearish_position,
            "bearish_min_core_factors": self.bearish_min_core_factors,
            "bearish_require_supply_or_demand": self.bearish_require_supply_or_demand,
            "bearish_confirmation_periods": self.bearish_confirmation_periods,
        }


DEFAULT_POSITION_POLICY = DashboardPositionPolicy()
