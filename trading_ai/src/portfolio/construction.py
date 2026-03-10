from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioConstructor:
    def __init__(
        self,
        top_n: int = 10,
        gross_leverage: float = 2.0,
        max_sector_weight: float = 1.0,
        use_volatility_targeting: bool = False,
        use_score_weighting: bool = True,
        use_long_short: bool = True,
    ) -> None:
        self.top_n = top_n
        self.gross_leverage = gross_leverage
        self.max_sector_weight = max_sector_weight
        self.use_volatility_targeting = use_volatility_targeting
        self.use_score_weighting = use_score_weighting
        self.use_long_short = use_long_short

    def build_weights(
        self,
        scores: pd.DataFrame,
        sector_by_ticker: dict[str, str] | None = None,
        volatility_snapshot: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        vol_snapshot = pd.DataFrame(columns=["date", "ticker", "volatility_60d"])
        if volatility_snapshot is not None and not volatility_snapshot.empty:
            vol_snapshot = volatility_snapshot.copy()
            if "realized_vol_60d" in vol_snapshot.columns and "volatility_60d" not in vol_snapshot.columns:
                vol_snapshot = vol_snapshot.rename(columns={"realized_vol_60d": "volatility_60d"})
            vol_snapshot = vol_snapshot[["date", "ticker", "volatility_60d"]].drop_duplicates(
                subset=["date", "ticker"],
                keep="last",
            )

        rows = []
        for dt, grp in scores.groupby("date", sort=True):
            frame = grp.copy()
            frame["decile"] = np.ceil(frame["score"].rank(method="first", pct=True) * 10).astype(int).clip(1, 10)
            ranked = frame.sort_values("score", ascending=False).reset_index(drop=True)
            ranked["rank"] = ranked.index + 1
            if ranked.empty:
                continue

            frame = ranked[["date", "ticker", "score", "rank"]].copy()
            if "sector" in ranked.columns:
                frame["sector"] = ranked["sector"]
            elif sector_by_ticker is not None:
                frame["sector"] = frame["ticker"].map(sector_by_ticker)
            if not vol_snapshot.empty:
                day_vol = vol_snapshot[vol_snapshot["date"] == dt][["ticker", "volatility_60d"]]
                frame = frame.merge(day_vol, on="ticker", how="left")
            else:
                frame["volatility_60d"] = np.nan
            frame["weight"] = 0.0
            frame["raw_score"] = frame["score"].astype(float)
            frame["score_weight"] = 0.0
            frame["risk_weight"] = 0.0
            frame["normalized_weight"] = 0.0
            frame["decile"] = ranked["decile"]
            frame["side"] = "flat"

            long_mask = frame["decile"].isin([8, 9])
            short_mask = frame["decile"].isin([1, 2]) if self.use_long_short else pd.Series(False, index=frame.index)
            long_count = int(long_mask.sum())
            short_count = int(short_mask.sum())
            if long_count == 0 and short_count == 0:
                rows.append(frame)
                continue
            if long_count > 0:
                self._assign_book_weights(frame, long_mask, side="long", budget=1.0 if not self.use_long_short else self.gross_leverage / 2.0)
            if short_count > 0:
                self._assign_book_weights(frame, short_mask, side="short", budget=self.gross_leverage / 2.0)
            rows.append(frame)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _assign_book_weights(self, frame: pd.DataFrame, mask: pd.Series, side: str, budget: float) -> None:
        if not bool(mask.any()) or budget <= 0:
            return

        sign = 1.0 if side == "long" else -1.0
        selected = frame.loc[mask].copy()
        normalized, score_weight, risk_weight = self._compute_book_weights(selected, side=side)

        frame.loc[mask, "side"] = side
        frame.loc[mask, "score_weight"] = score_weight.to_numpy()
        frame.loc[mask, "risk_weight"] = risk_weight.to_numpy()
        frame.loc[mask, "normalized_weight"] = sign * normalized.to_numpy()

        executable = normalized.copy()
        if "sector" in frame.columns and self.max_sector_weight < 1.0:
            capped = self._apply_sector_cap(selected.assign(weight=normalized.to_numpy()), target_total=1.0)
            executable = capped["weight"].astype(float)

        frame.loc[mask, "weight"] = sign * executable.to_numpy() * budget

    def _compute_book_weights(
        self,
        selected: pd.DataFrame,
        side: str,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        if selected.empty:
            empty = pd.Series(dtype=float)
            return empty, empty, empty

        if self.use_volatility_targeting:
            vol = pd.to_numeric(selected["volatility_60d"], errors="coerce")
            inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if float(inv_vol.sum()) > 0:
                normalized = inv_vol / float(inv_vol.sum())
            else:
                inv_vol = pd.Series(np.ones(len(selected)), index=selected.index, dtype=float)
                normalized = inv_vol / float(inv_vol.sum())
            score_weight = pd.to_numeric(selected["raw_score"], errors="coerce").fillna(0.0).abs()
            return normalized.astype(float), score_weight.astype(float), inv_vol.astype(float)

        if self.use_score_weighting:
            score_weight = pd.to_numeric(selected["raw_score"], errors="coerce").fillna(0.0)
            if side == "short":
                score_weight = score_weight.max() - score_weight
                positive_floor = score_weight[score_weight > 0].min()
                if pd.notna(positive_floor):
                    score_weight = score_weight + float(positive_floor)
            score_weight = score_weight.clip(lower=0.0)
            score_total = float(score_weight.sum())
            if score_total > 0:
                normalized = score_weight / score_total
            else:
                score_weight = pd.Series(np.ones(len(selected)), index=selected.index, dtype=float)
                normalized = score_weight / float(score_weight.sum())
            risk_weight = pd.Series(np.ones(len(selected)), index=selected.index, dtype=float)
            return normalized.astype(float), score_weight.astype(float), risk_weight

        score_weight = pd.Series(np.ones(len(selected)), index=selected.index, dtype=float)
        normalized = score_weight / float(score_weight.sum())
        risk_weight = pd.Series(np.ones(len(selected)), index=selected.index, dtype=float)
        return normalized.astype(float), score_weight, risk_weight

    def _apply_sector_cap(self, longs: pd.DataFrame, target_total: float = 1.0) -> pd.DataFrame:
        if longs.empty:
            return longs
        if self.max_sector_weight <= 0:
            longs["weight"] = 0.0
            return longs

        out = longs.copy()
        out["sector"] = out["sector"].fillna("UNKNOWN")
        weights = out["weight"].astype(float).copy()
        sectors = out["sector"]
        cap = float(self.max_sector_weight)
        total_target = float(target_total)
        tol = 1e-12

        for _ in range(max(10, 2 * sectors.nunique())):
            sector_weights = weights.groupby(sectors).sum()

            over = sector_weights[sector_weights > cap + tol]
            if over.empty:
                break

            for sector, sector_weight in over.items():
                idx = sectors == sector
                weights.loc[idx] = weights.loc[idx] * (cap / sector_weight)

            sector_weights = weights.groupby(sectors).sum()
            total_weight = float(weights.sum())
            residual = max(0.0, total_target - total_weight)
            if residual <= tol:
                continue

            under = sector_weights[sector_weights < cap - tol]
            if under.empty:
                break

            capacity = (cap - under).clip(lower=0.0)
            capacity_total = float(capacity.sum())
            if capacity_total <= tol:
                break

            to_allocate = min(residual, capacity_total)
            for sector, sector_capacity in capacity.items():
                add = to_allocate * (float(sector_capacity) / capacity_total)
                if add <= tol:
                    continue
                idx = sectors == sector
                sector_sum = float(weights.loc[idx].sum())
                if sector_sum <= tol:
                    weights.loc[idx] = weights.loc[idx] + (add / int(idx.sum()))
                else:
                    weights.loc[idx] = weights.loc[idx] + (weights.loc[idx] / sector_sum) * add

        out["weight"] = weights.clip(lower=0.0)
        return out
