"""Experience Feature Engine. Smooth curve scoring."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperienceFeatureEngine:

    def __init__(self, jd_profile: Dict):
        self._exp_min = jd_profile.get("experience_min",
            config.get("experience_curve", default=[[5, 0.7]])[0][0])
        self._exp_max = jd_profile.get("experience_max", 9.0)
        raw = config.get("experience_curve", default=[])
        self._curve: List[Tuple[float, float]] = [(r[0], r[1]) for r in raw] if raw else [
            (2,0.20),(3,0.40),(4,0.55),(5,0.70),(6,0.90),(8,1.00),
            (9,0.90),(11,0.75),(13,0.60),(15,0.50),(999,0.40)
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Experience features: {len(df):,} candidates")
        df = df.copy()
        # Ensure `total_experience` exists; if missing, create as NaN so _score handles it
        if "total_experience" not in df.columns:
            df["total_experience"] = np.nan

        df["experience_fit"] = df["total_experience"].apply(self._score)
        mean_val = df["experience_fit"].mean()
        try:
            logger.info(f"  experience_fit mean={mean_val:.4f}")
        except Exception:
            logger.info(f"  experience_fit mean={mean_val}")
        return df

    def _score(self, years) -> float:
        try:
            years = float(years)
        except (TypeError, ValueError):
            return 0.30
        if years < 0: return 0.30
        for upper, score in self._curve:
            if years <= upper:
                return round(float(score), 4)
        return 0.40