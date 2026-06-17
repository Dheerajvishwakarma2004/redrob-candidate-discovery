"""Education Feature Engine. Weak signal in this dataset (5% weight)."""

import numpy as np
import pandas as pd
from typing import Dict
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

TIER_MAP = {1: 1.0, 2: 0.75, 3: 0.55, 4: 0.40, 0: 0.30}

FIELD_MAP: Dict[str, float] = {
    "computer science": 1.0, "cs": 1.0, "artificial intelligence": 1.0,
    "machine learning": 1.0, "data science": 0.95, "statistics": 0.88,
    "mathematics": 0.85, "information technology": 0.82, "it": 0.82,
    "software engineering": 0.88, "computer engineering": 0.85,
    "electronics": 0.65, "electrical engineering": 0.60, "physics": 0.70,
    "operations research": 0.78, "information systems": 0.80,
    "mechanical engineering": 0.20, "civil engineering": 0.15,
    "commerce": 0.15, "marketing": 0.12, "business administration": 0.30,
    "economics": 0.35, "unknown": 0.30,
}

DEGREE_MAP: Dict[str, float] = {
    "phd": 1.0, "ms": 0.90, "mtech": 0.90, "msc": 0.85, "me": 0.85,
    "mba": 0.50, "btech": 0.70, "be": 0.70, "bsc": 0.65, "bs": 0.65,
    "ba": 0.55, "diploma": 0.45, "unknown": 0.50,
}


class EducationFeatureEngine:

    def __init__(self):
        self._w = config.get("education_weights", default={})

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Education features: {len(df):,} candidates")
        df = df.copy()
        df["education_score"] = df.apply(self._score, axis=1)
        logger.info(f"  education_score mean={df['education_score'].mean():.4f}")
        return df

    def _score(self, row) -> float:
        tier = self._tier(row)
        field = self._field(row)
        degree = self._degree(row)
        w = self._w
        s = (w.get("tier", 0.40) * tier
             + w.get("field", 0.40) * field
             + w.get("degree", 0.20) * degree)
        return round(float(np.clip(s, 0.0, 1.0)), 4)

    def _tier(self, row) -> float:
        try: return TIER_MAP.get(int(row.get("education_tier", 0)), 0.30)
        except: return 0.30

    def _field(self, row) -> float:
        f = str(row.get("education_field", "unknown")).lower()
        if f in FIELD_MAP: return FIELD_MAP[f]
        for k, v in FIELD_MAP.items():
            if k in f: return v
        if any(kw in f for kw in ["computer", "software", "data", "ai"]): return 0.85
        if any(kw in f for kw in ["math", "stat", "physics"]): return 0.75
        if any(kw in f for kw in ["electrical", "electronics"]): return 0.60
        return 0.30

    def _degree(self, row) -> float:
        d = str(row.get("education_degree", "unknown")).lower()
        if d in DEGREE_MAP: return DEGREE_MAP[d]
        if "phd" in d or "doctorate" in d: return 1.0
        if any(k in d for k in ["master", "m.tech", "m.sc", "m.s"]): return 0.88
        if any(k in d for k in ["bachelor", "b.tech", "b.sc", "b.e"]): return 0.68
        return 0.50