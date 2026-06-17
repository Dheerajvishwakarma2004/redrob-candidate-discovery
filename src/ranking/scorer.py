"""
Weighted Scoring Engine.
No LightGBM. No labels. Full weighted sum.
All weights from config.yaml.
"""

import numpy as np
import pandas as pd
from typing import Dict
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScoringEngine:

    def __init__(self):
        self._w: Dict[str, float] = config.get("scoring", default={})
        self._top_n = config.get("output", "top_n", default=100)

    def score_and_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Scoring: {len(df):,} candidates")
        df = df.copy()
        self._fill_missing(df)
        df["raw_score"] = df.apply(self._raw_score, axis=1)
        df["final_score"] = (
            df["raw_score"] - df["honeypot_risk_score"].fillna(0.0)
        ).clip(0.0, 1.0).round(6)
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        self._log_summary(df)
        return df

    def top_n(self, df: pd.DataFrame, n: int = None) -> pd.DataFrame:
        return df.head(n or self._top_n).copy()

    def _raw_score(self, row) -> float:
        w = self._w
        s = (
            w.get("career", 0.30) * row.get("career_score", 0.0)
            + w.get("technical", 0.26) * row.get("technical_score", 0.0)
            + w.get("alignment", 0.16) * row.get("skill_career_alignment", 0.0)
            + w.get("recruitability", 0.12) * row.get("recruitability_score", 0.0)
            + w.get("semantic", 0.08) * row.get("semantic_score", 0.0)
            + w.get("experience", 0.05) * row.get("experience_fit", 0.0)
            + w.get("archetype", 0.02) * row.get("archetype_score", 0.0)
            + w.get("education", 0.01) * row.get("education_score", 0.0)
        )
        return round(float(np.clip(s, 0.0, 1.0)), 6)

    def _fill_missing(self, df: pd.DataFrame) -> None:
        required = [
            "career_score", "technical_score", "skill_career_alignment",
            "recruitability_score", "semantic_score", "experience_fit",
            "archetype_score", "education_score", "honeypot_risk_score",
        ]
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}. Defaulting to 0.0")
                df[col] = 0.0

    def _log_summary(self, df: pd.DataFrame) -> None:
        logger.info(f"Score range: {df['final_score'].min():.4f} - {df['final_score'].max():.4f}")
        logger.info(f"Score mean:  {df['final_score'].mean():.4f}")
        preview = ["rank","candidate_id","final_score","archetype",
                   "career_score","technical_score","honeypot_risk_score"]
        cols = [c for c in preview if c in df.columns]
        logger.info("\nTop 10:\n" + df.head(10)[cols].to_string(index=False))