"""
Technical Feature Engine.
Weights derived from JD concept weights (JD-adaptive).
Config values are fallback defaults.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from src.utils.config_loader import config
from src.utils.logger import get_logger
from src.features.concepts import CONCEPT_BUCKETS, FOUNDATIONAL_SKILLS

logger = get_logger(__name__)

BUCKETS = ["retrieval", "ranking", "recommendation", "ml_engineering", "llm"]


class TechnicalFeatureEngine:

    def __init__(self, jd_profile: Dict):
        self.jd_profile = jd_profile
        jd_w = jd_profile.get("concept_weights", {})
        cfg_w = config.get("technical_weights", default={})
        self._weights = self._resolve(jd_w, cfg_w)
        self._max_hits = config.get("alignment", "max_exact_hits", default=5)
        logger.info("Technical weights:")
        for k, v in sorted(self._weights.items(), key=lambda x: -x[1]):
            logger.info(f"  {k:20s}: {v:.4f}")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Technical features: {len(df):,} candidates")
        df = df.copy()
        for bucket in BUCKETS:
            df[f"{bucket}_score"] = df["skills_normalized"].apply(
                lambda s: self._score_bucket(s, bucket)
            )
        df["advanced_skill_count"] = df.apply(self._count_advanced, axis=1)
        df["foundational_skill_count"] = df["skills_normalized"].apply(self._count_foundational)
        df["technical_score"] = df.apply(self._aggregate, axis=1)
        for feat in ["retrieval_score", "ranking_score", "technical_score"]:
            if feat in df.columns:
                logger.info(f"  {feat:30s} mean={df[feat].mean():.4f}")
        return df

    def _resolve(self, jd_w: Dict, cfg_w: Dict) -> Dict:
        if jd_w and all(b in jd_w for b in BUCKETS):
            weights = {b: jd_w[b] for b in BUCKETS}
        else:
            weights = {b: jd_w.get(b, cfg_w.get(b, 0.2)) for b in BUCKETS}
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    def _score_bucket(self, skills: List[str], bucket: str) -> float:
        # Normalize skills to a Python list to avoid ambiguous truth-value checks
        if skills is None:
            skills = []
        elif isinstance(skills, str):
            skills = [skills] if skills else []
        elif isinstance(skills, np.ndarray):
            try:
                skills = skills.tolist()
            except Exception:
                skills = [str(skills)]
        else:
            try:
                skills = list(skills)
            except Exception:
                skills = [skills]

        if len(skills) == 0:
            return 0.0

        terms = CONCEPT_BUCKETS.get(bucket, [])
        skills_set = set(skills)
        hits = 0.0
        for term in terms:
            if term in skills_set:
                hits += 1.0
            elif any(term in s for s in skills_set):
                hits += 0.5
            elif any(s in term for s in skills_set if len(s) >= 4):
                hits += 0.3
        return round(float(min(hits / self._max_hits, 1.0)), 4)

    def _count_advanced(self, row: pd.Series) -> int:
        skills = row.get("skills_normalized", [])
        if skills is None:
            skills = []
        elif isinstance(skills, str):
            skills = [skills] if skills else []
        elif isinstance(skills, np.ndarray):
            try:
                skills = skills.tolist()
            except Exception:
                skills = [str(skills)]
        else:
            try:
                skills = list(skills)
            except Exception:
                skills = [skills]

        if len(skills) == 0:
            return 0

        skills_set = set(skills)
        count = 0
        for bucket, terms in CONCEPT_BUCKETS.items():
            if any(t in skills_set for t in terms):
                count += 1
        return count

    def _count_foundational(self, skills: List[str]) -> int:
        if skills is None:
            skills = []
        elif isinstance(skills, str):
            skills = [skills] if skills else []
        elif isinstance(skills, np.ndarray):
            try:
                skills = skills.tolist()
            except Exception:
                skills = [str(skills)]
        else:
            try:
                skills = list(skills)
            except Exception:
                skills = [skills]

        if len(skills) == 0:
            return 0
        return len(set(skills) & FOUNDATIONAL_SKILLS)

    def _aggregate(self, row: pd.Series) -> float:
        score = sum(
            self._weights.get(b, 0.0) * float(row.get(f"{b}_score", 0.0))
            for b in BUCKETS
        )
        return round(float(np.clip(score, 0.0, 1.0)), 4)