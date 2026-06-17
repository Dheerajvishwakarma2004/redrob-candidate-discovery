"""
Archetype Classifier.
Feature score only. No multipliers.
archetype_score feeds into weighted scoring at 0.02 weight.
"""

import pandas as pd
from typing import Dict, Tuple
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

A_RET = "retrieval_recommendation"
A_ML  = "ml_engineer"
A_DS  = "data_software"
A_OTH = "other"


class ArchetypeClassifier:

    def __init__(self):
        self._scores: Dict[str, float] = config.get("archetype", "scores", default={
            A_RET: 1.0, A_ML: 0.80, A_DS: 0.60, A_OTH: 0.30
        })
        t = config.get("archetype", "thresholds", default={})
        self._ret_skill = t.get("retrieval_skill_min", 0.35)
        self._rank_skill = t.get("ranking_skill_min", 0.35)
        self._rec_skill = t.get("recommendation_skill_min", 0.35)
        self._career_ev = t.get("career_evidence_min", 0.28)
        self._ml_min = t.get("ml_score_min", 0.40)
        self._ml_title = t.get("title_for_ml", 0.65)
        self._ds_title = t.get("title_for_ds", 0.45)

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Archetype classification: {len(df):,} candidates")
        df = df.copy()
        results = df.apply(self._classify_row, axis=1)
        df["archetype"] = results.apply(lambda x: x[0])
        df["archetype_score"] = results.apply(lambda x: x[1])
        dist = df["archetype"].value_counts()
        for k, v in dist.items():
            logger.info(f"  {k:30s}: {v:5,} ({v/len(df)*100:.1f}%)")
        return df

    def _classify_row(self, row: pd.Series) -> Tuple[str, float]:
        ret = float(row.get("retrieval_score", 0.0))
        rank = float(row.get("ranking_score", 0.0))
        rec = float(row.get("recommendation_score", 0.0))
        ml = float(row.get("ml_engineering_score", 0.0))
        title = float(row.get("title_relevance", 0.0))
        career_ev = max(
            float(row.get("career_description_relevance", 0.0)),
            float(row.get("career_jd_semantic_similarity", 0.0))
        )
        if (max(ret, rank, rec) >= max(self._ret_skill, self._rank_skill, self._rec_skill)
                and career_ev >= self._career_ev):
            return A_RET, self._scores.get(A_RET, 1.0)
        if ml >= self._ml_min and title >= self._ml_title:
            return A_ML, self._scores.get(A_ML, 0.80)
        if title >= self._ds_title:
            return A_DS, self._scores.get(A_DS, 0.60)
        return A_OTH, self._scores.get(A_OTH, 0.30)