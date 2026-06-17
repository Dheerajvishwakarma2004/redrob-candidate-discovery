"""Semantic Feature Engine. Packages retrieval scores."""

import numpy as np
import pandas as pd
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticFeatureEngine:

    def __init__(self):
        self._bm25_w = config.get("retrieval", "bm25_top_k", default=2000)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Semantic features: {len(df):,} candidates")
        df = df.copy()
        if "bm25_score_norm" not in df.columns:
            df["bm25_score_norm"] = 0.0
        if "embedding_similarity" not in df.columns:
            df["embedding_similarity"] = 0.0
        df["semantic_score"] = (
            0.50 * df["bm25_score_norm"].fillna(0.0)
            + 0.50 * df["embedding_similarity"].fillna(0.0)
        ).clip(0.0, 1.0).round(4)
        logger.info(f"  semantic_score mean={df['semantic_score'].mean():.4f}")
        return df