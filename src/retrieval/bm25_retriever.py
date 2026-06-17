"""BM25 Retriever using rank-bm25."""

import re
import numpy as np
import pandas as pd
from typing import Dict, List
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.utils.config_loader import config
from src.utils.logger import get_logger
from src.features.concepts import CONCEPT_BUCKETS

logger = get_logger(__name__)


class BM25Retriever:

    def __init__(self):
        self._index: BM25Okapi = None
        self._ids: List[str] = []
        self._top_k = config.get("retrieval", "bm25_top_k", default=2000)

    def build(self, df: pd.DataFrame) -> None:
        logger.info(f"Building BM25 index: {len(df):,} candidates")
        self._ids = df["candidate_id"].tolist()
        corpus = [self._tokenize(str(t)) for t in tqdm(df["unified_text"], desc="BM25 tokenize")]
        self._index = BM25Okapi(corpus)
        logger.info("BM25 index built")

    def retrieve(self, jd_profile: Dict, top_k: int = None) -> pd.DataFrame:
        if self._index is None:
            raise RuntimeError("Call build() first")
        k = top_k or self._top_k
        query = self._build_query(jd_profile)
        scores = self._index.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_idx]
        mn, mx = top_scores.min(), top_scores.max()
        normed = (top_scores - mn) / (mx - mn + 1e-9)
        return pd.DataFrame({
            "candidate_id": [self._ids[i] for i in top_idx],
            "bm25_score_norm": normed
        })

    def _build_query(self, jd_profile: Dict) -> str:
        parts = []
        for t in jd_profile.get("required_titles", []):
            parts.extend([t, t])
        cw = jd_profile.get("concept_weights", {})
        for bucket, weight in sorted(cw.items(), key=lambda x: -x[1]):
            n = max(3, int(weight * 20))
            parts.extend(CONCEPT_BUCKETS.get(bucket, [])[:n])
        return " ".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]