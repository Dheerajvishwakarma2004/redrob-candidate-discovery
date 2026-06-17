"""
Hybrid Retriever.
Merges BM25, embedding results, title safety net, keyword safety net.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set
from src.utils.config_loader import config
from src.utils.logger import get_logger
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever

logger = get_logger(__name__)


class HybridRetriever:

    def __init__(self, bm25: BM25Retriever, emb: EmbeddingRetriever):
        self.bm25 = bm25
        self.emb = emb
        self._pool_size = config.get("retrieval", "final_pool_size", default=1500)
        self._title_net: List[str] = config.get("retrieval", "title_safety_net", default=[])
        self._keyword_net: List[str] = config.get("retrieval", "keyword_safety_net", default=[])

    def retrieve(self, jd_profile: Dict, all_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Hybrid retrieval starting")

        bm25_res = self.bm25.retrieve(jd_profile)
        emb_res = self.emb.retrieve(jd_profile)
        safety_ids = self._safety_net(all_df)

        merged = self._merge(bm25_res, emb_res, safety_ids)
        final = self._select_pool(merged, safety_ids)
        result = final.merge(all_df, on="candidate_id", how="left")

        logger.info(f"Pool: {len(result):,} candidates")
        return result

    def _safety_net(self, df: pd.DataFrame) -> Set[str]:
        ids = set()
        for term in self._title_net:
            mask = df["title_normalized"].str.contains(term, case=False, na=False)
            ids.update(df.loc[mask, "candidate_id"].tolist())

        for term in self._keyword_net:
            mask = (
                df["skills_normalized"].apply(lambda s: isinstance(s, list) and any(term in sk for sk in s))
                | df["career_descriptions_combined"].str.contains(term, case=False, na=False)
            )
            ids.update(df.loc[mask, "candidate_id"].tolist())

        logger.info(f"Safety net: {len(ids):,} candidates")
        return ids

    def _merge(self, bm25: pd.DataFrame, emb: pd.DataFrame, safety: Set[str]) -> pd.DataFrame:
        merged = bm25.merge(emb, on="candidate_id", how="outer")
        merged["bm25_score_norm"] = merged["bm25_score_norm"].fillna(0.0)
        merged["embedding_similarity"] = merged["embedding_similarity"].fillna(0.0)

        new_ids = safety - set(merged["candidate_id"])
        if new_ids:
            extra = pd.DataFrame({
                "candidate_id": list(new_ids),
                "bm25_score_norm": 0.10,
                "embedding_similarity": 0.10,
            })
            merged = pd.concat([merged, extra], ignore_index=True)

        merged["is_safety_net"] = merged["candidate_id"].isin(safety)
        merged["retrieval_score"] = 0.5 * merged["bm25_score_norm"] + 0.5 * merged["embedding_similarity"]
        return merged.sort_values("retrieval_score", ascending=False).reset_index(drop=True)

    def _select_pool(self, merged: pd.DataFrame, safety: Set[str]) -> pd.DataFrame:
        safe_df = merged[merged["is_safety_net"]].copy()
        rest_df = merged[~merged["is_safety_net"]].copy()
        slots = max(0, self._pool_size - len(safe_df))
        pool = pd.concat([safe_df, rest_df.head(slots)], ignore_index=True)
        return pool.sort_values("retrieval_score", ascending=False).reset_index(drop=True)