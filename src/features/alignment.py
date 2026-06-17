"""
Skill-Career Alignment Feature Engine.
Detects whether claimed skills are supported by career evidence.
Hybrid: exact match + semantic similarity.
Most important single feature for honeypot defense.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional
from src.utils.config_loader import config
from src.utils.logger import get_logger
from src.features.concepts import CONCEPT_BUCKETS
from src.jd.jd_extractor import CONCEPT_DESCRIPTIONS

logger = get_logger(__name__)

BUCKETS = list(CONCEPT_BUCKETS.keys())


class AlignmentFeatureEngine:

    def __init__(
        self,
        jd_profile: Dict,
        career_embeddings: Optional[Dict[str, np.ndarray]] = None
    ):
        self.jd_profile = jd_profile
        self.career_embeddings = career_embeddings or {}
        self._threshold = config.get("alignment", "support_threshold", default=0.28)
        self._exact_w = config.get("alignment", "exact_weight", default=0.35)
        self._sem_w = config.get("alignment", "semantic_weight", default=0.65)
        self._max_hits = config.get("alignment", "max_exact_hits", default=5)
        self._ret_boost = config.get("alignment", "retrieval_boost", default=0.08)
        self._rank_boost = config.get("alignment", "ranking_boost", default=0.08)
        self._bucket_embeddings: Dict[str, np.ndarray] = {}
        if self.career_embeddings:
            self._compute_bucket_embeddings()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Alignment features: {len(df):,} candidates")
        df = df.copy()
        use_sem = bool(self.career_embeddings and self._bucket_embeddings)
        df["skill_career_alignment"] = df.apply(
            lambda row: self._alignment(row, use_sem), axis=1
        )
        logger.info(f"  skill_career_alignment mean={df['skill_career_alignment'].mean():.4f}")
        return df

    def _alignment(self, row: pd.Series, use_sem: bool) -> float:
        skills = row.get("skills_normalized", [])
        career = str(row.get("career_descriptions_combined", "")).lower()
        cid = row.get("candidate_id", "")

        claimed = self._claimed_buckets(skills)
        if not claimed:
            return 0.50

        if use_sem:
            emb = self.career_embeddings.get(cid)
            supported = self._supported_hybrid(claimed, career, emb)
        else:
            supported = self._supported_exact(claimed, career)

        ratio = len(supported) / len(claimed)
        if "retrieval" in supported:
            ratio = min(ratio + self._ret_boost, 1.0)
        if "ranking" in supported:
            ratio = min(ratio + self._rank_boost, 1.0)

        return round(float(np.clip(ratio, 0.0, 1.0)), 4)

    def _claimed_buckets(self, skills: List[str]) -> Set[str]:
        claimed = set()
        skills_set = set(skills)
        for name, terms in CONCEPT_BUCKETS.items():
            if any(t in skills_set for t in terms) or \
               any(any(t in s for s in skills_set) for t in terms):
                claimed.add(name)
        return claimed

    def _supported_exact(self, claimed: Set[str], career: str) -> Set[str]:
        supported = set()
        for b in claimed:
            hits = sum(1 for t in CONCEPT_BUCKETS[b] if t in career)
            if min(hits / self._max_hits, 1.0) >= self._threshold:
                supported.add(b)
        return supported

    def _supported_hybrid(
        self, claimed: Set[str], career: str, emb: Optional[np.ndarray]
    ) -> Set[str]:
        supported = set()
        for b in claimed:
            hits = sum(1 for t in CONCEPT_BUCKETS[b] if t in career)
            exact = min(hits / self._max_hits, 1.0)
            sem = 0.0
            if emb is not None and b in self._bucket_embeddings:
                sem = float(max(np.dot(emb, self._bucket_embeddings[b]), 0.0))
            hybrid = self._exact_w * exact + self._sem_w * sem
            if hybrid >= self._threshold:
                supported.add(b)
        return supported

    def _compute_bucket_embeddings(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            from pathlib import Path
            model_name = config.get("embedding", "model")
            model_path = Path(model_name)
            if model_path.exists() and (model_path / "pytorch_model.bin").exists():
                model = SentenceTransformer(model_name)
            else:
                model = SentenceTransformer("BAAI/bge-small-en-v1.5")
                
            for name, desc in CONCEPT_DESCRIPTIONS.items():
                emb = model.encode(desc, normalize_embeddings=True, show_progress_bar=False)
                self._bucket_embeddings[name] = emb.astype(np.float32)
            logger.info(f"Bucket embeddings computed for {len(self._bucket_embeddings)} buckets")
        except Exception as e:
            logger.warning(f"Bucket embeddings failed: {e}. Using exact matching only.")