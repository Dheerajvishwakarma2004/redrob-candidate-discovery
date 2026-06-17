"""
Career Feature Engine.
Computes: title_relevance, consistency, progression, industry_alignment,
          description_relevance, career_jd_semantic_similarity, career_score.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.utils.config_loader import config
from src.utils.logger import get_logger
from src.features.concepts import CONCEPT_BUCKETS, NEGATIVE_CAREER_CONCEPTS

logger = get_logger(__name__)

TITLE_MAP: Dict[str, float] = {
    "search engineer": 1.0, "retrieval engineer": 1.0,
    "recommendation systems engineer": 1.0, "recommendation engineer": 1.0,
    "ranking engineer": 1.0, "relevance engineer": 1.0,
    "search relevance engineer": 1.0,
    "ml engineer": 0.92, "machine learning engineer": 0.92,
    "ai engineer": 0.92, "ai research engineer": 0.92,
    "nlp engineer": 0.92, "applied scientist": 0.90,
    "research engineer": 0.90, "senior ml engineer": 0.92,
    "staff ml engineer": 0.92, "principal ml engineer": 0.92,
    "data scientist": 0.85, "senior data scientist": 0.87,
    "lead data scientist": 0.88, "mlops engineer": 0.82,
    "ml platform engineer": 0.82,
    "software engineer": 0.70, "senior software engineer": 0.75,
    "staff software engineer": 0.78, "principal software engineer": 0.78,
    "backend engineer": 0.70, "senior backend engineer": 0.72,
    "data engineer": 0.72, "senior data engineer": 0.74,
    "platform engineer": 0.65, "junior ml engineer": 0.60,
    "junior data scientist": 0.60, "full stack engineer": 0.55,
    "data analyst": 0.45, "business analyst": 0.30,
    "qa engineer": 0.25, "devops engineer": 0.40,
    "project manager": 0.20, "product manager": 0.30,
    "sales executive": 0.0, "marketing manager": 0.0,
    "hr manager": 0.0, "accountant": 0.0,
    "graphic designer": 0.0, "operations manager": 0.0,
    "mechanical engineer": 0.05, "civil engineer": 0.05,
}

SENIORITY_MAP: Dict[str, int] = {
    "intern": 1, "trainee": 1, "fresher": 1, "junior": 2,
    "associate": 3, "mid": 4, "senior": 5, "lead": 6, "staff": 6,
    "principal": 7, "architect": 8, "manager": 5, "director": 7,
    "vp": 8, "head": 7, "chief": 9,
}

INDUSTRY_MAP: Dict[str, float] = {
    "ai/ml": 1.0, "artificial intelligence": 1.0, "machine learning": 1.0,
    "saas": 0.92, "software": 0.88, "technology": 0.85, "tech": 0.85,
    "fintech": 0.82, "e-commerce": 0.80, "ecommerce": 0.80,
    "edtech": 0.78, "healthtech": 0.78, "adtech": 0.75,
    "data analytics": 0.80, "it services": 0.65, "consulting": 0.62,
    "media": 0.50, "banking": 0.50, "financial services": 0.50,
    "healthcare": 0.45, "retail": 0.40, "education": 0.40,
    "manufacturing": 0.30, "construction": 0.20, "logistics": 0.30,
}

DEFAULT_TITLE = 0.30
DEFAULT_INDUSTRY = 0.35


class CareerFeatureEngine:

    def __init__(
        self,
        jd_profile: Dict,
        career_embeddings: Optional[Dict[str, np.ndarray]] = None
    ):
        self.jd_profile = jd_profile
        self.concept_weights = jd_profile.get("concept_weights", {})
        self.jd_embedding = jd_profile.get("jd_embedding")
        self.career_embeddings = career_embeddings or {}
        self._w = config.get("career_weights", default={})
        self._prog_w = config.get("career_progression_weights",
                                   default={"seniority": 0.5, "domain": 0.5})

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Career features: {len(df):,} candidates")
        df = df.copy()
        df["title_relevance"] = df["title_normalized"].apply(self.score_title)
        df["career_consistency"] = df.apply(self._consistency, axis=1)
        df["career_progression"] = df.apply(self._progression, axis=1)
        df["industry_alignment"] = df.apply(self._industry, axis=1)
        df["career_description_relevance"] = df.apply(self._desc_relevance, axis=1)
        df["career_jd_semantic_similarity"] = df.apply(self._jd_semantic, axis=1)
        df["career_score"] = df.apply(self._aggregate, axis=1)
        self._log_stats(df)
        return df

    def score_title(self, title: str) -> float:
        if title is None:
            return DEFAULT_TITLE
        # Coerce non-string title values to string safely
        try:
            t = str(title)
        except Exception:
            return DEFAULT_TITLE
        t = t.lower().strip()
        if t in TITLE_MAP:
            return TITLE_MAP[t]
        best = None
        for k, v in TITLE_MAP.items():
            if k in t:
                best = v if best is None else max(best, v)
        if best is not None:
            return best
        words = set(t.split())
        kw = {
            "search": 0.90, "retrieval": 0.92, "ranking": 0.90,
            "recommendation": 0.88, "relevance": 0.88, "ml": 0.85,
            "nlp": 0.88, "ai": 0.82, "scientist": 0.75, "research": 0.78,
        }
        scores = [kw[w] for w in words if w in kw]
        if scores:
            return min(max(scores), 1.0)
        return DEFAULT_TITLE

    def _consistency(self, row: pd.Series) -> float:
        titles = row.get("career_titles", [])
        # Normalize titles into a Python list to avoid ambiguous truth-value checks
        if titles is None:
            titles = []
        elif isinstance(titles, str):
            titles = [titles] if titles else []
        elif isinstance(titles, np.ndarray):
            try:
                titles = titles.tolist()
            except Exception:
                titles = [str(titles)]
        else:
            try:
                titles = list(titles)
            except Exception:
                titles = [titles]

        if len(titles) == 0:
            return self.score_title(row.get("title_normalized", "")) * 0.8

        scores = [self.score_title(t) for t in titles]
        n = len(scores)
        weights = list(range(1, n + 1))
        base = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        if all(s >= 0.65 for s in scores):
            base = min(base + 0.08, 1.0)
        if scores and scores[-1] < 0.30:
            base = max(base - 0.12, 0.0)
        neg = sum(1 for t in NEGATIVE_CAREER_CONCEPTS
                  if t in row.get("career_descriptions_combined", ""))
        if neg >= 3:
            base = max(base - 0.10, 0.0)
        return round(float(np.clip(base, 0.0, 1.0)), 4)

    def _progression(self, row: pd.Series) -> float:
        all_titles = row.get("career_titles", []) + [row.get("title_normalized", "")]
        all_titles = [t for t in all_titles if t]
        if len(all_titles) < 2:
            return round(float(np.clip(self.score_title(all_titles[0] if all_titles else "") * 0.7, 0, 1)), 4)
        first, last = all_titles[0], all_titles[-1]
        sen = self._seniority_delta(first, last)
        dom = self._domain_delta(first, last)
        score = 0.5 * sen + 0.5 * dom
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def _seniority_delta(self, first: str, last: str) -> float:
        def level(t):
            tl = t.lower()
            for k, v in sorted(SENIORITY_MAP.items(), key=lambda x: -len(x[0])):
                if k in tl:
                    return v
            return 4
        delta = level(last) - level(first)
        return {True: 1.0}.get(delta >= 3, 0.85 if delta == 2 else
               0.70 if delta == 1 else 0.55 if delta == 0 else 0.20)

    def _domain_delta(self, first: str, last: str) -> float:
        d = self.score_title(last) - self.score_title(first)
        if d > 0.40: return 1.0
        if d > 0.20: return 0.80
        if d > 0.05: return 0.65
        if d >= 0:   return 0.55
        return 0.30

    def _industry(self, row: pd.Series) -> float:
        industries = []
        direct = str(row.get("industry", "")).lower().strip()
        if direct:
            industries.append(direct)
        industries += [str(i).lower() for i in row.get("career_industries", []) if i]
        if not industries:
            return DEFAULT_INDUSTRY
        scored = [self._industry_score(i) for i in industries]
        n = len(scored)
        weights = list(range(1, n + 1))
        return round(float(np.clip(
            sum(s * w for s, w in zip(scored, weights)) / sum(weights), 0, 1
        )), 4)

    def _industry_score(self, industry: str) -> float:
        if industry in INDUSTRY_MAP:
            return INDUSTRY_MAP[industry]
        for k, v in INDUSTRY_MAP.items():
            if k in industry or industry in k:
                return v
        return DEFAULT_INDUSTRY

    def _desc_relevance(self, row: pd.Series) -> float:
        text = row.get("career_descriptions_combined", "")
        if not text:
            return 0.0
        max_hits = config.get("alignment", "max_exact_hits", default=5)
        bucket_scores = {}
        for name, terms in CONCEPT_BUCKETS.items():
            hits = sum(1 for t in terms if t in text)
            bucket_scores[name] = min(hits / max_hits, 1.0)
        total_w = sum(self.concept_weights.values()) or 1.0
        score = sum(
            self.concept_weights.get(b, 0.0) * s
            for b, s in bucket_scores.items()
        ) / total_w
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def _jd_semantic(self, row: pd.Series) -> float:
        if self.jd_embedding is None:
            return 0.0
        emb = self.career_embeddings.get(row.get("candidate_id", ""))
        if emb is None:
            return 0.0
        sim = float(np.dot(self.jd_embedding, emb))
        return round(float(np.clip(sim, 0.0, 1.0)), 4)

    def _aggregate(self, row: pd.Series) -> float:
        w = self._w
        jd_w = 0.08
        existing = 1.0 - jd_w
        score = (
            w.get("title_relevance", 0.30) * existing * row.get("title_relevance", 0.0)
            + w.get("consistency", 0.25) * existing * row.get("career_consistency", 0.0)
            + w.get("description_relevance", 0.20) * existing * row.get("career_description_relevance", 0.0)
            + w.get("industry_alignment", 0.15) * existing * row.get("industry_alignment", 0.0)
            + w.get("progression", 0.10) * existing * row.get("career_progression", 0.0)
            + jd_w * row.get("career_jd_semantic_similarity", 0.0)
        )
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def _log_stats(self, df: pd.DataFrame) -> None:
        for feat in ["title_relevance", "career_consistency", "career_score",
                     "career_jd_semantic_similarity"]:
            if feat in df.columns:
                logger.info(f"  {feat:35s} mean={df[feat].mean():.4f} max={df[feat].max():.4f}")