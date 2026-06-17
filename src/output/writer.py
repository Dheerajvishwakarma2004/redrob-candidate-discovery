"""
Output Writer. Competition-compliant submission generation.
Enforces: 100 rows, UTF-8, monotonic scores, unique ranks 1-100.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUBMISSION_COLS = ["candidate_id", "rank", "score", "reasoning"]

INTERNAL_COLS = [
    "rank", "candidate_id", "final_score", "raw_score",
    "archetype", "archetype_score",
    "career_score", "title_relevance", "career_consistency",
    "career_progression", "industry_alignment",
    "career_description_relevance", "career_jd_semantic_similarity",
    "technical_score", "retrieval_score", "ranking_score",
    "recommendation_score", "ml_engineering_score", "llm_score",
    "advanced_skill_count", "foundational_skill_count",
    "skill_career_alignment",
    "recruitability_score", "response_score", "saved_score",
    "interview_score", "open_to_work_score", "notice_score",
    "experience_fit", "total_experience",
    "education_score",
    "semantic_score", "bm25_score_norm", "embedding_similarity",
    "honeypot_risk_score", "p_zero_career", "p_mismatch",
    "p_foundational", "p_density", "p_timeline",
    "current_title", "industry", "open_to_work", "notice_period_days",
    "reasoning",
]


class OutputWriter:

    def __init__(self):
        self._internal = Path(config.get("paths", "internal_output"))
        self._submission = Path(config.get("paths", "submission_output"))
        self._top_n = config.get("output", "top_n", default=100)

    def write(self, df: pd.DataFrame) -> None:
        for p in [self._internal, self._submission]:
            p.parent.mkdir(parents=True, exist_ok=True)
        self._write_internal(df)
        self._write_submission(df)

    def _write_internal(self, df: pd.DataFrame) -> None:
        cols = [c for c in INTERNAL_COLS if c in df.columns]
        df[cols].to_csv(self._internal, index=False, encoding="utf-8", float_format="%.6f")
        logger.info(f"Internal: {self._internal} ({len(df):,} rows)")

    def _write_submission(self, df: pd.DataFrame) -> None:
        sub = df.copy()

        # Score column
        sub["score"] = sub["final_score"].round(6) if "final_score" in sub.columns else 0.0

        # Reasoning column
        if "reasoning" not in sub.columns:
            sub["reasoning"] = sub.get("explanation", "")
        sub["reasoning"] = sub["reasoning"].fillna("").astype(str)

        # Sort and take top N
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
        sub = sub.head(self._top_n)

        # Reassign ranks
        sub["rank"] = range(1, len(sub) + 1)

        # Verify monotonicity
        scores = sub["score"].values
        if not all(scores[i] >= scores[i+1] - 1e-9 for i in range(len(scores)-1)):
            logger.warning("Re-sorting scores for monotonicity")
            sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
            sub["rank"] = range(1, len(sub) + 1)

        # Select columns
        available = [c for c in SUBMISSION_COLS if c in sub.columns]
        final = sub[available].copy()
        final["candidate_id"] = final["candidate_id"].fillna("UNKNOWN").astype(str)

        final.to_csv(self._submission, index=False, encoding="utf-8", float_format="%.6f")

        # Integrity report
        self._report(final)

    def _report(self, df: pd.DataFrame) -> None:
        logger.info("=" * 50)
        logger.info("SUBMISSION INTEGRITY")
        logger.info("=" * 50)
        checks = {
            "Rows=100": len(df) == self._top_n,
            "Columns OK": set(SUBMISSION_COLS).issubset(set(df.columns)),
            "Ranks unique": len(df["rank"].unique()) == len(df),
            "Ranks 1-100": sorted(df["rank"].tolist()) == list(range(1, len(df)+1)),
            "Score monotonic": all(
                df["score"].iloc[i] >= df["score"].iloc[i+1] - 1e-9
                for i in range(len(df)-1)
            ),
            "No null IDs": df["candidate_id"].notna().all(),
            "Scores [0,1]": df["score"].between(0, 1).all(),
            "Reasoning present": (df["reasoning"].str.len() > 0).all(),
        }
        all_pass = True
        for name, result in checks.items():
            status = "PASS" if result else "FAIL"
            if not result: all_pass = False
            logger.info(f"  {name:25s}: {status}")
        logger.info(f"  {'Overall':25s}: {'ALL PASSED' if all_pass else 'FAILURES DETECTED'}")
        logger.info("=" * 50)
        if not all_pass:
            logger.error("Submission has compliance issues. Review before submitting.")