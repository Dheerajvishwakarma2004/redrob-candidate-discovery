"""
Schema Adapter.
Maps competition JSONL field names to internal schema.
Update FIELD_MAP if actual schema uses different names.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

FIELD_MAP: Dict[str, str] = {
    "id": "candidate_id", "_id": "candidate_id", "profile_id": "candidate_id",
    "name": "full_name", "candidate_name": "full_name",
    "title": "current_title", "job_title": "current_title",
    "current_role": "current_title", "designation": "current_title",
    "experience": "total_experience", "years_of_experience": "total_experience",
    "experience_years": "total_experience", "yoe": "total_experience",
    "sector": "industry", "domain": "industry",
    "skill_set": "skills", "technical_skills": "skills", "key_skills": "skills",
    "work_experience": "career_history", "employment_history": "career_history",
    "work_history": "career_history", "jobs": "career_history",
    "education_details": "education", "qualifications": "education",
    "certificates": "certifications", "credentials": "certifications",
    "signals": "redrob_signals", "platform_signals": "redrob_signals",
    "behavioral_signals": "redrob_signals",
}

SIGNAL_MAP: Dict[str, str] = {
    "response_rate": "recruiter_response_rate",
    "saved_by_recruiters": "saved_by_recruiters_30d",
    "recruiter_saves_30d": "saved_by_recruiters_30d",
    "profile_saves": "saved_by_recruiters_30d",
    "search_appearances": "search_appearances_30d",
    "interview_completion": "interview_completion_rate",
    "open_to_opportunities": "open_to_work",
    "actively_looking": "open_to_work",
    "notice_period": "notice_period_days",
    "joining_notice": "notice_period_days",
    "github_score": "github_activity_score",
    "github_activity": "github_activity_score",
    "profile_complete": "profile_completeness",
    "completeness_score": "profile_completeness",
    "connections": "connection_count",
    "endorsements": "endorsements_received",
    "applications_submitted": "applications_submitted_30d",
    "profile_views": "profile_views_30d",
    "salary_expectation": "expected_salary",
    "desired_salary": "expected_salary",
    "offer_acceptance": "offer_acceptance_rate",
    "work_mode": "work_mode_preference",
    "preferred_work_mode": "work_mode_preference",
    "willing_to_relocate": "relocation_willingness",
    "relocation": "relocation_willingness",
    "days_since_active": "last_active_days",
    "last_seen_days": "last_active_days",
    "response_time": "avg_response_time_hours",
    "avg_response_time": "avg_response_time_hours",
}


class SchemaAdapter:

    def __init__(self, schema_file: Optional[str] = None):
        self._schema = {}
        if schema_file and Path(schema_file).exists():
            with open(schema_file) as f:
                self._schema = json.load(f)
            logger.info(f"Official schema loaded: {schema_file}")

    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Adapting schema. Input columns: {list(df.columns)}")
        df = df.copy()
        df = self._rename_columns(df)
        df = self._flatten_signals(df)
        df = self._normalize_education(df)
        logger.info(f"Schema adapted. Output columns: {list(df.columns)}")
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {c: FIELD_MAP[c] for c in df.columns if c in FIELD_MAP}
        if rename:
            df = df.rename(columns=rename)
            logger.info(f"Renamed columns: {rename}")
        return df

    def _flatten_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "redrob_signals" not in df.columns:
            flat_rename = {c: SIGNAL_MAP[c] for c in df.columns if c in SIGNAL_MAP}
            if flat_rename:
                df = df.rename(columns=flat_rename)
            return df

        first = df["redrob_signals"].dropna()
        if first.empty or not isinstance(first.iloc[0], dict):
            df = df.drop(columns=["redrob_signals"], errors="ignore")
            return df

        signals = df["redrob_signals"].apply(
            lambda x: x if isinstance(x, dict) else {}
        ).apply(pd.Series)

        sig_rename = {c: SIGNAL_MAP[c] for c in signals.columns if c in SIGNAL_MAP}
        if sig_rename:
            signals = signals.rename(columns=sig_rename)

        df = df.drop(columns=["redrob_signals"])
        df = pd.concat([df.reset_index(drop=True), signals.reset_index(drop=True)], axis=1)
        logger.info(f"Flattened {len(signals.columns)} signal columns")
        return df

    def _normalize_education(self, df: pd.DataFrame) -> pd.DataFrame:
        if "education" not in df.columns:
            for col in ["education_tier", "education_field", "education_degree"]:
                df[col] = 0 if col == "education_tier" else "unknown"
            return df

        def to_list(v):
            if isinstance(v, list): return v
            if isinstance(v, dict): return [v]
            return []

        df["education"] = df["education"].apply(to_list)

        def get(lst, *keys):
            if not lst: return None
            e = lst[0]
            if not isinstance(e, dict): return None
            for k in keys:
                if k in e: return e[k]
            return None

        def parse_tier(value):
            if value is None:
                return 0
            if isinstance(value, (int, float)):
                return int(value)

            text = str(value).strip().lower()
            if not text:
                return 0

            if text.startswith("tier_"):
                text = text.split("tier_", 1)[1]

            try:
                return int(text)
            except (TypeError, ValueError):
                return 0

        df["education_tier"] = df["education"].apply(
            lambda x: parse_tier(get(x, "tier", "college_tier"))
        )
        df["education_field"] = df["education"].apply(
            lambda x: str(get(x, "field_of_study", "field", "major", "branch") or "unknown").lower()
        )
        df["education_degree"] = df["education"].apply(
            lambda x: str(get(x, "degree", "qualification") or "unknown").lower()
        )
        df["education_institution"] = df["education"].apply(
            lambda x: str(get(x, "institution", "college", "university") or "unknown")
        )
        return df