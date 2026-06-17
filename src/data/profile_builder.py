"""
Profile Builder.
Transforms cleaned candidates into processed profiles.
Produces unified_text for BM25 and embeddings.
"""

import re
import pandas as pd
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProfileBuilder:

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Building profiles for {len(df):,} candidates")
        df = df.copy()

        if "current_title" not in df.columns and "profile" in df.columns:
            df["current_title"] = df["profile"].apply(self._extract_current_title)

        df["title_normalized"] = df["current_title"].apply(self._norm)
        df["skills_normalized"] = df["skills"].apply(self._norm_skills)
        df["career_titles"] = df["career_history"].apply(self._extract_titles)
        df["career_descriptions_combined"] = df["career_history"].apply(self._extract_descriptions)
        df["career_duration_years"] = df["career_history"].apply(self._extract_durations)
        df["total_experience"] = df["career_history"].apply(self._extract_total_experience)
        df["career_companies"] = df["career_history"].apply(self._extract_companies)
        df["career_industries"] = df["career_history"].apply(self._extract_industries)
        df["unified_text"] = df.apply(self._build_unified_text, axis=1)

        logger.info("Profile building complete")
        return df

    def _norm(self, text) -> str:
        if not isinstance(text, str): return ""
        return " ".join(text.lower().strip().split())

    def _extract_current_title(self, profile) -> str:
        if not isinstance(profile, dict):
            return ""
        title = profile.get("current_title") or profile.get("current_role") or profile.get("designation") or ""
        return str(title).strip()

    def _norm_skills(self, skills) -> List[str]:
        if isinstance(skills, pd.Series):
            skills = skills.tolist()
        if hasattr(skills, "tolist") and not isinstance(skills, list):
            try:
                skills = skills.tolist()
            except Exception:
                return []
        if not isinstance(skills, list):
            return []
        seen, result = set(), []
        for s in skills:
            name = None
            if isinstance(s, str):
                name = s
            elif isinstance(s, dict):
                name = s.get("name") or s.get("skill") or s.get("label") or ""
            elif hasattr(s, "item"):
                try:
                    name = s.item()
                except Exception:
                    name = ""

            if isinstance(name, str) and name.strip():
                n = name.lower().strip()
                if n not in seen:
                    seen.add(n)
                    result.append(n)
        return result

    def _extract_titles(self, career) -> List[str]:
        if not isinstance(career, list): return []
        titles = []
        for e in career:
            if isinstance(e, dict):
                t = e.get("title") or e.get("job_title") or e.get("position") or ""
                if t.strip():
                    titles.append(t.lower().strip())
        return titles

    def _extract_descriptions(self, career) -> str:
        if not isinstance(career, list): return ""
        parts = []
        for e in career:
            if isinstance(e, dict):
                d = e.get("description") or e.get("responsibilities") or e.get("summary") or ""
                if d.strip():
                    parts.append(d.strip())
        return " ".join(parts).lower()

    def _extract_durations(self, career) -> List[float]:
        if not isinstance(career, list): return []
        durations = []
        for e in career:
            if isinstance(e, dict):
                d = (
                    e.get("duration_months")
                    or e.get("duration")
                    or e.get("years")
                    or 0
                )
                try:
                    value = float(d)
                    if e.get("duration_months") is not None:
                        value = value / 12.0
                    durations.append(value)
                except (TypeError, ValueError):
                    durations.append(0.0)
        return durations

    def _extract_total_experience(self, career) -> float:
        if not isinstance(career, list):
            return float("nan")
        total_months = 0.0
        has_value = False
        for e in career:
            if not isinstance(e, dict):
                continue
            d = e.get("duration_months")
            if d is None:
                d = e.get("duration") or e.get("years") or 0
            try:
                value = float(d)
            except (TypeError, ValueError):
                continue
            if e.get("duration_months") is not None:
                total_months += max(value, 0.0)
            else:
                total_months += max(value, 0.0) * 12.0
            has_value = True
        if not has_value or total_months <= 0:
            return float("nan")
        return round(total_months / 12.0, 2)

    def _extract_companies(self, career) -> List[str]:
        if not isinstance(career, list): return []
        return [
            str(e.get("company") or e.get("organization") or "").lower().strip()
            for e in career if isinstance(e, dict)
        ]

    def _extract_industries(self, career) -> List[str]:
        if not isinstance(career, list): return []
        return [
            str(e.get("industry") or "").lower().strip()
            for e in career if isinstance(e, dict) and e.get("industry")
        ]

    def _build_unified_text(self, row: pd.Series) -> str:
        parts = []
        title = row.get("title_normalized", "")
        if title:
            parts.extend([title, title])  # Repeat for BM25 weight
        skills = row.get("skills_normalized", [])
        if skills:
            parts.append(" ".join(skills))
        career_text = row.get("career_descriptions_combined", "")
        if career_text:
            parts.append(career_text)
        career_titles = row.get("career_titles", [])
        if career_titles:
            parts.append(" ".join(career_titles))
        edu = row.get("education_field", "")
        if edu and edu != "unknown":
            parts.append(edu)
        industry = str(row.get("industry", "")).lower().strip()
        if industry:
            parts.append(industry)
        return " ".join(parts)