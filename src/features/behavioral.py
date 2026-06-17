"""
Behavioral Feature Engine.
Computes recruitability_score from Redrob signals.

REDROB SIGNAL AUDIT (all 23 signals):

Signal                      | Used | Reason
----------------------------|------|------------------------------------------
recruiter_response_rate     | YES  | Weight 0.38. Strong hiring viability signal.
saved_by_recruiters_30d     | YES  | Weight 0.27. Crowdsourced recruiter validation.
interview_completion_rate   | YES  | Weight 0.22. Practical hiring viability.
open_to_work                | YES  | Weight 0.08. Immediate availability signal.
notice_period_days          | YES  | Weight 0.05. Time-to-hire indicator.
github_activity_score       | NO   | Median=-1 (missing). Non-discriminating.
search_appearances_30d      | NO   | Passive signal. Gameable. No hiring signal.
profile_completeness        | NO   | Gameable. Anyone can fill a profile.
connection_count            | NO   | Gameable. No direct hiring correlation.
endorsements_received       | NO   | Gameable. Low signal quality.
applications_submitted_30d  | NO   | Ambiguous (enthusiasm vs desperation).
profile_views_30d           | NO   | Passive. Keyword-driven not quality-driven.
expected_salary             | NO   | Cannot interpret without compensation bands.
offer_acceptance_rate       | NO   | -1 for most candidates. Too sparse.
work_mode_preference        | NO   | JD does not specify work mode requirement.
relocation_willingness      | NO   | Location data too synthetic.
last_active_days            | NO   | Could add value but field often missing.
avg_response_time_hours     | NO   | Correlated with response_rate. Redundant.
candidate_score             | NO   | If present: unknown provenance, circular.
platform_tenure_days        | NO   | No direct hiring relevance.
total_applications          | NO   | Lifetime version of applications_30d.
shortlist_rate              | NO   | If present: could be circular signal.
hire_rate                   | NO   | If present: leakage risk. Do not use.
"""

import numpy as np
import pandas as pd
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BehaviorFeatureEngine:

    def __init__(self):
        self._w = config.get("recruitability_weights", default={})

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Behavioral features: {len(df):,} candidates")
        df = df.copy()
        df["response_score"] = df.apply(self._response, axis=1)
        df["saved_score"] = df.apply(self._saved, axis=1)
        df["interview_score"] = df.apply(self._interview, axis=1)
        df["open_to_work_score"] = df.apply(self._otw, axis=1)
        df["notice_score"] = df.apply(self._notice, axis=1)
        df["recruitability_score"] = df.apply(self._aggregate, axis=1)
        logger.info(f"  recruitability_score mean={df['recruitability_score'].mean():.4f}")
        return df

    def _response(self, row) -> float:
        v = row.get("recruiter_response_rate", -1)
        if v == -1 or pd.isna(v): return 0.44
        return round(float(np.clip(v, 0.0, 1.0)), 4)

    def _saved(self, row) -> float:
        v = row.get("saved_by_recruiters_30d", 0)
        if pd.isna(v) or v < 0: return 0.0
        return round(float(min(v / 20.0, 1.0)), 4)

    def _interview(self, row) -> float:
        v = row.get("interview_completion_rate", -1)
        if v == -1 or pd.isna(v): return 0.62
        return round(float(np.clip(v, 0.0, 1.0)), 4)

    def _otw(self, row) -> float:
        v = row.get("open_to_work", False)
        if isinstance(v, bool): return 1.0 if v else 0.30
        if isinstance(v, (int, float)): return 1.0 if v else 0.30
        return 1.0 if str(v).lower().strip() in {"true", "yes", "1"} else 0.30

    def _notice(self, row) -> float:
        v = row.get("notice_period_days", 90)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.60
        if pd.isna(v): return 0.60
        if v <= 30:  return 1.0
        if v <= 60:  return 0.80
        if v <= 90:  return 0.60
        if v <= 120: return 0.40
        return 0.20

    def _aggregate(self, row) -> float:
        w = self._w
        score = (
            w.get("recruiter_response_rate", 0.38) * row.get("response_score", 0.0)
            + w.get("saved_by_recruiters_30d", 0.27) * row.get("saved_score", 0.0)
            + w.get("interview_completion_rate", 0.22) * row.get("interview_score", 0.0)
            + w.get("open_to_work", 0.08) * row.get("open_to_work_score", 0.0)
            + w.get("notice_period_days", 0.05) * row.get("notice_score", 0.0)
        )
        return round(float(np.clip(score, 0.0, 1.0)), 4)