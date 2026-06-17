"""
Honeypot Detection Engine.
Produces honeypot_risk_score subtracted from final score.
All logic is transparent and explainable.
Disqualification threshold: >10% honeypots in top 100.
"""

import numpy as np
import pandas as pd
import json
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HoneypotEngine:

    def __init__(self):
        c = config.get("honeypot", default={})
        self._zero_penalty = c.get("zero_career_penalty", 0.20)
        self._zero_skill_thresh = c.get("zero_career_skill_threshold", 0.45)
        self._mm_strong_p = c.get("mismatch_strong_penalty", 0.25)
        self._mm_strong_title = c.get("mismatch_strong_title_max", 0.35)
        self._mm_strong_tech = c.get("mismatch_strong_tech_min", 0.72)
        self._mm_weak_p = c.get("mismatch_weak_penalty", 0.15)
        self._mm_weak_title = c.get("mismatch_weak_title_max", 0.45)
        self._mm_weak_tech = c.get("mismatch_weak_tech_min", 0.78)
        self._mm_cons_p = c.get("mismatch_consistency_penalty", 0.15)
        self._mm_cons_max = c.get("mismatch_consistency_max", 0.22)
        self._mm_cons_tech = c.get("mismatch_consistency_tech", 0.68)
        self._found_p = c.get("foundational_penalty", 0.10)
        self._adv_min = c.get("advanced_min_count", 5)
        self._found_req = c.get("foundational_required", 1)
        self._dens_p = c.get("density_penalty", 0.10)
        self._dens_total = c.get("density_total_max", 5)
        self._dens_adv = c.get("density_advanced_min", 3)
        self._tl_strong_y = c.get("timeline_strong_years", 7)
        self._tl_weak_y = c.get("timeline_weak_years", 5)
        self._tl_strong_p = c.get("timeline_strong_penalty", 0.10)
        self._tl_weak_p = c.get("timeline_weak_penalty", 0.05)
        self._cap = c.get("total_cap", 0.45)
        self._p_assess = c.get("p_assess_penalty", 0.10)
        self._p_low_resp = c.get("p_low_resp_penalty", 0.15)
        self._p_unverified_claims = c.get("p_unverified_claims_penalty", 0.15)
        self._p_signup = c.get("p_signup_penalty", 0.05)
        self._p_inactive_otw = c.get("p_inactive_otw_penalty", 0.03)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Honeypot detection: {len(df):,} candidates")
        df = df.copy()
        
        # Calculate max active date dynamically in the dataset
        if "last_active_date" in df.columns:
            self._max_active_dt = pd.to_datetime(df["last_active_date"]).max()
        else:
            self._max_active_dt = pd.to_datetime("2026-05-27")
            
        df["p_zero_career"] = df.apply(self._zero_career, axis=1)
        df["p_mismatch"] = df.apply(self._mismatch, axis=1)
        df["p_foundational"] = df.apply(self._foundational, axis=1)
        df["p_density"] = df.apply(self._density, axis=1)
        df["p_timeline"] = df.apply(self._timeline, axis=1)
        
        # New Honeypot checks
        df["p_assess"] = df.apply(self._assessment_contradiction, axis=1)
        df["p_low_resp"] = df.apply(self._low_response_high_interview, axis=1)
        df["p_unverified_claims"] = df.apply(self._unverified_claims, axis=1)
        df["p_signup"] = df.apply(self._signup_after_active, axis=1)
        df["p_inactive_otw"] = df.apply(self._inactive_open_to_work, axis=1)
        
        df["honeypot_risk_score"] = (
            df["p_zero_career"] + df["p_mismatch"] +
            df["p_foundational"] + df["p_density"] + df["p_timeline"] +
            df["p_assess"] + df["p_low_resp"] + df["p_unverified_claims"] +
            df["p_signup"] + df["p_inactive_otw"]
        ).clip(0.0, self._cap)
 
        fired = (df["honeypot_risk_score"] > 0).sum()
        strong = (df["honeypot_risk_score"] >= 0.20).sum()
        logger.info(f"  Penalized: {fired:,} | Strong: {strong:,}")
        return df

    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, np.ndarray):
            try:
                return value.tolist()
            except Exception:
                return [str(value)]
        try:
            return list(value)
        except Exception:
            return [value]

    def _zero_career(self, row) -> float:
        titles = self._as_list(row.get("career_titles", []))
        desc = str(row.get("career_descriptions_combined", "")).strip()
        durs = self._as_list(row.get("career_duration_years", []))
        has_career = len(titles) > 0 or len(desc) > 20 or len(durs) > 0
        if has_career: return 0.0
        tech = float(row.get("technical_score", 0.0))
        ret = float(row.get("retrieval_score", 0.0))
        rank = float(row.get("ranking_score", 0.0))
        if max(tech, ret, rank) >= self._zero_skill_thresh:
            return self._zero_penalty
        return 0.0

    def _mismatch(self, row) -> float:
        title = float(row.get("title_relevance", 0.5))
        tech = float(row.get("technical_score", 0.0))
        cons = float(row.get("career_consistency", 0.5))
        if title < self._mm_strong_title and tech > self._mm_strong_tech:
            return self._mm_strong_p
        if title < self._mm_weak_title and tech > self._mm_weak_tech:
            return self._mm_weak_p
        if cons < self._mm_cons_max and tech > self._mm_cons_tech:
            return self._mm_cons_p
        return 0.0

    def _foundational(self, row) -> float:
        adv = int(row.get("advanced_skill_count", 0))
        found = int(row.get("foundational_skill_count", 0))
        if adv >= self._adv_min and found < self._found_req:
            return self._found_p
        return 0.0

    def _density(self, row) -> float:
        skills = self._as_list(row.get("skills_normalized", []))
        total = len(skills)
        adv = int(row.get("advanced_skill_count", 0))
        if 0 < total <= self._dens_total and adv >= self._dens_adv:
            return self._dens_p
        return 0.0

    def _timeline(self, row) -> float:
        profile = row.get("profile")
        if isinstance(profile, str):
            import json
            try:
                profile = json.loads(profile)
            except Exception:
                profile = {}
        if not isinstance(profile, dict):
            profile = {}
        stated = float(profile.get("years_of_experience", 0.0))

        durs = self._as_list(row.get("career_duration_years", []))
        if len(durs) == 0 or stated <= 0:
            return 0.0
        valid = [d for d in durs if isinstance(d, (int, float)) and d > 0]
        if not valid: return 0.0
        disc = abs(sum(valid) - stated)
        if disc >= self._tl_strong_y: return self._tl_strong_p
        if disc >= self._tl_weak_y: return self._tl_weak_p
        return 0.0

    def _assessment_contradiction(self, row) -> float:
        assess = row.get("skill_assessment_scores")
        if isinstance(assess, str):
            try:
                assess = json.loads(assess)
            except Exception:
                assess = {}
        if not isinstance(assess, dict):
            assess = {}
            
        valid_assess = {k: v for k, v in assess.items() if v is not None}
        skills = self._as_list(row.get("skills_normalized", []))
        skills_set = {s.lower() for s in skills}
        
        for k, v in valid_assess.items():
            if v >= 80 and k.lower() not in skills_set:
                return self._p_assess
        return 0.0

    def _low_response_high_interview(self, row) -> float:
        resp = float(row.get("recruiter_response_rate", -1.0))
        intv = float(row.get("interview_completion_rate", -1.0))
        if resp >= 0 and intv >= 0:
            if resp < 0.10 and intv > 0.80:
                return self._p_low_resp
        return 0.0

    def _unverified_claims(self, row) -> float:
        tech = float(row.get("technical_score", 0.0))
        career_ev = float(row.get("career_description_relevance", 0.0))
        github = float(row.get("github_activity_score", -1.0))
        
        assess = row.get("skill_assessment_scores")
        if isinstance(assess, str):
            try:
                assess = json.loads(assess)
            except Exception:
                assess = {}
        if not isinstance(assess, dict):
            assess = {}
        valid_assess = [v for v in assess.values() if v is not None]
        max_assess = max(valid_assess) if valid_assess else 0.0
        
        if tech > 0.40 and career_ev < 0.15 and github <= 0 and max_assess < 40:
            return self._p_unverified_claims
        return 0.0

    def _signup_after_active(self, row) -> float:
        signup_str = row.get("signup_date")
        active_str = row.get("last_active_date")
        if signup_str and active_str:
            try:
                signup = pd.to_datetime(signup_str)
                active = pd.to_datetime(active_str)
                if signup > active:
                    return self._p_signup
            except Exception:
                pass
        return 0.0

    def _inactive_open_to_work(self, row) -> float:
        active_str = row.get("last_active_date")
        otw = bool(row.get("open_to_work_flag", False))
        if active_str and otw:
            try:
                active = pd.to_datetime(active_str)
                days = (self._max_active_dt - active).days
                if days > 180:
                    return self._p_inactive_otw
            except Exception:
                pass
        return 0.0