"""
Reasoning Generator.
Produces fact-grounded, JD-connected, concern-aware reasoning.
Single paragraph per candidate. Varies across all 100.
No hallucinations: only references facts present in profile.
Satisfies Stage 4 manual review requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

EVIDENCE_PATTERNS = [
    ("search", "search system development"),
    ("retrieval", "retrieval pipeline work"),
    ("ranking", "ranking model development"),
    ("recommendation", "recommendation system experience"),
    ("personalization", "personalization pipeline work"),
    ("matching", "candidate or item matching"),
    ("relevance", "relevance optimization"),
    ("vector search", "vector search implementation"),
    ("information retrieval", "information retrieval"),
    ("feature engineering", "feature engineering work"),
    ("mlops", "MLOps and deployment"),
    ("model deployment", "model deployment experience"),
    ("candidate discovery", "candidate discovery systems"),
    ("semantic search", "semantic search"),
]


class ReasoningGenerator:

    def __init__(self, jd_profile: Dict = None):
        self._top_n = config.get("output", "top_n", default=100)
        self._jd = jd_profile or {}
        self._domain = self._jd.get("primary_domain", "search and ranking")
        self._exp_min = self._jd.get("experience_min", 5)
        self._exp_max = self._jd.get("experience_max", 9)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Generating reasoning for top {self._top_n} candidates")
        df = df.copy()
        df["reasoning"] = df.apply(
            lambda row: self._generate(row) if row.get("rank", 999) <= self._top_n else "",
            axis=1
        )
        logger.info("Reasoning complete")
        return df

    def _generate(self, row: pd.Series) -> str:
        s1 = self._who(row)
        s2 = self._jd_connection(row)
        s3 = self._strongest(row)
        s4 = self._concern(row)
        s5 = self._assessment(row)
        parts = [s1, s2, s3]
        if s4:
            parts.append(s4)
        parts.append(s5)
        return " ".join(parts)

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

    def _who(self, row) -> str:
        title = str(row.get("current_title", row.get("title_normalized", "Technical professional"))).strip()
        if not title: title = "Technical professional"
        exp = self._safe_float(row.get("total_experience", 0))
        industry = str(row.get("industry", "")).strip()
        skills = self._top_skills(row)

        if exp > 0 and industry and industry.lower() not in ["", "unknown", "nan"]:
            base = f"{title.title()} with {exp:.1f} years of experience in the {industry} sector"
        elif exp > 0:
            base = f"{title.title()} with {exp:.1f} years of experience"
        else:
            base = title.title()

        if skills:
            base += f", skilled in {', '.join(skills[:3])}"
        return base + "."

    def _jd_connection(self, row) -> str:
        career = str(row.get("career_descriptions_combined", "")).lower()
        jd_sim = self._safe_float(row.get("career_jd_semantic_similarity", 0))
        desc_rel = self._safe_float(row.get("career_description_relevance", 0))
        ret = self._safe_float(row.get("retrieval_score", 0))
        rank = self._safe_float(row.get("ranking_score", 0))
        rec = self._safe_float(row.get("recommendation_score", 0))

        evidence = self._find_evidence(career)
        if evidence and desc_rel >= 0.18:
            return (f"Career history includes {evidence}, directly relevant to "
                    f"the JD requirement for {self._domain} expertise.")
        if jd_sim >= 0.28:
            return (f"Career content shows semantic alignment with the JD's focus "
                    f"on {self._domain}, even where exact terminology differs.")
        if ret >= 0.38 or rank >= 0.38:
            dom = "retrieval and ranking" if ret >= rank else "ranking and relevance"
            return (f"Skill set includes {dom} concepts relevant to JD requirements, "
                    f"though career description evidence is limited.")
        if rec >= 0.38:
            return (f"Recommendation systems experience connects to the JD's "
                    f"emphasis on candidate discovery and personalization.")
        return (f"ML engineering background provides a relevant foundation "
                f"for the {self._domain} role described in the JD.")

    def _strongest(self, row) -> str:
        candidates = []
        alignment = self._safe_float(row.get("skill_career_alignment", 0))
        saved = self._safe_int(row.get("saved_by_recruiters_30d", 0))
        response = self._safe_float(row.get("recruiter_response_rate", -1))
        consistency = self._safe_float(row.get("career_consistency", 0))
        ret = self._safe_float(row.get("retrieval_score", 0))
        rank_s = self._safe_float(row.get("ranking_score", 0))
        exp_fit = self._safe_float(row.get("experience_fit", 0))
        exp = self._safe_float(row.get("total_experience", 0))

        if alignment >= 0.68:
            candidates.append((alignment,
                f"Skill-career alignment is strong ({alignment:.2f}): claimed technical skills "
                f"are supported by career evidence."))
        if saved >= 10:
            candidates.append((min(saved/20, 1.0),
                f"Saved by {saved} recruiters in the past 30 days, indicating strong market demand."))
        if response >= 0.75 and response != -1:
            candidates.append((response,
                f"Recruiter response rate of {response:.0%} demonstrates active engagement."))
        if consistency >= 0.76:
            n = len(row.get("career_titles", []))
            candidates.append((consistency,
                f"Consistent technical career across {n} roles strengthens domain confidence."))
        if max(ret, rank_s) >= 0.58:
            dom = "retrieval" if ret >= rank_s else "ranking"
            candidates.append((max(ret, rank_s),
                f"Strong {dom} engineering skills score ({max(ret, rank_s):.2f}) directly match JD."))
        if exp_fit >= 0.88 and exp > 0:
            candidates.append((exp_fit,
                f"Experience ({exp:.1f} years) is an excellent fit for the "
                f"{self._exp_min:.0f}–{self._exp_max:.0f} year target range."))

        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
        return (f"Profile shows a reasonable combination of technical skills "
                f"and career relevance for the target role.")

    def _concern(self, row) -> Optional[str]:
        z = self._safe_float(row.get("p_zero_career", 0))
        mm = self._safe_float(row.get("p_mismatch", 0))
        found = self._safe_float(row.get("p_foundational", 0))
        align = self._safe_float(row.get("skill_career_alignment", 1))
        tl = self._safe_float(row.get("p_timeline", 0))
        exp_fit = self._safe_float(row.get("experience_fit", 1))
        exp = self._safe_float(row.get("total_experience", 0))
        notice = row.get("notice_period_days", None)

        if z > 0:
            return ("Concern: advanced technical skills claimed but no career history "
                    "present to substantiate them.")
        if mm >= 0.20:
            return ("Concern: current role suggests limited direct domain experience; "
                    "skill claims appear partially unsupported by career evidence.")
        if found > 0:
            return ("Concern: profile lists advanced technical terms but lacks foundational "
                    "engineering skills, which is atypical for experienced practitioners.")
        if align < 0.36 and align > 0:
            return (f"Concern: skill-career alignment is low ({align:.2f}); some claimed "
                    f"technical skills lack career evidence, suggesting possible skill inflation.")
        if mm > 0:
            return ("Moderate concern: career background shows limited alignment with "
                    "the technical requirements of this role.")
        if tl > 0:
            return ("Minor note: stated experience shows some inconsistency "
                    "with computed career timeline.")
        if exp_fit < 0.62 and exp > 0:
            direction = "below" if exp < self._exp_min - 1 else "above"
            return (f"Note: experience ({exp:.1f} years) is {direction} the target range "
                    f"of {self._exp_min:.0f}–{self._exp_max:.0f} years.")
        try:
            notice_val = float(notice) if notice is not None else None
        except Exception:
            notice_val = None
        if notice_val is not None and not np.isnan(notice_val) and notice_val > 90:
            return f"Note: notice period of {int(notice_val)} days may delay onboarding."
        return None

    def _assessment(self, row) -> str:
        score = self._safe_float(row.get("final_score", 0))
        rank = int(row.get("rank", 50))
        archetype = str(row.get("archetype", "other"))
        otw = row.get("open_to_work", False)
        avail = " Candidate is currently open to opportunities." if otw else ""

        arch_desc = {
            "retrieval_recommendation": "a specialist retrieval and recommendation profile",
            "ml_engineer": "a strong ML engineering background",
            "data_software": "a solid software and data engineering foundation",
            "other": "a general technical profile",
        }.get(archetype, "a technical background")

        if score >= 0.74 and rank <= 10:
            return (f"Overall: strong candidate with {arch_desc} and multiple converging "
                    f"signals of fit. Recommended for immediate shortlist.{avail}")
        if score >= 0.64:
            return (f"Overall: good candidate with {arch_desc}. Recommended for "
                    f"screening interview to validate technical depth.{avail}")
        if score >= 0.52:
            return (f"Overall: moderate fit with {arch_desc}. Worth considering if "
                    f"the candidate pool is limited.{avail}")
        return (f"Overall: marginal fit. Included based on specific signal strength; "
                f"further qualification recommended before shortlisting.{avail}")

    def _top_skills(self, row) -> List[str]:
        skills = self._as_list(row.get("skills_normalized", []))
        if len(skills) == 0:
            return []
        priority = {
            "bm25", "faiss", "elasticsearch", "learning to rank",
            "recommendation", "ranking", "retrieval", "vector search",
            "mlflow", "mlops", "feature engineering", "pytorch",
            "tensorflow", "transformers", "rag", "embeddings",
            "python", "spark", "kafka", "llamaindex", "opensearch",
            "pinecone", "weaviate", "milvus", "qdrant", "hybrid search",
            "semantic search", "relevance", "llm", "llms", "fine-tuning",
            "lora", "qlora", "sentence transformers", "langchain",
            "huggingface", "nlp", "bert", "gpt", "kubeflow", "airflow",
            "kubernetes", "docker", "scikit-learn", "xgboost", "lightgbm"
        }
        exclude_terms = [
            "illustrator", "photoshop", "canvas", "design", "marketing",
            "sales", "hr", "recruiter", "recruitment", "finance",
            "accounting", "graphic", "angular", "react", "html", "css",
            "bootstrap"
        ]
        p = [s for s in skills if s in priority]
        o = [s for s in skills if s not in priority and not any(t in s for t in exclude_terms)]
        return (p[:4] + o[:1])[:5]

    def _find_evidence(self, career_text: str) -> Optional[str]:
        if not career_text or len(career_text) < 10:
            return None
        found = [phrase for kw, phrase in EVIDENCE_PATTERNS if kw in career_text]
        if not found: return None
        if len(found) == 1: return found[0]
        if len(found) == 2: return f"{found[0]} and {found[1]}"
        return f"{found[0]}, {found[1]}, and {found[2]}"

    def _safe_float(self, v, default=0.0) -> float:
        try: return float(v) if v is not None and not np.isnan(float(v)) else default
        except: return default

    def _safe_int(self, v, default=0) -> int:
        try: return int(v) if v is not None else default
        except: return default