"""
Candidate data loader.
Supports: .jsonl, .jsonl.gz, .json, .parquet
Applies schema adapter automatically.
"""

import gzip
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from src.utils.logger import get_logger
from src.data.schema_adapter import SchemaAdapter

logger = get_logger(__name__)


class CandidateLoader:

    def __init__(self, schema_file: Optional[str] = None):
        self.adapter = SchemaAdapter(schema_file)

    def load(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p.absolute()}")

        name = p.name.lower()
        suffix = p.suffix.lower()

        if name.endswith(".jsonl.gz"):
            df = self._load_jsonl_gz(p)
        elif suffix == ".jsonl":
            df = self._load_jsonl(p)
        elif suffix == ".json":
            df = self._load_json(p)
        elif suffix == ".parquet":
            df = pd.read_parquet(p)
            logger.info(f"Parquet loaded: {len(df):,} rows")
            return df
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        logger.info(f"Loaded {len(df):,} candidates")
        df = self.adapter.adapt(df)
        df = self._clean(df)
        return df

    def _load_jsonl(self, p: Path) -> pd.DataFrame:
        records = []
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Loading JSONL")):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Bad JSON at line {i+1}")
        return pd.DataFrame(records)

    def _load_jsonl_gz(self, p: Path) -> pd.DataFrame:
        records = []
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Loading JSONL.GZ")):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Bad JSON at line {i+1}")
        return pd.DataFrame(records)

    def _load_json(self, p: Path) -> pd.DataFrame:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        for key in ["candidates", "data", "records"]:
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])
        return pd.DataFrame([data])

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        import ast

        # Numeric fields: preserve -1 as sentinel for missing
        float_cols = [
            "total_experience", "recruiter_response_rate",
            "interview_completion_rate", "github_activity_score",
            "profile_completeness", "offer_acceptance_rate",
            "expected_salary",
        ]
        int_cols = [
            "saved_by_recruiters_30d", "search_appearances_30d",
            "notice_period_days", "connection_count",
            "endorsements_received", "applications_submitted_30d",
            "profile_views_30d", "education_tier",
        ]
        bool_cols = ["open_to_work", "relocation_willingness"]
        list_cols = ["skills", "career_history", "certifications"]

        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)

        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        if "total_experience" in df.columns:
            mask = df["total_experience"] != -1
            df.loc[mask, "total_experience"] = df.loc[mask, "total_experience"].clip(lower=0)

        true_vals = {"true", "1", "yes", "y"}
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: bool(v) if isinstance(v, bool)
                    else (str(v).lower().strip() in true_vals)
                )

        for col in list_cols:
            if col not in df.columns:
                df[col] = [[] for _ in range(len(df))]
                continue

            def to_list(v):
                if isinstance(v, list):
                    return v
                if isinstance(v, str):
                    if v.strip() in ("", "[]"):
                        return []
                    try:
                        r = ast.literal_eval(v)
                        return r if isinstance(r, list) else [r]
                    except Exception:
                        return [x.strip() for x in v.split(",") if x.strip()]
                return []

            df[col] = df[col].apply(to_list)

        # Ensure candidate_id
        if "candidate_id" not in df.columns or df["candidate_id"].eq("").all():
            df["candidate_id"] = [f"CAND_{i:06d}" for i in range(len(df))]
        else:
            df["candidate_id"] = df["candidate_id"].astype(str).str.strip()
            empty = df["candidate_id"].eq("")
            if empty.any():
                df.loc[empty, "candidate_id"] = [
                    f"AUTO_{i}" for i in range(empty.sum())
                ]

        return df