# sandbox/utils/metrics.py
import pandas as pd

def calculate_honeypot_rate(df: pd.DataFrame, top_n: int = 100) -> float:
    """Calculates the percentage of candidates in the top_n that are penalized as honeypots."""
    if len(df) == 0:
        return 0.0
    sub = df.head(top_n)
    if "honeypot_risk_score" not in sub.columns:
        return 0.0
    penalized = (sub["honeypot_risk_score"] > 0).sum()
    return float(penalized / len(sub))

def get_summary_stats(df: pd.DataFrame):
    """Generates simple statistics from the final scored candidate pool."""
    if len(df) == 0:
        return {
            "total_processed": 0,
            "max_score": 0.0,
            "mean_score": 0.0,
            "top_candidate": "N/A"
        }
    return {
        "total_processed": len(df),
        "max_score": float(df["final_score"].max()),
        "mean_score": float(df["final_score"].mean()),
        "top_candidate": str(df.iloc[0]["candidate_id"]) if "candidate_id" in df.columns else "N/A"
    }
