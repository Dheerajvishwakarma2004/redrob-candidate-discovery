# sandbox/utils/formatter.py
import pandas as pd

def format_submission(results_df: pd.DataFrame) -> pd.DataFrame:
    """Formats the results dataframe into the required submission columns."""
    df = results_df.copy()
    if "rank" not in df.columns:
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        
    df = df.rename(columns={"final_score": "score"})
    required_cols = ["candidate_id", "rank", "score", "reasoning"]
    
    # Ensure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
            
    # Sort and slice
    df = df.sort_values("rank").reset_index(drop=True)
    return df[required_cols]
