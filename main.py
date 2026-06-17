"""
Redrob Intelligent Candidate Discovery
Single entry point for full pipeline.

Usage:
    python main.py --candidates candidates.jsonl --jd job_description.docx --output submission.csv
    python main.py --candidates candidates.jsonl.gz --jd job_description.docx
    python main.py --candidates candidates.jsonl --jd job_description.md --skip-cache
    python main.py --candidates candidates.jsonl --jd job_description.docx --validate
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.config_loader import config
from src.utils.logger import get_logger
from src.jd.document_loader import DocumentLoader
from src.jd.jd_extractor import JDExtractor
from src.data.loader import CandidateLoader
from src.data.profile_builder import ProfileBuilder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.features.career import CareerFeatureEngine
from src.features.technical import TechnicalFeatureEngine
from src.features.behavioral import BehaviorFeatureEngine
from src.features.experience import ExperienceFeatureEngine
from src.features.education import EducationFeatureEngine
from src.features.semantic import SemanticFeatureEngine
from src.features.alignment import AlignmentFeatureEngine
from src.ranking.honeypot import HoneypotEngine
from src.ranking.archetype import ArchetypeClassifier
from src.ranking.scorer import ScoringEngine
from src.reasoning.generator import ReasoningGenerator
from src.output.writer import OutputWriter

logger = get_logger("main")


def _needs_profile_rebuild(df: pd.DataFrame) -> bool:
    required = ["skills_normalized", "total_experience", "title_normalized", "career_titles"]
    if any(col not in df.columns for col in required):
        return True

    def _nonempty_seq(v):
        if v is None:
            return False
        if isinstance(v, (list, tuple)):
            return len(v) > 0
        if hasattr(v, "size"):
            try:
                return int(v.size) > 0
            except Exception:
                return False
        return False

    skills_ok = df["skills_normalized"].apply(_nonempty_seq).any()
    exp_ok = df["total_experience"].notna().any()
    titles_ok = df["career_titles"].apply(_nonempty_seq).any()
    return not (skills_ok and exp_ok and titles_ok)


def parse_args():
    p = argparse.ArgumentParser(description="Redrob Candidate Discovery v3.0")
    p.add_argument("--candidates", required=True,
                   help="candidates.jsonl or candidates.jsonl.gz")
    p.add_argument("--jd", required=True,
                   help="job_description.docx / .md / .txt")
    p.add_argument("--output", default=None,
                   help="Override submission output path")
    p.add_argument("--schema", default=None,
                   help="candidate_schema.json (optional)")
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--skip-cache", action="store_true",
                   help="Regenerate embeddings even if cache exists")
    p.add_argument("--validate", action="store_true",
                   help="Run validator after submission generation")
    p.add_argument("--mode", choices=["submission", "generalized"], default="submission",
                   help="Execution mode: submission (uses precomputed cache) or generalized (BM25 first + pool embedding on-the-fly)")
    return p.parse_args()


def run(args) -> pd.DataFrame:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("REDROB CANDIDATE DISCOVERY PIPELINE v3.0")
    logger.info("=" * 60)

    processed_path = Path(config.get("paths", "processed_candidates"))

    # ── Stage 1: Data Loading ──────────────────────────────────────────
    t = time.time()
    logger.info("[1/11] Data Loading")
    if processed_path.exists() and not args.skip_cache:
        logger.info(f"Loading cached: {processed_path}")
        df = pd.read_parquet(processed_path)
        if _needs_profile_rebuild(df):
            logger.info("Cached profiles are stale; rebuilding from raw candidates")
            loader = CandidateLoader(schema_file=args.schema)
            df = loader.load(args.candidates)
            builder = ProfileBuilder()
            df = builder.build(df)
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(processed_path, index=False)
            logger.info(f"Processed data refreshed: {processed_path}")
    else:
        loader = CandidateLoader(schema_file=args.schema)
        df = loader.load(args.candidates)
        builder = ProfileBuilder()
        df = builder.build(df)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path, index=False)
        logger.info(f"Processed data cached: {processed_path}")
    logger.info(f"  {len(df):,} candidates | {time.time()-t:.1f}s")

    # Ensure string columns don't have NaN to prevent type crashes in features
    for col in ["career_descriptions_combined", "industry", "title_normalized"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # ── Stage 2: JD Understanding ──────────────────────────────────────
    t = time.time()
    logger.info("[2/11] JD Understanding")
    jd_text = DocumentLoader().load(args.jd)
    jd_profile = JDExtractor().extract(jd_text)
    logger.info(f"  Domain: {jd_profile['primary_domain']} | {time.time()-t:.1f}s")

    # ── Stage 3: Embeddings ────────────────────────────────────────────
    t = time.time()
    emb_retriever = EmbeddingRetriever()
    if args.mode == "submission":
        logger.info("[3/11] Embedding Preparation")
        emb_retriever.prepare(df, source_path=None if args.skip_cache else args.candidates)
    else:
        logger.info("[3/11] Embedding Preparation (Skipped in generalized mode)")
    logger.info(f"  {time.time()-t:.1f}s")

    # ── Stage 4: BM25 ─────────────────────────────────────────────────
    t = time.time()
    logger.info("[4/11] BM25 Index")
    bm25 = BM25Retriever()
    bm25.build(df)
    logger.info(f"  {time.time()-t:.1f}s")

    # ── Stage 5: Retrieval ──────────────────────────────────────
    t = time.time()
    if args.mode == "submission":
        logger.info("[5/11] Hybrid Retrieval")
        pool = HybridRetriever(bm25, emb_retriever).retrieve(jd_profile, df)
    else:
        logger.info("[5/11] BM25-First Retrieval (Generalized Mode)")
        bm25_res = bm25.retrieve(jd_profile, top_k=config.get("retrieval", "final_pool_size", default=1500))
        pool = bm25_res.merge(df, on="candidate_id", how="left")
    logger.info(f"  Pool: {len(pool):,} candidates | {time.time()-t:.1f}s")

    # Ensure string columns don't have NaN in the pool to prevent type crashes in features
    for col in ["career_descriptions_combined", "industry", "title_normalized"]:
        if col in pool.columns:
            pool[col] = pool[col].fillna("")

    # Ensure list columns don't have NaN or non-list values in the pool
    list_cols = ["skills_normalized", "career_titles", "career_duration_years", "career_companies", "career_industries"]
    for col in list_cols:
        if col in pool.columns:
            pool[col] = pool[col].apply(lambda d: d if isinstance(d, (list, np.ndarray)) else [])

    # ── Stage 6: Feature Engineering & Reranking ──────────────────────
    t = time.time()
    if args.mode == "submission":
        logger.info("[6/11] Feature Engineering")
        career_embs = emb_retriever.get_career_embeddings_batch(pool["candidate_id"].tolist())
    else:
        logger.info("[6/11] On-the-fly Embedding Reranking & Feature Engineering")
        career_embs = emb_retriever.embed_pool_on_the_fly(pool, jd_profile)
        pool["retrieval_score"] = 0.5 * pool["bm25_score_norm"] + 0.5 * pool["embedding_similarity"]

    pool = CareerFeatureEngine(jd_profile, career_embs).compute(pool)
    pool = TechnicalFeatureEngine(jd_profile).compute(pool)
    pool = BehaviorFeatureEngine().compute(pool)
    pool = ExperienceFeatureEngine(jd_profile).compute(pool)
    pool = EducationFeatureEngine().compute(pool)
    pool = SemanticFeatureEngine().compute(pool)
    pool = AlignmentFeatureEngine(jd_profile, career_embs).compute(pool)
    logger.info(f"  {time.time()-t:.1f}s")

    # ── Stage 7: Honeypot Detection ────────────────────────────────────
    t = time.time()
    logger.info("[7/11] Honeypot Detection")
    pool = HoneypotEngine().compute(pool)
    logger.info(f"  {time.time()-t:.1f}s")

    # ── Stage 8: Archetype Classification ─────────────────────────────
    t = time.time()
    logger.info("[8/11] Archetype Classification")
    pool = ArchetypeClassifier().classify(pool)
    logger.info(f"  {time.time()-t:.1f}s")

    # ── Stage 9: Scoring ───────────────────────────────────────────────
    t = time.time()
    logger.info("[9/11] Scoring & Ranking")
    scorer = ScoringEngine()
    ranked = scorer.score_and_rank(pool)
    logger.info(f"  {time.time()-t:.1f}s")

    # ── Stage 10: Reasoning ────────────────────────────────────────────
    t = time.time()
    logger.info("[10/11] Reasoning Generation")
    ranked = ReasoningGenerator(jd_profile).generate(ranked)
    logger.info(f"  {time.time()-t:.1f}s")

    # ── Stage 11: Output ───────────────────────────────────────────────
    t = time.time()
    logger.info("[11/11] Output")
    top_n = args.top_n
    output_df = scorer.top_n(ranked, n=top_n)
    writer = OutputWriter()
    writer.write(output_df)
    logger.info(f"  {time.time()-t:.1f}s")

    if args.output:
        import shutil
        src = Path(config.get("paths", "submission_output"))
        dst = Path(args.output)
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            logger.info(f"Copied to: {args.output}")
        else:
            logger.info(f"Output already at: {args.output}")

    total = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE: {total:.1f}s")
    logger.info(f"Submission: {config.get('paths', 'submission_output')}")
    logger.info("=" * 60)

    if args.validate:
        from tools.validate import run_validation
        run_validation(config.get("paths", "submission_output"))

    return output_df


def main():
    args = parse_args()
    try:
        run(args)
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()