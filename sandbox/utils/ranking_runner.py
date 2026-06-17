# sandbox/utils/ranking_runner.py
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.utils.config_loader import config
from src.data.loader import CandidateLoader
from src.data.profile_builder import ProfileBuilder
from src.jd.jd_extractor import JDExtractor
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

class SandboxRankingRunner:
    def __init__(self, model=None):
        self._model = model

    def run_pipeline(
        self,
        candidates_data: List[Dict],
        jd_text: str,
        mode: str = "submission",
        custom_weights: Optional[Dict] = None,
        custom_honeypot_cap: float = 0.45
    ) -> Tuple[pd.DataFrame, Dict]:
        t_start = time.time()
        
        # 1. Update config in memory with Streamlit slider weights
        if custom_weights:
            # Normalize weights to sum to 1.0
            total_w = sum(custom_weights.values()) or 1.0
            norm_weights = {k: v / total_w for k, v in custom_weights.items()}
            config._data["scoring"] = norm_weights
            
        if custom_honeypot_cap is not None:
            config._data["honeypot"]["total_cap"] = custom_honeypot_cap

        # 2. Ingest candidates from list of dicts (candidates_data)
        loader = CandidateLoader()
        df = pd.DataFrame(candidates_data)
        df = loader.adapter.adapt(df)
        df = loader._clean(df)
        
        # Build profiles
        builder = ProfileBuilder()
        df = builder.build(df)
        
        # Clean columns to match main.py
        for col in ["career_descriptions_combined", "industry", "title_normalized"]:
            if col in df.columns:
                df[col] = df[col].fillna("")
                
        # 3. JD Extraction
        extractor = JDExtractor()
        if self._model is not None:
            extractor._model = self._model
        jd_profile = extractor.extract(jd_text)
        
        # 4. Retrieval & Sourcing
        t_retrieval_start = time.time()
        
        emb_retriever = EmbeddingRetriever()
        if self._model is not None:
            emb_retriever._model = self._model
            
        bm25 = BM25Retriever()
        bm25.build(df)
        
        if mode == "submission":
            # In sandbox mode, prepare model embeddings for the sample dataset (usually 50-500 candidates)
            # Since the global cache is for the 100k set, the sample set must generate embeddings on-the-fly
            # so that FAISS indexing works correctly on this subset.
            emb_retriever._id_order = np.array(df["candidate_id"].tolist())
            emb_retriever._unified_embs = emb_retriever._encode(df["unified_text"].fillna("").tolist(), "Unified")
            emb_retriever._career_embs = emb_retriever._encode(df["career_descriptions_combined"].fillna("").tolist(), "Career")
            # Build FAISS index strictly in RAM (bypass disk write)
            import faiss
            dim = emb_retriever._unified_embs.shape[1]
            emb_retriever._index = faiss.IndexFlatIP(dim)
            emb_retriever._index.add(emb_retriever._unified_embs.astype(np.float32))
            
            pool = HybridRetriever(bm25, emb_retriever).retrieve(jd_profile, df)
            pool_career_embs = emb_retriever.get_career_embeddings_batch(pool["candidate_id"].tolist())
        else:
            # Generalized Mode
            bm25_res = bm25.retrieve(jd_profile, top_k=min(config.get("retrieval", "final_pool_size", default=1500), len(df)))
            pool = bm25_res.merge(df, on="candidate_id", how="left")
            pool_career_embs = emb_retriever.embed_pool_on_the_fly(pool, jd_profile)
            
        t_retrieval_duration = time.time() - t_retrieval_start
        
        # Clean string columns in pool
        for col in ["career_descriptions_combined", "industry", "title_normalized"]:
            if col in pool.columns:
                pool[col] = pool[col].fillna("")
                
        # Clean list columns
        list_cols = ["skills_normalized", "career_titles", "career_duration_years", "career_companies", "career_industries"]
        for col in list_cols:
            if col in pool.columns:
                pool[col] = pool[col].apply(lambda d: d if isinstance(d, (list, np.ndarray)) else [])
                
        # 5. Feature Engineering
        pool = CareerFeatureEngine(jd_profile, pool_career_embs).compute(pool)
        pool = TechnicalFeatureEngine(jd_profile).compute(pool)
        pool = BehaviorFeatureEngine().compute(pool)
        pool = ExperienceFeatureEngine(jd_profile).compute(pool)
        pool = EducationFeatureEngine().compute(pool)
        pool = SemanticFeatureEngine().compute(pool)
        pool = AlignmentFeatureEngine(jd_profile, pool_career_embs).compute(pool)
        
        # 6. Honeypot & Archetype
        pool = HoneypotEngine().compute(pool)
        pool = ArchetypeClassifier().classify(pool)
        
        # 7. Scoring & Ranking
        scorer = ScoringEngine()
        ranked = scorer.score_and_rank(pool)
        
        # 8. Grounded Reasoning Generation
        ranked = ReasoningGenerator(jd_profile).generate(ranked)
        
        duration = time.time() - t_start
        
        metadata = {
            "runtime_s": duration,
            "retrieval_s": t_retrieval_duration,
            "processed_count": len(df),
            "pool_size": len(ranked),
            "top_candidate": ranked.iloc[0]["candidate_id"] if len(ranked) > 0 else "N/A",
            "top_score": float(ranked["final_score"].max()) if len(ranked) > 0 else 0.0,
            "avg_score": float(ranked["final_score"].mean()) if len(ranked) > 0 else 0.0,
            "honeypots_count": int((ranked["honeypot_risk_score"] > 0).sum())
        }
        
        return ranked, metadata
