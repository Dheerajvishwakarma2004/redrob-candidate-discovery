"""
Embedding Retriever.
Generates and caches BGE-small embeddings.
FAISS flat index for exact search.
"""

import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingRetriever:

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._model_name = config.get("embedding", "model")
        self._batch_size = config.get("embedding", "batch_size", default=256)
        self._top_k = config.get("retrieval", "embedding_top_k", default=3000)
        self._unified_path = Path(config.get("paths", "unified_embeddings"))
        self._career_path = Path(config.get("paths", "career_embeddings"))
        self._faiss_path = Path(config.get("paths", "faiss_index"))
        self._id_path = Path(config.get("paths", "id_order"))
        self._index: Optional[faiss.Index] = None
        self._id_order: Optional[np.ndarray] = None
        self._unified_embs: Optional[np.ndarray] = None
        self._career_embs: Optional[np.ndarray] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def prepare(self, df: pd.DataFrame, source_path: Optional[str] = None) -> None:
        if self._cache_valid(source_path):
            self._load_cache()
            logger.info(f"Loading FAISS index from: {self._faiss_path}")
            self._index = faiss.read_index(str(self._faiss_path))
        else:
            missing = []
            for name, path in [
                ("unified_embeddings", self._unified_path),
                ("career_embeddings", self._career_path),
                ("faiss_index", self._faiss_path),
                ("id_order", self._id_path)
            ]:
                if not path.exists():
                    missing.append(f"{name} ({path})")
            if missing:
                raise FileNotFoundError(
                    f"Embedding cache validation failed. Missing artifacts: {missing}. "
                    "Regeneration is disabled during evaluation to guarantee Stage 3 reproducibility. "
                    "Ensure all precomputed embedding files are present in the repository."
                )
            else:
                logger.warning("Embedding cache files are stale relative to candidate source, but all are present. Proceeding with cache loading.")
                self._load_cache()
                logger.info(f"Loading FAISS index from: {self._faiss_path}")
                self._index = faiss.read_index(str(self._faiss_path))

    def retrieve(self, jd_profile: Dict, top_k: int = None) -> pd.DataFrame:
        if self._index is None:
            raise RuntimeError("Call prepare() first")
        k = top_k or self._top_k
        q = jd_profile["jd_embedding"].reshape(1, -1).astype(np.float32)
        dists, idxs = self._index.search(q, k)
        dists, idxs = dists[0], idxs[0]
        valid = idxs >= 0
        return pd.DataFrame({
            "candidate_id": [str(self._id_order[i]) for i in idxs[valid]],
            "embedding_similarity": np.clip(dists[valid], 0.0, 1.0)
        })

    def get_career_embeddings_batch(self, ids: List[str]) -> Dict[str, np.ndarray]:
        if self._career_embs is None or self._id_order is None:
            return {}
        id_to_idx = {str(cid): idx for idx, cid in enumerate(self._id_order)}
        return {
            cid: self._career_embs[idx]
            for cid in ids
            if (idx := id_to_idx.get(str(cid))) is not None
        }

    def _cache_valid(self, source_path: Optional[str]) -> bool:
        files = [self._unified_path, self._career_path, self._faiss_path, self._id_path]
        if not all(f.exists() for f in files):
            return False
        if source_path:
            sp = Path(source_path)
            if sp.exists():
                src_mtime = sp.stat().st_mtime
                cache_mtime = min(f.stat().st_mtime for f in files)
                if src_mtime > cache_mtime:
                    logger.info("Cache stale — source newer than embeddings")
                    return False
        logger.info("Embedding cache valid")
        return True

    def _load_cache(self) -> None:
        logger.info("Loading embeddings from cache")
        self._unified_embs = np.load(str(self._unified_path))
        self._career_embs = np.load(str(self._career_path))
        self._id_order = np.load(str(self._id_path), allow_pickle=True)
        logger.info(f"Cache loaded: {len(self._unified_embs):,} candidates")

    def _generate(self, df: pd.DataFrame) -> None:
        for p in [self._unified_path, self._career_path]:
            p.parent.mkdir(parents=True, exist_ok=True)
        self._id_order = np.array(df["candidate_id"].tolist())
        logger.info(f"Generating unified embeddings for {len(df):,} candidates")
        self._unified_embs = self._encode(df["unified_text"].fillna("").tolist(), "Unified")
        logger.info("Generating career embeddings")
        self._career_embs = self._encode(df["career_descriptions_combined"].fillna("").tolist(), "Career")
        np.save(str(self._unified_path), self._unified_embs)
        np.save(str(self._career_path), self._career_embs)
        np.save(str(self._id_path), self._id_order)
        logger.info("Embeddings saved")

    def _encode(self, texts: List[str], desc: str) -> np.ndarray:
        all_embs = []
        for i in tqdm(range(0, len(texts), self._batch_size), desc=desc):
            batch = [t if t.strip() else "no information" for t in texts[i:i+self._batch_size]]
            embs = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embs.append(embs.astype(np.float32))
        return np.vstack(all_embs)

    def _build_faiss(self) -> None:
        dim = self._unified_embs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._unified_embs.astype(np.float32))
        faiss.write_index(self._index, str(self._faiss_path))
        logger.info(f"FAISS index: {self._index.ntotal:,} vectors, dim={dim}")

    def embed_pool_on_the_fly(self, pool_df: pd.DataFrame, jd_profile: Dict) -> Dict[str, np.ndarray]:
        logger.info(f"Generating on-the-fly embeddings for pool of {len(pool_df):,} candidates")
        model = self.model
        
        # Embed career descriptions (truncated to 1000 chars to speed up CPU self-attention)
        career_texts = pool_df["career_descriptions_combined"].fillna("").astype(str).tolist()
        career_texts = [t[:1000] if t.strip() else "no information" for t in career_texts]
        career_embs = model.encode(career_texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False)
        
        career_emb_dict = {
            str(cid): career_embs[i]
            for i, cid in enumerate(pool_df["candidate_id"])
        }
        
        # Reuse career embeddings for query-candidate semantic similarity
        jd_emb = jd_profile["jd_embedding"]
        similarities = np.dot(career_embs, jd_emb.astype(np.float32))
        pool_df["embedding_similarity"] = np.clip(similarities, 0.0, 1.0)
        
        return career_emb_dict