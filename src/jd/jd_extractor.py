"""
JD Feature Extractor.
Produces jd_profile: concept weights, JD embedding, experience range.
Concept weights are derived from JD embedding similarity.
They drive all downstream feature scoring dynamically.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONCEPT_DESCRIPTIONS = {
    "retrieval": (
        "search retrieval information retrieval vector search BM25 FAISS "
        "Elasticsearch semantic search query understanding document retrieval "
        "search engine dense retrieval sparse retrieval nearest neighbor"
    ),
    "ranking": (
        "ranking learning to rank relevance scoring NDCG MAP ranking model "
        "reranking relevance modeling candidate ranking result ranking "
        "pointwise pairwise listwise LambdaMART"
    ),
    "recommendation": (
        "recommendation systems recommender collaborative filtering "
        "personalization matching engine candidate discovery job matching "
        "content-based filtering matrix factorization similar profiles"
    ),
    "ml_engineering": (
        "machine learning MLOps feature engineering model deployment "
        "training pipeline production ML model serving experiment tracking "
        "A/B testing airflow spark data pipeline scikit-learn xgboost"
    ),
    "llm": (
        "large language models LLM RAG fine-tuning embeddings transformers "
        "prompt engineering NLP BERT GPT huggingface sentence embeddings "
        "text classification natural language processing"
    ),
}

EXPERIENCE_PATTERNS = [
    r"(\d+)\s*[-–to]+\s*(\d+)\s*years?",
    r"(\d+)\+\s*years?",
    r"(?:minimum|at least|min\.?)\s*(\d+)\s*years?",
]

KNOWN_TITLES = [
    "search engineer", "retrieval engineer", "recommendation systems engineer",
    "recommendation engineer", "ranking engineer", "relevance engineer",
    "ml engineer", "machine learning engineer", "ai engineer",
    "ai research engineer", "nlp engineer", "data scientist",
    "applied scientist", "research engineer", "software engineer",
    "backend engineer", "data engineer", "mlops engineer",
    "senior ml engineer", "staff ml engineer", "principal ml engineer",
]


class JDExtractor:

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            model_name = config.get("embedding", "model")
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
        return self._model

    def extract(self, jd_text: str) -> Dict:
        logger.info("Extracting JD features")
        clean = jd_text.lower()

        jd_embedding = self._embed(jd_text)
        concept_weights = self._compute_concept_weights(jd_embedding)
        exp_min, exp_max = self._extract_experience(clean)
        titles = self._extract_titles(clean)
        primary_domain = max(concept_weights, key=concept_weights.get)

        profile = {
            "raw_text": jd_text,
            "clean_text": clean,
            "jd_embedding": jd_embedding,
            "concept_weights": concept_weights,
            "primary_domain": primary_domain,
            "experience_min": exp_min,
            "experience_max": exp_max,
            "required_titles": titles,
        }

        self._log_profile(profile)
        return profile

    def _embed(self, text: str) -> np.ndarray:
        emb = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return emb.astype(np.float32)

    def _compute_concept_weights(self, jd_emb: np.ndarray) -> Dict[str, float]:
        threshold = config.get("jd", "low_confidence_threshold", default=0.28)
        sims = {}
        for name, desc in CONCEPT_DESCRIPTIONS.items():
            desc_emb = self.model.encode(desc, normalize_embeddings=True, show_progress_bar=False)
            sims[name] = float(max(np.dot(jd_emb, desc_emb.astype(np.float32)), 0.0))

        if max(sims.values()) < threshold:
            logger.warning("Low JD confidence. Using fallback weights.")
            fallback = config.get("jd", "fallback_weights", default={})
            total = sum(fallback.values()) or 1.0
            return {k: v / total for k, v in fallback.items()}

        total = sum(sims.values()) or 1.0
        normalized = {k: v / total for k, v in sims.items()}
        return normalized

    def _extract_experience(self, text: str) -> Tuple[float, float]:
        for pattern in EXPERIENCE_PATTERNS:
            m = re.search(pattern, text)
            if m:
                groups = m.groups()
                if len(groups) == 2 and groups[1]:
                    return float(groups[0]), float(groups[1])
                elif groups[0]:
                    mn = float(groups[0])
                    return mn, mn + 4.0
        return 5.0, 9.0

    def _extract_titles(self, text: str) -> List[str]:
        return [t for t in KNOWN_TITLES if t in text]

    def _log_profile(self, p: Dict) -> None:
        logger.info("JD Profile:")
        logger.info(f"  Domain:     {p['primary_domain']}")
        logger.info(f"  Experience: {p['experience_min']}-{p['experience_max']} years")
        logger.info(f"  Titles:     {p['required_titles']}")
        for k, v in sorted(p["concept_weights"].items(), key=lambda x: -x[1]):
            bar = "█" * int(v * 30)
            logger.info(f"  {k:20s}: {v:.4f} {bar}")