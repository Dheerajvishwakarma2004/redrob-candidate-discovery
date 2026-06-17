"""
Concept bucket definitions. Central registry. Import from here only.
"""

from typing import Dict, List

CONCEPT_BUCKETS: Dict[str, List[str]] = {
    "retrieval": [
        "search", "retrieval", "bm25", "faiss", "elasticsearch", "opensearch",
        "vector search", "vector database", "pinecone", "weaviate", "milvus",
        "semantic search", "lexical search", "hybrid search", "candidate matching",
        "document retrieval", "information retrieval", "search engine",
        "query understanding", "query expansion", "inverted index", "tf-idf",
        "dense retrieval", "sparse retrieval", "ann", "approximate nearest neighbor",
        "nearest neighbor search", "vector store", "search infrastructure",
        "search quality", "search relevance", "retrieval system",
    ],
    "ranking": [
        "ranking", "learning to rank", "ltr", "lambdamart", "xgboost ranker",
        "lightgbm ranker", "pointwise", "pairwise", "listwise", "ndcg", "map",
        "mrr", "relevance", "relevance score", "relevance modeling", "ranking model",
        "reranking", "reranker", "cross-encoder", "bi-encoder", "ranking pipeline",
        "search ranking", "result ranking", "candidate ranking", "profile ranking",
    ],
    "recommendation": [
        "recommendation", "recommender", "collaborative filtering",
        "content-based filtering", "matrix factorization", "personalization",
        "personalized", "suggestion", "recommendation engine",
        "recommendation system", "candidate discovery", "matching",
        "matching system", "matching engine", "job matching", "talent matching",
        "similar candidates", "similar profiles", "user-item",
    ],
    "ml_engineering": [
        "mlops", "mlflow", "kubeflow", "feature engineering", "feature store",
        "feature pipeline", "model deployment", "model serving", "inference",
        "training pipeline", "experiment tracking", "model monitoring",
        "a/b testing", "production ml", "model evaluation", "pipeline",
        "airflow", "spark", "kafka", "etl", "data pipeline", "machine learning",
        "scikit-learn", "xgboost", "lightgbm", "gradient boosting",
        "random forest", "deep learning", "neural network", "pytorch", "tensorflow",
        "model optimization", "hyperparameter", "cross-validation",
    ],
    "llm": [
        "llm", "large language model", "rag", "retrieval augmented generation",
        "fine-tuning", "lora", "qlora", "embeddings", "sentence embeddings",
        "transformers", "huggingface", "openai", "gpt", "prompt engineering",
        "instruction tuning", "sft", "rlhf", "vector embeddings", "text embeddings",
        "bert", "roberta", "t5", "llama", "mistral", "foundation model",
        "natural language processing", "nlp", "text classification",
        "named entity recognition", "sentiment analysis",
    ],
}

NEGATIVE_CAREER_CONCEPTS: List[str] = [
    "sales", "marketing", "accounting", "finance manager", "hr manager",
    "human resources", "recruitment consultant", "operations manager",
    "manufacturing", "mechanical engineering", "civil engineering",
    "construction", "graphic design", "customer service", "retail",
    "administration", "supply chain manager", "logistics manager",
    "purchasing", "procurement", "real estate", "insurance agent",
]

FOUNDATIONAL_SKILLS = {
    "python", "sql", "pandas", "numpy", "scikit-learn", "jupyter",
    "git", "docker", "linux", "bash", "java", "scala", "r", "matlab",
    "tensorflow", "pytorch", "spark", "hadoop", "airflow", "kubernetes",
}