# tools/download_artifacts.py
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Public GitHub Release tag and Hugging Face Dataset mirror for binary tensors
GITHUB_REPO = "dheer/redrob-candidate-discovery"
RELEASE_TAG = "v3.0"
GITHUB_BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

HF_REPO = "dheer/redrob-candidate-discovery-artifacts"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# Local mapping to download targets (GitHub Release is primary, HF is fallback)
ARTIFACT_FILENAMES = {
    "data/processed/candidates.parquet": "candidates.parquet",
    "data/embeddings/unified.npy": "unified.npy",
    "data/embeddings/career.npy": "career.npy",
    "data/embeddings/id_order.npy": "id_order.npy",
    "data/embeddings/index.faiss": "index.faiss",
}

# Local Model Files
MODEL_FILES = [
    "config.json",
    "pytorch_model.bin",
    "model.safetensors",
    "vocab.txt",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "modules.json",
]
MODEL_BASE_URL = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main"
MODEL_DIR = "models/bge-small-en-v1.5"

def download_file(urls, dest):
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"Already exists: {dest}")
        return
    
    if not isinstance(urls, list):
        urls = [urls]
        
    for url in urls:
        print(f"Downloading {url} -> {dest} ...")
        try:
            urlretrieve(url, str(dest_path))
            print("Success.")
            return
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            
    print(f"ERROR: Could not download {dest} from any mirror.")
    sys.exit(1)

def main():
    print("=== REDROB ARTIFACT DOWNLOADER ===")
    
    # 1. Download cache embeddings
    print("\nDownloading precomputed embedding and index artifacts...")
    for local_path, filename in ARTIFACT_FILENAMES.items():
        primary = f"{GITHUB_BASE_URL}/{filename}"
        fallback = f"{HF_BASE_URL}/{filename}"
        download_file([primary, fallback], local_path)
        
    # 2. Download local model weights
    print("\nDownloading local model files...")
    for f in MODEL_FILES:
        url = f"{MODEL_BASE_URL}/{f}"
        dest = f"{MODEL_DIR}/{f}"
        download_file(url, dest)
        
    print("\nAll artifacts successfully downloaded.")

if __name__ == "__main__":
    main()
