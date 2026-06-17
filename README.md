# Intelligent Candidate Discovery & Ranking System (Redrob Hackathon v4)

A CPU-optimized candidate discovery and ranking system designed to identify the top 100 candidates from a 100,000-candidate database for a Retrieval & Ranking AI Engineer role.

This repository implements a **Dual-Mode Architecture** designed to balance high-speed cached evaluation with generalized out-of-domain matching on CPU-only machines.

---

## 📦 Submission Deliverables

This repository contains the following submission artifacts:

* Source Code Repository
* `outputs/submission.csv` (Top-100 ranked candidates)
* Streamlit Sandbox (`sandbox/`)
* Technical Documentation (`docs/`)

---

## 🚀 Quick Start (Stage 3 Grader Replication)

To reproduce the submission on a fresh machine, run these commands in sequence.

### 1. Prerequisite Setup
Ensure you have **Python 3.10+** installed. Clone the repository and install all dependencies:
```bash
# Clone the repository
git clone <repository_url>
cd redrob-candidate-discovery

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Download Large Binary Assets (Model Weights & Cache Tensors)
Due to Git's 100 MB single-file limit and LFS bandwidth caps, precomputed embedding tensors and offline model weights are hosted on our public Hugging Face mirror. Download the required model and retrieval artifacts:
```bash
python tools/download_artifacts.py
```
*Note: This script resolves missing files, verifying local checksums before terminating. If offline weights or index caches are not retrieved, Mode A execution will fail-fast with a descriptive error instead of attempting 25 minutes of embedding re-generation.*

### 3. Run Grader Evaluation (Mode A — Official Submission)
Run the pipeline in **Submission Mode** using the precomputed candidate database index:
```bash
python main.py --candidates data/raw/candidates.jsonl --jd data/raw/job_description.docx --mode submission --validate
```
*   **Output File:** `outputs/submission.csv` (validated and verified).

---

## 🔒 Reproducibility

A fresh clone requires the artifact download step before execution:

```bash
python tools/download_artifacts.py
```

Mode A relies on precomputed retrieval artifacts and embedding caches.

Mode B relies on locally downloaded embedding model weights.

Both modes execute fully offline after the artifact download step has completed successfully.

---

## 🌐 Dual-Mode Architectural Performance

| Metric | Mode A: Submission Mode | Mode B: Generalized Mode |
| :--- | :--- | :--- |
| **CLI Flag** | `--mode submission` (Default) | `--mode generalized` |
| **Underlying Retrieval** | Hybrid BM25 + Cached FAISS Dense Sourcing | Sourced pool via BM25 retrieval only |
| **Embedding Generation** | Loaded from disk (`unified.npy`, `career.npy`) | Generated on-the-fly for the Top 1,500 candidates |
| **Target Use Case** | Official static grader evaluation | Arbitrary candidate pools & hidden test sets |

**Reference benchmark environment:**
Both execution modes operate within the hackathon CPU-only runtime constraints. Actual execution time depends on hardware configuration and dataset characteristics.

### Mode B Execution Command
To run in generalized mode (bypassing precomputed embeddings to evaluate a new, unseen candidate pool):
```bash
python main.py --candidates data/raw/candidates.jsonl --jd data/raw/job_description.docx --mode generalized --validate
```

---

## 🛠️ Scoring Overview

The final fit score ($S$) is calculated deterministically as a weighted sum of normalized sub-scores, evaluating candidate profiles across technical claims, career history, semantic similarity to the JD, educational background, and behavioral recruitability telemetry.

For a complete breakdown of the mathematical formulas, scoring weights, and architectural pipeline, please refer to:
* [docs/architecture_one_pager.md](docs/architecture_one_pager.md)
* [docs/judge_faq.md](docs/judge_faq.md)

---

## 🛡️ Explainable Honeypot Defense System

The dataset contains a fraction of "honeypot" profiles with simulated or impossible attributes. Our system incorporates deterministic honeypot detection rules to heavily penalize synthetic bot accounts, protecting the final top 100 output from compromised data.

The defense engine actively flags:
* title-skill mismatches
* timeline inconsistencies
* assessment contradictions
* behavioral anomalies
* skill inflation

*(Detailed logic and rule thresholds are kept in the technical documentation).*

---

## 📁 Repository Directory Structure

```text
├── config.yaml                   # Global hyperparameter configuration
├── requirements.txt              # Package dependencies
├── submission_metadata.yaml      # Participant metadata file
├── main.py                       # Pipeline entrypoint
├── data/
│   ├── raw/                      # Input candidate dataset and job description
│   └── processed/                # Pre-parsed candidate caches (downloaded)
├── docs/
│   ├── architecture_one_pager.md # Architecture technical documentation
│   └── judge_faq.md              # Technical defense FAQs
├── src/                          # Core backend pipeline modules
├── sandbox/
│   ├── app.py                    # Streamlit Candidate Inspector
│   ├── requirements.txt          # Sandbox specific dependencies
│   ├── sample_data/              # Sample JSON candidates for Demo
│   └── utils/                    # Wrappers for sandbox engine
└── tools/
    └── download_artifacts.py     # Pulls model and cache tensors from mirrors
```

---

## 🎨 Streamlit Sandbox

An interactive Streamlit application is available in the `sandbox/` directory for demonstrating the ranking pipeline on a representative sample candidate pool.

To launch the sandbox app:
```bash
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```

**The sandbox:**
* accepts uploaded Job Descriptions (.docx, .txt, .md)
* allows interactive JD editing
* executes the same ranking pipeline used by the submission system
* evaluates candidates from a bundled demonstration dataset
* displays rankings, reasoning summaries, and honeypot indicators
* supports CSV export of ranked results
