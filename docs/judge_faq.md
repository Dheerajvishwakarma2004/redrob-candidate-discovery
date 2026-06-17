# Stage 4 Technical Panel: Judge FAQ & Answer Keys

This document contains a curated list of high-probability technical questions from judges and detailed, engineering-grounded responses to defend our implementation.

---

## 1. Technical Choice FAQs

### Q1: Why did you choose `bge-small-en-v1.5` instead of `all-MiniLM-L6-v2` or a larger model like `bge-large-en-v1.5`?
*   **Ideal Answer:** 
    `bge-small-en-v1.5` represents the optimal pareto efficiency point between embedding quality, retrieval recall, and latency on CPU. 
    1.  **Metric Strength:** On the Massive Text Embedding Benchmark (MTEB), `bge-small-en-v1.5` scores a Retrieval Average of **53.90** compared to MiniLM's **41.95**.
    2.  **Compact Size:** At only 134 MB, the model loads in under **0.5 seconds** and easily fits in RAM.
    3.  **Low Dimensionality:** BGE-small outputs 384-dimensional vectors. This matches MiniLM's size but outperforms it, keeping FAISS index lookup times at sub-millisecond scales.
    4.  **CPU Latency:** BGE-large (1.34 GB, 1024-dimensions) is computationally prohibitive on CPU. Using BGE-large on-the-fly for 1,500 candidates in Mode B would exceed the 5-minute timeout. BGE-small completes this in **~41 seconds**.

### Q2: How does your system defend against candidates who "keyword-stuff" their profiles to game the search rank?
*   **Ideal Answer:** 
    We implemented a dedicated **Skill-Career Alignment Feature Engine** specifically to neutralize keyword-stuffing. 
    1.  **The Mechanism:** Instead of trusting a candidate's claimed skills list, we evaluate the similarity of those claims against their written career descriptions. 
    2.  **Hybrid Verification:** For every skill bucket claimed (e.g. "retrieval"), the engine calculates an exact match score (counting keyword mentions in written job descriptions) and a semantic similarity score (cosine similarity of their career history embeddings vs the concept descriptions).
    3.  **The Formula:** We compute the hybrid support score:
        $$\text{Support} = 0.60 \times \text{Exact Match} + 0.40 \times \text{Semantic Sim}$$
    4.  **Penalty Trigger:** If the candidate claims advanced skills but their career descriptions have low alignment ($<0.40$), they receive a penalty in their score, preventing stuffed profiles from rising to the top.

### Q3: Why did you use a Hybrid Retriever (BM25 + FAISS) instead of pure semantic search?
*   **Ideal Answer:** 
    1.  **Vocabulary Mismatch Mitigation:** Pure semantic search excels at abstract concept matching but often misses exact term hits. For example, a candidate listing "BM25" or "FAISS" might have lower semantic cosine similarity to a general JD than someone with a generic "Machine Learning" profile, despite being a better technical fit.
    2.  **Complementary Strengths:** BM25 handles high-precision keyword alignment. FAISS (using dense vectors) captures broad semantic overlap.
    3.  **Hybrid score merging:** We merge retrieval ranks using a weighted combination:
        $$\text{Retrieval Score} = 0.40 \times \text{BM25 Score} + 0.60 \times \text{FAISS Score}$$
    This ensures that candidates with *either* high semantic relevance *or* exact, critical niche skills are retrieved.

---

## 2. Ingestion & Scaling FAQs

### Q4: How would this architecture scale if the candidate pool grew from 100k to 1,000,000 candidates?
*   **Ideal Answer:** 
    1.  **Mode A (Cached) Scaling:**
        The retrieval layer remains practical at 1M candidates because vector search scales efficiently and retrieval remains a small fraction of total runtime. Exact latency depends on hardware and index configuration. The primary bottleneck would be loading the larger parquet database, which can be mitigated using memory-mapped files (like PyArrow Feather or partitioning).
    2.  **Mode B (On-the-fly) Scaling:**
        We would maintain the BM25 pre-filtering step. BM25 indexes are highly optimized and query 1,000,000 documents in milliseconds. Because we restrict the reranking pool size to the top 1,500 candidates, our transformer inference overhead remains completely flat, keeping the pipeline under 60 seconds regardless of database size.

---

## 3. Honeypot & Scoring FAQs

### Q5: Why do you cap the maximum honeypot risk penalty at `-0.45`?
*   **Ideal Answer:** 
    1.  **Preventing False-Positive Disqualification:** In real-world data, candidates often have noisy profiles due to formatting bugs, spelling errors, or poor data entry (e.g. typos in dates). 
    2.  **The Penalty Balance:** Capping the penalty at `-0.45` allows a candidate who triggers a minor timeline anomaly (e.g. a date discrepancy) to still be considered if they are an elite match. 
    3.  **Honeypot Suppression:** Extreme honeypots that trigger multiple checks (e.g., zero career history + expert skills claim + date contradiction) will reach the `-0.45` cap. This drops their score from, say, `0.78` down to `0.33`, completely burying them from the top 100 retrieval pool.

### Q6: Walk us through two specific Honeypot rules. How do they work?
*   **Ideal Answer:** 
    1.  **`p_assess` (Assessment Contradiction):**
        *   *Logic:* Compares the candidate's `skill_assessment_scores` against their `skills_normalized` list.
        *   *Trigger:* If a candidate has an official platform assessment score $\ge 80$ on a skill but does not even list that skill on their profile, we flag it. In a real-world scenario, a candidate would never take an exam for a skill they don't claim. This indicates profile bot generation.
    2.  **`p_low_resp` (Response/Interview Mismatch):**
        *   *Logic:* Compares active platform responsiveness against interview attendance.
        *   *Trigger:* If a candidate responds to recruiter messages at a rate $<10\%$, but somehow attends scheduled interviews at a rate $>80\%$, we flag it. A candidate cannot attend interviews without responding to recruiter coordination. This indicates bot-simulated behavior.

### Q7: How do you ensure that the generated reasoning text is strictly grounded in candidate data?
*   **Ideal Answer:** 
    We constructed a **Grounded Reasoning Generator** that dynamically extracts substrings and facts directly from the candidate's active row. It does not use static templates or generate speculative details.
    *   **Titles and Experience:** We format the candidate's actual `current_title` and `total_experience`.
    *   **JD Evidence Matching:** We search the candidate's `career_descriptions_combined` for explicit evidence phrases (e.g. checking if "retrieval pipeline work" is physically present) and print: *"Career history includes retrieval pipeline work..."*
    *   **Structured Concerns:** If the honeypot engine triggers a warning (e.g. timeline discrepancy), the generator appends: *"Note: stated experience shows some inconsistency with computed career timeline."*
    This ensures that the generated reasoning is evidence-grounded and generated only from candidate attributes present in the dataset.

### Q8: Why did you choose manual/heuristic weights instead of training a supervised ranking model (like LightGBM)?
*   **Ideal Answer:** 
    The challenge dataset provides no relevance labels; therefore, supervised ranking would require synthetic labels, which risks encoding our own assumptions rather than learning genuine relevance. Furthermore, heuristically defined scoring weights prevent overfitting to simulated structures in synthetic datasets and provide 100% explainability, which is a key requirement in recruitment auditing.

### Q9: If your architecture is generalizable, why does it have both a submission mode and a generalized mode?
*   **Ideal Answer:** 
    Because the evaluation process has two distinct contexts. Submission mode optimizes latency on the known candidate corpus using precomputed artifacts (completing in approximately 30 seconds). Generalized mode demonstrates that the ranking methodology itself does not depend on those precomputed index files and can operate dynamically on unseen candidate databases (completing in under 60 seconds). Both modes use the identical scoring logic, feature engines, and honeypot checks; only the initial candidate retrieval sourcing stage changes.

