# Individual Reflection - Lab 18: Production RAG

**Student:** Le Nguyen Thai Duong  
**Track:** K4 - Track 3A  
**Completion date:** October 4, 2026

## Part 1: Lecture Mapping

| Lecture concept | Module | Implemented function | Observation |
|----------------|--------|----------------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Splits complete sentences, embeds them with `all-MiniLM-L6-v2`, and starts a new chunk when adjacent cosine similarity falls below the configured threshold. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Creates parent chunks for answer context and smaller child chunks for retrieval. Each child keeps a valid `parent_id`. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Uses Markdown headings as logical boundaries and preserves tables, lists, and fenced code blocks. |
| Vietnamese lexical retrieval | M2 | `segment_vietnamese()` and `BM25Search` | `underthesea` compound words are expanded from `_` to spaces so queries such as “nghỉ phép” match correctly. |
| Dense retrieval | M2 | `DenseSearch.index()` and `.search()` | Uses `BAAI/bge-m3` vectors and Qdrant `query_points()` when available. A local fallback keeps development runnable without Qdrant. |
| Hybrid fusion | M2 | `reciprocal_rank_fusion()` | Combines BM25 and dense rankings without requiring their raw score scales to match. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Scores query-document pairs jointly with `BAAI/bge-reranker-v2-m3` and keeps the best three contexts. |
| RAGAS evaluation | M4 | `evaluate_ragas()` | Builds a Dataset and requests faithfulness, answer relevancy, context precision, and context recall. |
| Diagnostic tree | M4 | `failure_analysis()` | Finds the lowest metric for each question and maps it to a diagnosis and suggested fix. |
| Contextual enrichment | M5 | `contextual_prepend()` and `_enrich_single_call()` | Adds document/topic context before embedding and combines summary, HyQA, context, and metadata into one LLM call. |

The full run produced 97 hierarchical chunks from 25 Markdown documents,
enriched all 97 chunks, indexed them, reranked 20 test queries, and wrote
both baseline and production reports. The final scores were zero placeholders
because the local environment had no OpenAI key and RAGAS could not import
`langchain_community.chat_models.vertexai`; the retrieval artifacts were
still useful for manual diagnosis.

## Part 2: Challenges and Debugging

### 1. Model loading tried to access the network

The initial pipeline stalled on errors such as:

`[WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`

The Hugging Face client repeatedly tried `HEAD` requests even though model
weights were already cached. I changed the Sentence Transformers loaders to
use `local_files_only=True` and added deterministic local fallbacks. This
reduced the second end-to-end run from repeated network retries to about
38 seconds.

### 2. Optional dependencies were missing

The environment did not have `pypdf`, `underthesea`, `rank_bm25`, or
`qdrant-client`. PDF files were therefore skipped with an explicit warning,
while BM25 and Dense Search use local fallbacks when their optional
dependencies are unavailable. This keeps tests and local development
functional, while production can use the real libraries after installation.

### 3. RAGAS could not import its integration dependency

The evaluation failed with:

`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`

Instead of letting the pipeline crash, `evaluate_ragas()` returns zero
aggregate values plus one `EvalResult` per question. This preserves the
report shape and makes failure analysis possible. A real score run still
requires installing a compatible RAGAS/LangChain stack and configuring
`OPENAI_API_KEY`.

### 4. Version conflicts were not resolved automatically

The retrieval contexts often contained both the current and obsolete policy.
For example, the annual leave question retrieved v2023 (12 days) and v2024
(15 days), but the first answer used the old version. This showed that
semantic relevance alone is not enough for enterprise policy search.

The next fix is to extract `version`, `effective_date`, and `status` metadata
in M5, filter replaced documents in M2, and explicitly boost the latest
effective document during M3 reranking.

### 5. Retrieval and generation failures are different

Some multi-step questions retrieved all required facts but did not calculate
the final number, such as 85% of a Junior salary or a five-day pro-rata
late-payment fee. The diagnostic process must therefore distinguish:

- missing or wrongly ranked context: M1/M2/M3;
- incomplete arithmetic or synthesis: the answer-generation layer;
- unsupported claims: prompt/LLM settings and M4 faithfulness.

## Part 3: Action Plan for a Personal Project

### Project: Internal Vietnamese Operations Assistant

#### 1. Current state

- **Current pipeline:** Markdown and PDF policies are chunked, embedded,
  retrieved with hybrid search, reranked, and passed to an LLM.
- **Main bottlenecks:** obsolete policy versions can outrank current ones;
  multi-step numeric questions need explicit calculation; PDF extraction and
  evaluation dependencies make reproducibility difficult.

#### 2. Improvement plan

1. **Chunking strategy:** Use structure-aware plus hierarchical chunking.
   Keep headings and tables attached to their sections, and retrieve child
   chunks while returning the parent context.
2. **Search retrieval:** Use Vietnamese BM25 plus dense retrieval with RRF.
   BM25 protects exact policy numbers, dates, and abbreviations; dense search
   handles paraphrases.
3. **Reranking:** Use `BAAI/bge-reranker-v2-m3` for the top 20 candidates.
   Add metadata-aware penalties for `status=replaced` documents.
4. **Evaluation:** Run the four RAGAS metrics on a versioned golden test set.
   Add custom checks for exact numbers, negation, version selection, and
   arithmetic results.
5. **Enrichment:** Use one combined LLM call for contextual prepend,
   hypothetical questions, summaries, and metadata. Cache enrichment by
   document hash to avoid repeated API cost.

#### 3. Implementation timeline

- **Week 1:** Normalize PDF/Markdown ingestion, add metadata schemas, and
  build version-status filters.
- **Week 2:** Add hybrid retrieval, reranking, and tests for exact numbers,
  negation, and current-versus-obsolete policy pairs.
- **Week 3:** Add contextual enrichment and cache results by content hash.
- **Week 4:** Run RAGAS on a fixed evaluation set, inspect Bottom-5 failures,
  and add regression tests for every resolved issue.

The production target is at least 0.75 on faithfulness and context precision,
with p95 retrieval plus reranking latency below 500 ms before the LLM call.
