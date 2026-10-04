# Failure Analysis - Lab 18: Production RAG

**Student:** Le Nguyen Thai Duong
**Track:** K4 - Track 3A
**Run date:** October 4, 2026

## RAGAS Scores

The end-to-end run completed all 20 questions. The four RAGAS values are
currently zero because this environment has no `.env` OpenAI key and the
installed RAGAS package cannot import
`langchain_community.chat_models.vertexai`. The report therefore records
retrieval outputs and zero-score placeholders, rather than claiming real
quality scores.

| Metric | Naive Baseline | Production | Delta |
|--------|---------------:|-----------:|------:|
| Faithfulness | 0.0000 | 0.0000 | +0.0000 |
| Answer Relevancy | 0.0000 | 0.0000 | +0.0000 |
| Context Precision | 0.0000 | 0.0000 | +0.0000 |
| Context Recall | 0.0000 | 0.0000 | +0.0000 |

## Bottom-5 Failures

The cases below are selected from the 20-question run as representative
failures. Since RAGAS was unavailable, correctness is assessed manually by
comparing the generated answer and retrieved contexts with `test_set.json`.

### #1 - Annual leave version conflict

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** The current v2024 policy gives 15 paid days; v2023 gives 12
  days but has been replaced.
- **Got:** The first answer context came from `nghi_phep_nam_v2023.md` and
  stated 12 days. A v2024 context was also retrieved, but it was not used as
  the answer.
- **Is the answer correct?** No. It selected the obsolete version.
- **Do the contexts contain the answer?** Yes. The retrieved contexts contain
  both 12 days and 15 days, including the explicit replacement statement.
- **Does the question need rewriting?** Slightly. Adding “the current policy”
  or “v2024” would remove ambiguity, but the system should still resolve
  current-vs-obsolete versions for the original question.
- **Module to fix:** M2 and M3. Add version/status metadata filtering and
  reranker features that prefer effective documents; keep both versions only
  when explaining the conflict.
- **Worst metric in the placeholder report:** `faithfulness` (all metrics are
  zero because RAGAS did not run).

### #2 - Password rotation version conflict

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** The current v2.0 policy requires a change every 120 days;
  v1.0 required 90 days and is replaced.
- **Got:** The production answer selected the older v1.0 passage with 90 days
  in the first context, even though v2.0 was also retrieved.
- **Is the answer correct?** No. It reports an obsolete 90-day rule.
- **Do the contexts contain the answer?** Yes. Both 120 days and 90 days
  appear in the retrieved contexts.
- **Does the question need rewriting?** Not necessarily. “Current policy”
  would help, but “how often” is a valid query and version resolution is a
  retrieval/ranking responsibility.
- **Module to fix:** M2 metadata-aware hybrid search and M3 cross-encoder
  reranking. Penalize documents marked “replaced” and boost the latest
  effective date.
- **Worst metric in the placeholder report:** `faithfulness`.

### #3 - Multi-step junior salary calculation

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior maximum is 20,000,000 VND/month; 85% probation pay is
  17,000,000 VND/month.
- **Got:** The answer retrieved the probation percentage and salary table but
  did not explicitly perform the 85% calculation.
- **Is the answer correct?** Incomplete. The evidence is relevant, but the
  final numeric result is missing.
- **Do the contexts contain the answer?** Yes. The contexts contain both the
  Junior range and the 85% rule.
- **Does the question need rewriting?** It could be clearer as “Tính lương
  thử việc tối đa của Junior khi lương chính thức tối đa là bao nhiêu?”,
  although the original question reasonably implies the calculation.
- **Module to fix:** The answer-generation step after M3. Add a calculation
  instruction and require the model to combine all retrieved facts. M2/M3
  should also preserve both supporting contexts.
- **Worst metric in the placeholder report:** `faithfulness`.

### #4 - Purchase approval threshold

- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** An order above 50,000,000 VND needs CEO approval.
- **Got:** The first answer context came from the unrelated `tam_ung.md`
  approval section. The correct `mua_sam.md` table was present as a later
  context, but the answer did not use it.
- **Is the answer correct?** No. It does not answer the purchase approval
  question.
- **Do the contexts contain the answer?** Yes. The purchase table explicitly
  maps amounts above 50,000,000 VND to the CEO.
- **Does the question need rewriting?** No. The entity “mua thiết bị” and
  amount 55 million are sufficiently specific.
- **Module to fix:** M2 BM25/entity matching and M3 reranking. Preserve the
  document topic (`mua_sam`) and filter out semantically similar but
  different approval domains such as advances.
- **Worst metric in the placeholder report:** `faithfulness`.

### #5 - Advance payment pro-rata calculation

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt
  bao nhiêu?
- **Expected:** Five days late; 2% monthly on 15,000,000 VND is 300,000
  VND/month, approximately 50,000 VND pro-rata for five days.
- **Got:** The answer retrieved the 15-day deadline and 2% monthly fee, but did
  not calculate the 50,000 VND pro-rata amount.
- **Is the answer correct?** Incomplete. The source evidence is correct, but
  the requested calculation is absent.
- **Do the contexts contain the answer?** Mostly yes. They contain the
  deadline and fee formula; the arithmetic must be performed by generation.
- **Does the question need rewriting?** No. The dates and amount are clear.
- **Module to fix:** The answer-generation layer, with a numeric reasoning
  instruction and an arithmetic verification step. M4 should separately
  distinguish retrieval failure from generation/calculation failure.
- **Worst metric in the placeholder report:** `faithfulness`.

## Case Study

**Question selected:** Nhân viên được nghỉ bao nhiêu ngày phép năm?

**Error Tree walkthrough:**
1. **Is the output correct?** No. It chose 12 days from v2023 instead of
   15 days from the current v2024 policy.
2. **Is the context correct?** Partly. The top context is obsolete, while
   another retrieved context contains the correct v2024 rule.
3. **Is query rewriting required?** Helpful but not sufficient. The query can
   include “hiện hành” or “v2024”; the system should also resolve document
   status automatically.
4. **Where should it be fixed?** Add `effective_date`, `version`, and
   `status=replaced/current` metadata in M5/M2; use these fields during
   filtering and reranking in M3; instruct the answer model to mention the
   active version when conflicts remain.

**If there were one more hour, I would optimize:**

- Add a deterministic current-version filter before RRF.
- Add tests for every v2023/v2024 and v1/v2 conflict.
- Add arithmetic answer checks for percentage, threshold, and pro-rata
  questions.
- Install the missing RAGAS integration and run the real evaluation with an
  OpenAI key.
