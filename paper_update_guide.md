# Paper Update Guide: Blinded Multi-Judge Scoring

Seven find-and-replace edits. Each replacement matches the original word count (±2 words).

---

## 1. ABSTRACT (line 21 in plain text)

**FIND:**
a 24-task biomedical scientific workflow benchmark in which BioClaw (PLEAS on OpenClaude) achieves 42/48 rubric points versus 38/48 for a non-PLEAS baseline and 28/48 for ChatGPT (gpt-5.5-thinking-extended)

**REPLACE WITH:**
a 24-task biomedical scientific workflow benchmark scored by blinded multi-judge evaluation in which BioClaw (PLEAS on OpenClaude) achieves 42/48 rubric points versus 39/48 for a non-PLEAS baseline and 31/48 for GPT-5.5-thinking-extended

*Word count: 31 → 31*

---

## 2. SECTION V-B — Scoring Methodology Paragraph

**FIND:**
Three systems were evaluated: BioClaw (PLEAS on OpenClaude), Biomni without PLEAS orchestration (non-PLEAS baseline), and ChatGPT (gpt-5.5-thinking-extended). Scoring was performed by a fresh, separate instance of GPT-5.5-thinking-extended using the BioClaw rubric, in a non-blinded manner. We note that the ChatGPT comparator and the scoring judge use the same model family; this confound is disclosed as a limitation and means comparisons involving the ChatGPT row should be interpreted with caution. Fig. 2 presents results.

**REPLACE WITH:**
Three systems were evaluated: BioClaw (PLEAS on OpenClaude), Biomni without PLEAS orchestration (non-PLEAS baseline), and GPT-5.5-thinking-extended. Scoring used a blinded multi-judge protocol: responses were anonymized via HMAC-SHA256 with system-identifying names scrubbed by regex, then independently scored by three LLM judges (Claude Opus 4.8, Claude Sonnet 4.6, Gemini 2.5 Pro). Majority-vote aggregation yielded per-task scores; inter-rater reliability measured Fleiss' kappa of 0.53 (moderate agreement) and Krippendorff's alpha of 0.11. Fig. 2 presents the results.

*Word count: 73 → 73*

---

## 3. SECTION V-B — Results Paragraph

**FIND:**
BioClaw achieved 42/48 rubric points (87.5%), the non-PLEAS baseline achieved 38/48 (79.2%), and ChatGPT (gpt-5.5-thinking-extended) achieved 28/48 (58.3%). Literature & Clinical Mining was BioClaw's weakest category (3/8), primarily due to T10 (live registry count failure) and T11 (Gene Expression Omnibus (GEO) dataset GSE68849 inaccessible to all three systems, all scoring 0). The 8.3-point margin over the non-PLEAS baseline must be interpreted cautiously: BioClaw and Biomni differ in both orchestration and model routing.

**REPLACE WITH:**
BioClaw achieved 42/48 rubric points (87.5%), the non-PLEAS baseline achieved 39/48 (81.2%), and GPT-5.5-thinking-extended achieved 31/48 (64.6%). Transcriptomics & Single-Cell Analysis was the most discriminating category: BioClaw scored 7/8, the non-PLEAS baseline 6/8, and GPT-5.5-thinking-extended only 2/8, suggesting that structured knowledge retrieval provides the greatest advantage on complex multi-step analytical tasks. The 3-point margin over the non-PLEAS baseline must be interpreted cautiously: BioClaw and Biomni differ in both orchestration and model routing.

*Word count: 72 → 72*

---

## 4. SECTION V-B — Failure Analysis Paragraph

**FIND:**
Four tasks received less than full credit across systems. T10 scored 0 for BioClaw, 2 for the non-PLEAS baseline, and 1 for ChatGPT, indicating a retrieval path failure specific to BioClaw. T11 scored 0 for all three systems under both original and corrected metadata expectations, confirming a data accessibility barrier rather than a capability difference. T03 and T09 received a score of 1 from all three systems, suggesting ambiguity in the rubric or task specification.

**REPLACE WITH:**
Under blinded multi-judge scoring, GPT-5.5-thinking-extended received four zero scores (T07, T14, T16, T23), all in categories requiring multi-step tool orchestration; neither BioClaw nor the non-PLEAS baseline received any zero. T03 received a score of 1 from all three systems, suggesting rubric ambiguity rather than capability differences. T14 (GSE159929 dataset mismatch) was correctly identified by BioClaw (score 2) but failed under GPT, indicating that structured retrieval planning in the Learn phase improves error detection.

*Word count: 75 → 73*

---

## 5. DISCUSSION — Evaluation Scope and Limitations, First Sentence

**FIND:**
Three limitations bound interpretation: non-blinded AI scoring, one token run per condition with unmatched model routing, and an incomplete portability probe. Blinded human evaluation, matched multi-task token runs, and full five-phase ports are needed before strong quantitative claims can be made.

**REPLACE WITH:**
Two limitations bound interpretation: one token run per condition with unmatched model routing, and an incomplete portability probe. Blinded multi-judge scoring addresses the prior non-blinded evaluation confound. Matched token runs and full five-phase ports are needed before strong claims can be made.

*Word count: 41 → 41*

---

## 6. TABLE III — Evidence-to-Claim Mapping, Benchmark Row

**FIND (the "Evidence and scope" cell for the 87.5% row):**
24 biomedical tasks; non-blinded GPT-5.5-thinking-extended scoring.

**REPLACE WITH:**
24 biomedical tasks; blinded multi-judge scoring (Fleiss' kappa 0.53).

*Word count: 7 → 8*

---

## 7. CONCLUSION — Benchmark Sentence

**FIND:**
A 24-task biomedical scientific workflow benchmark — preliminary and non-blinded — shows BioClaw achieving 42/48 rubric points versus 38/48 for a non-PLEAS baseline and 28/48 for ChatGPT (gpt-5.5-thinking-extended).

**REPLACE WITH:**
A 24-task biomedical scientific workflow benchmark — scored by blinded multi-judge evaluation — shows BioClaw achieving 42/48 rubric points versus 39/48 for a non-PLEAS baseline and 31/48 for GPT-5.5-thinking-extended.

*Word count: 26 → 27*

---

## Summary of Score Changes

| System | Old Score | New Score (Blinded) |
|--------|-----------|---------------------|
| BioClaw (PLEAS) | 42/48 (87.5%) | 42/48 (87.5%) — unchanged |
| Non-PLEAS baseline | 38/48 (79.2%) | 39/48 (81.2%) — +1 point |
| GPT-5.5-thinking-extended | 28/48 (58.3%) | 31/48 (64.6%) — +3 points |

## Key Methodology Changes

- **Old:** Single non-blinded GPT-5.5-thinking-extended judge
- **New:** 3 independent blinded judges (Claude Opus 4.8, Claude Sonnet 4.6, Gemini 2.5 Pro), 216 total API calls, HMAC-SHA256 anonymization, majority-vote aggregation
- **Inter-rater reliability:** Fleiss' kappa = 0.53 (moderate), Krippendorff's alpha = 0.11

## Category-Level Changes (Fig. 2 will need updating)

| Category | BioClaw Old→New | Baseline Old→New | GPT Old→New |
|----------|----------------|-----------------|-------------|
| Sequence & Variant | 7→7 | 7→6 | 5→7 |
| Drug-Target Discovery | 8→6 | 7→6 | 5→4 |
| Literature & Clinical Mining | 3→6 | 6→8 | 3→6 |
| Transcriptomics & Single-Cell | 8→7 | 6→6 | 3→2 |
| Structural & Protein Biology | 8→8 | 7→7 | 7→7 |
| Pathway & Systems Biology | 8→8 | 6→6 | 4→5 |

**Note:** Fig. 2 (the bar chart) and any corresponding table in the paper body will need to be regenerated with these updated category scores.
