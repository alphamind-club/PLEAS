# Biomni Baseline - Consolidated Answer

Source: GPT-5.5 Thinking Extended strict grading of the submitted consolidated
Biomni answer against the BioClaw 24-task rubric. This was not author-scored.

Reported score: 38/48 = 79.2%.

Assumption from the grading note: stated database/API usage is accepted, although
raw API logs are not included. Tasks are penalized when the answer changes the
requested query or dataset, or omits required fields such as GO IDs, log2 fold
changes, or full URLs.

| Task | Score | Rationale |
|------|------:|-----------|
| T01 | 2 | Correct TP53 accession, gene, organism. |
| T02 | 2 | KRAS identified, 100% identity, BLAST-style statistics included. |
| T03 | 1 | Pathogenic classification and conditions were correct, but variant ID appears wrong under the rubric, which expects VCV000017661 rather than VCV000017677. |
| T04 | 2 | EGFR coordinates were within the accepted range; GRCh38 was stated. |
| T05 | 2 | Count >=4 and 3+ correct EGFR drugs were listed. Count may include salt forms, but rubric accepts >=4. |
| T06 | 2 | PubChem formula, molecular weight, and SMILES were all correct. |
| T07 | 1 | Provided 3 diseases and scores, but melanoma was not ranked #1 and the top 3 did not match the benchmark's expected cancer-focused associations. |
| T08 | 2 | 2+ correct EGFR/HER2 dual inhibitors with indications. |
| T09 | 1 | Provided count and PMID, but changed the query from the exact requested phrase to individual TIAB terms. |
| T10 | 2 | Count >30 and valid-looking NCT example/title. |
| T11 | 0 | Regrade needed. Original GPT-5.5 Thinking Extended grading marked this failed under an older incorrect rubric. Raw output should be regraded against the corrected NCBI GEO metadata before final C4 scoring. |
| T12 | 2 | Nature and citation count well above threshold. |
| T13 | 2 | 5+ plausible CD8 cytotoxic T-cell markers including CD8A/CD8B. |
| T14 | 1 | Provided plausible COVID ISGs, but no log2 fold changes and did not actually solve the requested GEO2R comparison. |
| T15 | 2 | p53 signaling and cell cycle were present with adjusted p-values. |
| T16 | 1 | AT2 was included, but the second cell type was questionable under benchmark expectations, and source citation was not specific enough. |
| T17 | 1 | Mean pLDDT was given, but only file name/model was listed, not the full structure file URL. |
| T18 | 2 | 5 TP53 partners with scores >700; acceptable even if not the exact expected examples. |
| T19 | 2 | Valid high-resolution CDK2 inhibitor complex with ligand. |
| T20 | 2 | 5+ plausible EGFR phosphorylation sites listed. |
| T21 | 2 | Gene count in accepted range and 5+ pathway genes. |
| T22 | 2 | Correct Reactome top-level pathway and stable ID. |
| T23 | 1 | Degree values were reported and TP53 was top, but the answer changed the requested 10-gene network to a 17-gene network. |
| T24 | 1 | Count was reported and terms were plausible, but GO IDs were missing. |
