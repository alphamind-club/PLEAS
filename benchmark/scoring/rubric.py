"""
Structured 24-task biomedical benchmark rubric.

Each task is a dataclass carrying the full grading specification from
CaseStudy/bioclaw_24tasks/bioclaw_24_tasks.txt  (IEEE IRI 2026 — C4).
The rubric text is embedded verbatim so that judge prompts are
self-contained and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    category: str
    difficulty: str
    task_text: str
    expected_output: str
    scoring_criteria: str


TASKS: list[BenchmarkTask] = [
    # ── Category A: Sequence & Variant Analysis ─────────────────────────────
    BenchmarkTask(
        task_id="T01",
        category="Sequence & Variant Analysis",
        difficulty="Easy",
        task_text=(
            "Using the UniProt database, retrieve the canonical UniProt accession "
            "number and gene name for human TP53 protein. State the organism, "
            "accession, and gene name."
        ),
        expected_output=(
            "Organism: Homo sapiens\n"
            "Accession: P04637\n"
            "Gene name: TP53"
        ),
        scoring_criteria=(
            "SCORE 2: P04637 returned, gene name TP53, organism Homo sapiens, UniProt cited.\n"
            "SCORE 1: Correct accession but wrong gene name or organism, OR correct answer "
            "but source not specified or hallucinated.\n"
            "SCORE 0: Wrong accession or no answer."
        ),
    ),
    BenchmarkTask(
        task_id="T02",
        category="Sequence & Variant Analysis",
        difficulty="Easy",
        task_text=(
            "The following protein sequence is a fragment of a human protein. Use NCBI "
            "BLAST (blastp) to identify the protein and return: (1) the protein name, "
            "(2) the gene symbol, and (3) the percent identity of the top hit.\n\n"
            "Sequence (FASTA):\n>query\nMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY"
        ),
        expected_output=(
            "Protein: GTPase KRas (or KRAS proto-oncogene GTPase)\n"
            "Gene: KRAS\n"
            "Identity: >=95% (exact value depends on BLAST run; accept 95-100%)"
        ),
        scoring_criteria=(
            "SCORE 2: KRAS identified, identity >= 95%, BLAST used.\n"
            "SCORE 1: Correct protein but identity not reported, OR identity reported "
            "but gene symbol missing.\n"
            "SCORE 0: Wrong protein or no BLAST call made."
        ),
    ),
    BenchmarkTask(
        task_id="T03",
        category="Sequence & Variant Analysis",
        difficulty="Medium",
        task_text=(
            "Query the ClinVar database to find the clinical significance classification "
            "for the BRCA1 variant NM_007294.4(BRCA1):c.5266dupC (also known as 5382insC). "
            "Return the variant ID, clinical significance, and the condition(s) listed."
        ),
        expected_output=(
            "Variant ID: 17661 (ClinVar accession VCV000017661)\n"
            "Clinical significance: Pathogenic\n"
            "Condition: Hereditary breast and ovarian cancer syndrome"
        ),
        scoring_criteria=(
            "SCORE 2: Pathogenic returned, ClinVar accession or variant ID cited, condition named.\n"
            "SCORE 1: Pathogenic returned but no variant ID or condition listed.\n"
            "SCORE 0: Wrong classification or ClinVar not queried."
        ),
    ),
    BenchmarkTask(
        task_id="T04",
        category="Sequence & Variant Analysis",
        difficulty="Medium",
        task_text=(
            "Use Ensembl REST API to retrieve the chromosome location (chromosome, start, "
            "end, strand) of the human gene EGFR (Ensembl gene ID: ENSG00000146648)."
        ),
        expected_output=(
            "Chromosome: 7\n"
            "Start: 55,019,017\n"
            "End: 55,211,628\n"
            "Strand: +1 (forward)\n"
            "Assembly: GRCh38"
        ),
        scoring_criteria=(
            "SCORE 2: Chromosome 7, correct start/end within +/-1000 bp, Ensembl used.\n"
            "SCORE 1: Chromosome correct but coordinates significantly wrong, OR coordinates "
            "correct but assembly not specified.\n"
            "SCORE 0: Wrong chromosome or no API call."
        ),
    ),

    # ── Category B: Drug-Target Discovery ───────────────────────────────────
    BenchmarkTask(
        task_id="T05",
        category="Drug-Target Discovery",
        difficulty="Easy",
        task_text=(
            "Query the ChEMBL database to find how many approved (Phase 4) small-molecule "
            "drugs target the protein EGFR (ChEMBL target ID: CHEMBL203). Report the count "
            "and name at least 3 of those drugs."
        ),
        expected_output=(
            "Count: >= 4 approved drugs targeting EGFR\n"
            "Examples: Gefitinib, Erlotinib, Osimertinib, Afatinib, Lapatinib"
        ),
        scoring_criteria=(
            "SCORE 2: Count >= 4, at least 3 correct drug names, ChEMBL cited.\n"
            "SCORE 1: Count correct but fewer than 3 drug names, OR 3+ drugs named without "
            "citing ChEMBL or using a hallucinated count.\n"
            "SCORE 0: Wrong count (< 2) or no drugs named."
        ),
    ),
    BenchmarkTask(
        task_id="T06",
        category="Drug-Target Discovery",
        difficulty="Easy",
        task_text=(
            "Use the PubChem database to retrieve the molecular formula, molecular weight, "
            "and canonical SMILES for Imatinib (CID: 5291)."
        ),
        expected_output=(
            "Molecular formula: C29H31N7O\n"
            "Molecular weight: 493.6 g/mol (accept 493.0-494.0)\n"
            "SMILES: Cn1cc(nc1-c1ccc(cc1)NC(=O)c1ccc(cc1)CN1CCN(CC1)C)-c1cccnc1 "
            "(or equivalent canonical form)"
        ),
        scoring_criteria=(
            "SCORE 2: Formula C29H31N7O, weight 493-494, SMILES returned, PubChem cited.\n"
            "SCORE 1: Two of three properties correct.\n"
            "SCORE 0: All wrong or no PubChem call."
        ),
    ),
    BenchmarkTask(
        task_id="T07",
        category="Drug-Target Discovery",
        difficulty="Medium",
        task_text=(
            "Using the Open Targets Platform API, retrieve the top 3 disease associations "
            "for the target BRAF (Ensembl ID: ENSG00000157764), ranked by overall "
            "association score. Report the disease name and score for each."
        ),
        expected_output=(
            "1. Melanoma — score > 0.85\n"
            "2. Colorectal cancer — score > 0.70\n"
            "3. Thyroid carcinoma — score > 0.65\n"
            "(Accept any top-3 where melanoma is #1 and all three are known BRAF cancers)"
        ),
        scoring_criteria=(
            "SCORE 2: Melanoma in top 3 (ideally #1), 3 diseases listed with scores, API cited.\n"
            "SCORE 1: Melanoma present but ranked incorrectly, or fewer than 3 diseases, "
            "or scores missing.\n"
            "SCORE 0: Melanoma absent or Open Targets not used."
        ),
    ),
    BenchmarkTask(
        task_id="T08",
        category="Drug-Target Discovery",
        difficulty="Hard",
        task_text=(
            "Query the DrugBank or ChEMBL database to find all drugs that target both "
            "EGFR and HER2 (ERBB2) simultaneously (dual inhibitors). List at least "
            "2 confirmed dual inhibitors and their indication."
        ),
        expected_output=(
            "Lapatinib — indicated for HER2+ breast cancer (EGFR+HER2 dual inhibitor)\n"
            "Neratinib — indicated for HER2+ breast cancer\n"
            "Afatinib — indicated for NSCLC (EGFR/HER2/HER4 inhibitor)\n"
            "(Accept any 2 of these 3)"
        ),
        scoring_criteria=(
            "SCORE 2: 2+ correct dual inhibitors with correct indications, source cited.\n"
            "SCORE 1: 2 correct names but indications wrong or missing.\n"
            "SCORE 0: Fewer than 2 correct or no database used."
        ),
    ),

    # ── Category C: Literature & Clinical Mining ────────────────────────────
    BenchmarkTask(
        task_id="T09",
        category="Literature & Clinical Mining",
        difficulty="Easy",
        task_text=(
            'Use the PubMed API (NCBI E-utilities) to find the total number of publications '
            'returned for the search query: "CRISPR Cas9 cancer therapy"[Title/Abstract] '
            "published between 2020 and 2024. Report the count and the PMID of the most "
            "recent article returned."
        ),
        expected_output=(
            "Count: > 500 (exact count varies by query date; accept any count > 300)\n"
            "Most recent PMID: any valid PMID published 2023-2024"
        ),
        scoring_criteria=(
            "SCORE 2: Count > 300, valid PMID from 2023-2024 returned, PubMed E-utilities used.\n"
            "SCORE 1: Count plausible but PMID missing or not verified as recent, OR correct "
            "PMID but count not reported.\n"
            "SCORE 0: Count < 100 (clearly wrong) or PubMed not used."
        ),
    ),
    BenchmarkTask(
        task_id="T10",
        category="Literature & Clinical Mining",
        difficulty="Easy",
        task_text=(
            'Query ClinicalTrials.gov API to find the number of currently RECRUITING '
            'clinical trials for "triple negative breast cancer" in the United States. '
            "Report the count and the NCT ID + title of one active trial."
        ),
        expected_output=(
            "Count: > 50 recruiting trials (accept any count > 30)\n"
            "Example: Any valid NCT ID for a recruiting TNBC trial in the US"
        ),
        scoring_criteria=(
            "SCORE 2: Count > 30, valid NCT ID provided, ClinicalTrials.gov API used.\n"
            "SCORE 1: Count plausible but no NCT ID, OR NCT ID given but count missing.\n"
            "SCORE 0: Count < 10 or source hallucinated."
        ),
    ),
    BenchmarkTask(
        task_id="T11",
        category="Literature & Clinical Mining",
        difficulty="Medium",
        task_text=(
            "Use the NCBI GEO database to retrieve the title, organism, and number of "
            "samples for dataset GSE68849."
        ),
        expected_output=(
            "Dataset: GDS6063 / Series GSE68849\n"
            "Title: Influenza A effect on plasmacytoid dendritic cells "
            "(series title: Impact of influenza A on human plasmacytoid "
            "dendritic cells (pDC) gene expression)\n"
            "Organism: Homo sapiens\n"
            "Samples: 10 samples\n"
            "Type: Expression profiling by array"
        ),
        scoring_criteria=(
            "SCORE 2: Correct title (partial match OK), organism Homo sapiens, sample count 10, GEO cited.\n"
            "SCORE 1: Title or organism correct but sample count wrong or missing.\n"
            "SCORE 0: Wrong dataset or GEO not used."
        ),
    ),
    BenchmarkTask(
        task_id="T12",
        category="Literature & Clinical Mining",
        difficulty="Medium",
        task_text=(
            "Using the Europe PMC API, find the total citation count for the paper with "
            "DOI: 10.1038/s41586-021-03819-2 (AlphaFold2 paper). Report the citation "
            "count and the journal name."
        ),
        expected_output=(
            "DOI: 10.1038/s41586-021-03819-2\n"
            "Journal: Nature\n"
            "Citations: > 10,000 (as of 2025; accept any count > 5,000)"
        ),
        scoring_criteria=(
            "SCORE 2: Citation count > 5,000, journal = Nature, Europe PMC used.\n"
            "SCORE 1: Citation count plausible but journal wrong, or journal correct "
            "but count not retrieved from API.\n"
            "SCORE 0: Citation count < 1,000 or source not used."
        ),
    ),

    # ── Category D: Transcriptomics & Single-Cell ──────────────────────────
    BenchmarkTask(
        task_id="T13",
        category="Transcriptomics & Single-cell",
        difficulty="Easy",
        task_text=(
            "Query the CellMarker 2.0 database (or equivalent) to retrieve the canonical "
            "marker genes for human CD8+ cytotoxic T cells. List at least 5 marker genes."
        ),
        expected_output=(
            "Any 5 of: CD8A, CD8B, GZMB, PRF1, IFNG, TBX21, EOMES, NKG7, GNLY, "
            "KLRG1, LAG3, PDCD1"
        ),
        scoring_criteria=(
            "SCORE 2: 5+ correct markers listed, CD8A or CD8B included, source cited.\n"
            "SCORE 1: 3-4 correct markers with CD8A/B included.\n"
            "SCORE 0: Fewer than 3 correct or CD8A/B absent."
        ),
    ),
    BenchmarkTask(
        task_id="T14",
        category="Transcriptomics & Single-cell",
        difficulty="Medium",
        task_text=(
            "Using the GEO2R tool or NCBI GEO API, retrieve the top 5 differentially "
            "expressed genes (by adjusted p-value) from GEO dataset GSE159929, comparing "
            "COVID-19 patients vs healthy controls. Report gene symbol and log2 fold change."
        ),
        expected_output=(
            "Top DEGs include: IFI27, ISG15, IFI44L, RSAD2, MX1\n"
            "(Accept any 5 interferon-stimulated genes with |log2FC| > 1)"
        ),
        scoring_criteria=(
            "SCORE 2: 5 genes listed, >= 3 are known interferon-stimulated genes, FC values given.\n"
            "SCORE 1: 5 genes listed but < 3 are ISGs, or FC values missing.\n"
            "SCORE 0: Genes are not biologically plausible for COVID-19 vs healthy."
        ),
    ),
    BenchmarkTask(
        task_id="T15",
        category="Transcriptomics & Single-cell",
        difficulty="Medium",
        task_text=(
            "Given the following gene list, use the Enrichr API or KEGG REST API to "
            "identify the top 3 enriched KEGG pathways. Report pathway name and adjusted p-value.\n\n"
            "Gene list: TP53, BRCA1, CHEK2, ATM, MDM2, CDKN1A, PTEN, RB1, CCND1, CDK4"
        ),
        expected_output=(
            "p53 signaling pathway — adjusted p < 0.001\n"
            "Cell cycle — adjusted p < 0.001\n"
            "Pathways in cancer — adjusted p < 0.01\n"
            "(Accept any ordering where p53/cell-cycle appear in top 3)"
        ),
        scoring_criteria=(
            "SCORE 2: p53 signaling and Cell cycle both in top 3, p-values reported, API used.\n"
            "SCORE 1: One of the two expected pathways in top 3, or both present but "
            "p-values missing.\n"
            "SCORE 0: Neither expected pathway in top 3."
        ),
    ),
    BenchmarkTask(
        task_id="T16",
        category="Transcriptomics & Single-cell",
        difficulty="Hard",
        task_text=(
            "Use the Human Cell Atlas (HCA) Data Portal API or a published scRNA-seq resource "
            "to identify which cell types in the human lung express ACE2 at the highest level. "
            "Report the top 2 cell types and the evidence source (dataset or publication DOI)."
        ),
        expected_output=(
            "Top cell types: AT2 cells (alveolar type 2), club cells / ciliated cells\n"
            "Key paper: Ziegler et al. Cell 2020 or equivalent "
            "(DOI: 10.1016/j.cell.2020.04.035)"
        ),
        scoring_criteria=(
            "SCORE 2: AT2 cells identified as top, second cell type biologically plausible, "
            "DOI or dataset cited.\n"
            "SCORE 1: AT2 cells identified but second cell type missing or wrong, or no citation.\n"
            "SCORE 0: AT2 cells not identified."
        ),
    ),

    # ── Category E: Structural & Protein Biology ───────────────────────────
    BenchmarkTask(
        task_id="T17",
        category="Structural & Protein Biology",
        difficulty="Easy",
        task_text=(
            "Use the AlphaFold DB API to retrieve the predicted structure confidence (mean "
            "pLDDT score) for human EGFR protein (UniProt: P00533). Report the mean pLDDT "
            "and the URL of the structure file."
        ),
        expected_output=(
            "UniProt: P00533\n"
            "Mean pLDDT: ~80-92 (accept 70-95 as plausible range)\n"
            "File URL: https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v4.cif "
            "(or equivalent versioned URL)"
        ),
        scoring_criteria=(
            "SCORE 2: pLDDT value in range 70-95, correct UniProt ID, AlphaFold DB URL returned.\n"
            "SCORE 1: pLDDT value given but URL missing, or URL correct but pLDDT not computed.\n"
            "SCORE 0: Wrong protein or AlphaFold DB not used."
        ),
    ),
    BenchmarkTask(
        task_id="T18",
        category="Structural & Protein Biology",
        difficulty="Easy",
        task_text=(
            "Query the STRING database API to retrieve the top 5 interaction partners of "
            "human TP53 (STRING ID: 9606.ENSP00000269305) with a combined score > 700. "
            "Report the partner gene name and score."
        ),
        expected_output=(
            "Any 5 of:\n"
            "MDM2 — score > 900\n"
            "MDM4 — score > 800\n"
            "CDKN1A — score > 800\n"
            "ATM — score > 800\n"
            "PTEN — score > 700\n"
            "CHEK2 — score > 700"
        ),
        scoring_criteria=(
            "SCORE 2: 5 partners listed, all scores > 700, all are known TP53 interactors, STRING cited.\n"
            "SCORE 1: 3-4 correct partners, or 5 partners but 1-2 are not known interactors.\n"
            "SCORE 0: Fewer than 3 correct or STRING not used."
        ),
    ),
    BenchmarkTask(
        task_id="T19",
        category="Structural & Protein Biology",
        difficulty="Medium",
        task_text=(
            "Use the RCSB PDB API to retrieve the PDB structure with the highest resolution "
            "for human CDK2 kinase in complex with a small molecule inhibitor. Report: "
            "(1) the PDB ID, (2) resolution in Angstroms, and (3) the ligand name."
        ),
        expected_output=(
            "PDB ID: Any PDB entry for CDK2 with resolution < 1.5 A\n"
            "Resolution: < 1.5 A (many CDK2 structures exist at 1.2-1.4 A)\n"
            "Ligand: Any named small molecule (not water)\n"
            "Example: 1AQ1 at 1.49 A with ligand ATP, or equivalent"
        ),
        scoring_criteria=(
            "SCORE 2: Valid PDB ID for CDK2, resolution < 2.0 A, ligand named, RCSB PDB API used.\n"
            "SCORE 1: Valid PDB ID for CDK2 but resolution > 2.0 A or ligand missing.\n"
            "SCORE 0: Wrong protein or no PDB API call."
        ),
    ),
    BenchmarkTask(
        task_id="T20",
        category="Structural & Protein Biology",
        difficulty="Hard",
        task_text=(
            "Use UniProt or PhosphoSitePlus to retrieve all confirmed phosphorylation sites "
            "on human EGFR (UniProt: P00533). Report at least 5 phosphorylation sites "
            "(residue + position) and the kinase responsible (if annotated)."
        ),
        expected_output=(
            "Any 5 of:\n"
            "Y992 — phosphorylated (autophosphorylation)\n"
            "Y1045 — phosphorylated (CBL binding site)\n"
            "Y1068 — phosphorylated (GRB2 binding)\n"
            "Y1086 — phosphorylated\n"
            "Y1148 — phosphorylated\n"
            "Y1173 — phosphorylated (major autophosphorylation site)"
        ),
        scoring_criteria=(
            "SCORE 2: 5+ correct phosphorylation sites, positions match known EGFR sites, source cited.\n"
            "SCORE 1: 3-4 correct sites, or 5 sites but positions are off by > 5 residues.\n"
            "SCORE 0: Fewer than 3 correct sites or source hallucinated."
        ),
    ),

    # ── Category F: Pathway & Systems Biology ──────────────────────────────
    BenchmarkTask(
        task_id="T21",
        category="Pathway & Systems Biology",
        difficulty="Easy",
        task_text=(
            "Use the KEGG REST API to retrieve all genes in the human insulin signaling "
            "pathway (KEGG pathway ID: hsa04910). Report the total gene count and list "
            "5 genes in the pathway."
        ),
        expected_output=(
            "Pathway: hsa04910 — Insulin signaling pathway\n"
            "Gene count: ~137 genes (accept 100-160)\n"
            "Examples (any 5): INSR, IRS1, IRS2, PIK3CA, AKT1, FOXO1, GSK3B, "
            "GLUT4 (SLC2A4)"
        ),
        scoring_criteria=(
            "SCORE 2: Gene count 100-160, 5 correct pathway members, KEGG API used.\n"
            "SCORE 1: Gene count correct but < 5 members, or 5 members but count wrong.\n"
            "SCORE 0: Gene count < 50 or genes not from this pathway."
        ),
    ),
    BenchmarkTask(
        task_id="T22",
        category="Pathway & Systems Biology",
        difficulty="Easy",
        task_text=(
            "Query the Reactome API to find the top-level pathway that contains the event "
            '"WNT ligand biogenesis and trafficking" (Reactome ID: R-HSA-3238698). '
            "Report the top-level pathway name and its Reactome stable ID."
        ),
        expected_output=(
            "Top-level pathway: Signal Transduction\n"
            "Stable ID: R-HSA-162582"
        ),
        scoring_criteria=(
            'SCORE 2: "Signal Transduction" returned, R-HSA-162582 cited, Reactome API used.\n'
            "SCORE 1: Correct pathway name but stable ID missing or wrong.\n"
            "SCORE 0: Wrong pathway or Reactome not used."
        ),
    ),
    BenchmarkTask(
        task_id="T23",
        category="Pathway & Systems Biology",
        difficulty="Medium",
        task_text=(
            "Using the STRING API, build a PPI (protein-protein interaction) network for "
            "the following 10 hub gene candidates in glioblastoma and identify the top 3 "
            "nodes by degree (number of interactions at score > 700).\n\n"
            "Gene list: EGFR, TP53, PTEN, IDH1, CDKN2A, MDM2, RB1, PIK3CA, VEGFA, MET"
        ),
        expected_output=(
            "Top 3 by degree (approximate, accept any order where EGFR/TP53/MDM2 appear):\n"
            "1. EGFR — degree >= 5\n"
            "2. TP53 — degree >= 5\n"
            "3. MDM2 — degree >= 4"
        ),
        scoring_criteria=(
            "SCORE 2: EGFR and TP53 in top 3, degree values reported, STRING API used.\n"
            "SCORE 1: One of EGFR/TP53 in top 3, or both present but degree values missing.\n"
            "SCORE 0: Neither EGFR nor TP53 in top 3."
        ),
    ),
    BenchmarkTask(
        task_id="T24",
        category="Pathway & Systems Biology",
        difficulty="Hard",
        task_text=(
            'Use the QuickGO API (EMBL-EBI) to retrieve all Gene Ontology (GO) terms with '
            'the category "Biological Process" directly annotated to human BRCA1 '
            "(UniProt: P38398). Report the total count and list 5 GO terms with their IDs."
        ),
        expected_output=(
            "Count: > 20 BP terms (accept any count > 10)\n"
            "Examples (any 5):\n"
            "GO:0000724 — double-strand break repair via homologous recombination\n"
            "GO:0006281 — DNA repair\n"
            "GO:0007049 — cell cycle\n"
            "GO:0045786 — negative regulation of cell cycle\n"
            "GO:0097681 — transcription-coupled nucleotide-excision repair"
        ),
        scoring_criteria=(
            "SCORE 2: Count > 10, 5 GO terms with IDs, all biologically plausible for BRCA1, "
            "QuickGO or AmiGO API cited.\n"
            "SCORE 1: 3-4 GO terms correct, or 5 terms but count not reported.\n"
            "SCORE 0: Fewer than 3 biologically plausible GO terms."
        ),
    ),
]

TASK_INDEX: dict[str, BenchmarkTask] = {t.task_id: t for t in TASKS}

CATEGORIES: list[str] = [
    "Sequence & Variant Analysis",
    "Drug-Target Discovery",
    "Literature & Clinical Mining",
    "Transcriptomics & Single-cell",
    "Structural & Protein Biology",
    "Pathway & Systems Biology",
]
