"""
C13 - T11 GEO to Europe PMC Fallback Demo
=========================================
BioClaw Task T11: Retrieve title, organism, and n_samples for GSE68849.

BioPLEASE F4 wraps LEARN-phase tool calls with retry + fallback logic. This
script shows what happens when NCBI GEO returns HTTP 408 (timeout): the agent
switches to Europe PMC as the registered fallback source.

Run from any directory:
    python supporting_results/C13_pubchem_fallback/geo_pubchem_fallback_demo.py

Output: geo_pubchem_fallback_demo.log  (same directory)

The verified GSE68849 metadata in Section B is from NCBI GEO DataSets /
ESummary for GDS6063.
"""

import json
import os
import time
import urllib.request

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(OUT_DIR, "geo_pubchem_fallback_demo.log")
DATASET = "GSE68849"
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def try_url(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "BioPLEASE/1.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data, round(time.time() - t0, 2)


emit("=" * 68)
emit("C13  T11 GEO to Europe PMC Fallback Demo")
emit("=" * 68)
emit()
emit("Target  : " + DATASET)
emit("Task    : T11 (BioClaw 24-task benchmark)")
emit()
emit("-" * 68)
emit("SECTION A  Live API Calls")
emit()

GEO_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    "?db=gds&term=" + DATASET + "&retmode=json"
)
EPMC_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    "?query=" + DATASET + "&resultType=core&format=json&pageSize=1"
)

geo_ok = False
geo_result = {}

emit("STEP A1  Primary: NCBI GEO ESearch/ESummary")
emit("  URL: " + GEO_URL)
emit()
try:
    data, elapsed = try_url(GEO_URL)
    ids = data.get("esearchresult", {}).get("idlist", [])
    emit("  HTTP 200 OK  (" + str(elapsed) + "s)")
    if not ids:
        raise ValueError("Empty ID list")

    gds_id = "6063" if "6063" in ids else ids[0]
    summ_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        "?db=gds&id=" + gds_id + "&retmode=json"
    )
    summ, elapsed = try_url(summ_url)
    entry = summ.get("result", {}).get(gds_id, {})
    geo_result = {
        "dataset": entry.get("accession", "N/A"),
        "series": "GSE" + str(entry.get("gse", "N/A")),
        "title": entry.get("title", "N/A"),
        "series_title": entry.get("seriestitle", "N/A"),
        "organism": entry.get("taxon", "N/A"),
        "n_samples": entry.get("n_samples", "N/A"),
        "type": entry.get("gdstype", "N/A"),
        "platform": entry.get("platformtitle", "N/A"),
        "pubmed_id": ",".join(entry.get("pubmedids", [])),
    }
    geo_ok = True
    emit("  OK  GEO data:")
    for k, v in geo_result.items():
        emit("     " + k.ljust(14) + ": " + str(v))
except Exception as exc:
    emit("  FAIL  " + str(exc))
    emit("        F4 fallback triggered")

emit()
emit("STEP A2  Fallback: Europe PMC")
emit("  URL: " + EPMC_URL)
emit()

fallback_ok = False
if not geo_ok:
    try:
        data, elapsed = try_url(EPMC_URL)
        hits = data.get("resultList", {}).get("result", [])
        emit("  HTTP 200 OK  (" + str(elapsed) + "s)  " + str(len(hits)) + " hit(s)")
        if hits:
            h = hits[0]
            emit("  pmid: " + str(h.get("pmid", "N/A")))
            emit("  title: " + str(h.get("title", "N/A"))[:80])
            fallback_ok = True
    except Exception as exc:
        emit("  FAIL  " + str(exc))
        emit("        ToolCallError raised; LLM re-prompted")
else:
    emit("  [SKIPPED - primary succeeded]")

emit()
emit("STEP A3  Live outcome:")
if geo_ok:
    emit("  PRIMARY SUCCESS")
elif fallback_ok:
    emit("  FALLBACK SUCCESS")
else:
    emit("  BOTH FAILED  F4 ToolCallError path exercised")

emit()
emit("-" * 68)
emit("SECTION B  Simulated Fallback  (verified GEO data, replayed offline)")
emit()
emit("Scenario: GEO returns HTTP 408 x3. F4 escalates to Europe PMC.")

REAL = {
    "accession": "GSE68849",
    "dataset": "GDS6063",
    "title": "Influenza A effect on plasmacytoid dendritic cells",
    "series_title": "Impact of influenza A on human plasmacytoid dendritic cells (pDC) gene expression",
    "organism": "Homo sapiens",
    "n_samples": "10",
    "type": "Expression profiling by array",
    "platform": "GPL10558 / Illumina HumanHT-12 V4.0 expression beadchip",
    "pubmed_id": "26826244",
    "submitted": "2016-02-01",
    "verified": "NCBI GEO DataSets GDS6063 / ESummary, retrieved 2026-05-13",
}

emit()
emit("[Sim] STEP 1  GEO ESummary:")
emit("  Response: HTTP 408 x3  max_retries exhausted")
emit("  Escalating to fallback_sources[0]='europe_pmc'")
emit()
emit("[Sim] STEP 2  Europe PMC  200 OK  0.34s")
emit("  PMID " + REAL["pubmed_id"] + " linked to GSE68849")
emit("  Agent re-queries GEO via accession in paper abstract")
emit()
emit("[Sim] STEP 3  Direct GEO lookup  SUCCESS:")
for k, v in REAL.items():
    emit("  " + k.ljust(14) + ": " + str(v))
emit()
emit("  SIMULATED FALLBACK SUCCESS  correct answer recovered via F4.")

emit()
emit("-" * 68)
emit("SECTION C  F4 Code Path  (bioplease/agent/pleas.py)")
emit()
emit("  for attempt in range(self.max_retries + 1):")
emit("      try:")
emit("          result = self._call_tool(step)  # primary source")
emit("          break")
emit("      except ToolCallError:")
emit("          if attempt < self.max_retries:")
emit("              time.sleep(2 ** attempt)        # backoff: 1s, 2s, 4s")
emit("          elif self.fallback_sources:")
emit("              result = self._call_fallback(step)  # Europe PMC etc.")
emit("          else:")
emit("              state.errors.append(str(e))")
emit("              state.phase = 'ASSESS'           # re-route for LLM retry")
emit("              return state")
emit("  state.executions.append(ExecutionRecord(...))  # always recorded (F4)")
emit()
emit("=" * 68)
emit("C13 complete.")
emit("=" * 68)

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print()
print("Log saved: " + LOG_PATH)
