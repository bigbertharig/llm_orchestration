# Runtime Upgrade & Gemma-4 Extraction Test - May 1, 2026

Historical record. Runtime tags, plan state, and controller behavior below are
observations from May 2026; current operations are documented in
`workspace/quickstart.md` and `workspace/gpu_lane_reservations.md`.

Session context: Upgrading the rig runtime image to support Gemma-4 model family,
then running the research_prospector cloud search plan with gemma-4:e2b-q8 for
candidate extraction (identify_people task).

## Issue 11: Shoulder Scripts Still Reference WORKER_OLLAMA_URL

**Location**: `plans/shoulders/research_assistant/scripts/{identify_people,plan_searches,score_person}.py`

**Symptom**: After the May 1 hardening session removed the `WORKER_OLLAMA_URL`
compatibility shim from `worker.py`, the shoulder scripts would fall back to
`http://localhost:11434` (Ollama legacy port) instead of the worker's llama-server
port. Submissions would pass the submit-time legacy guard (which scans the plan
directory, not the shoulder directory) but tasks would silently connect to the
wrong port at runtime.

**Root cause**: The hardening session removed the env var export from worker.py
but did not update the shoulder scripts that consumed it. Three of four LLM-calling
shoulder scripts still read `WORKER_OLLAMA_URL` with fallback to port 11434:
- `identify_people.py` line 425
- `plan_searches.py` line 119
- `score_person.py` line 300

`transform_output.py` was already updated to use `BRAIN_API_BASE`/`WORKER_API_BASE`.

**Fix applied**: Changed env var references in all three scripts from
`WORKER_OLLAMA_URL` to `WORKER_API_BASE`:
```python
# Before:
default=os.environ.get("WORKER_OLLAMA_URL", "http://localhost:11434")
# After:
default=os.environ.get("WORKER_API_BASE", "http://localhost:11434")
```

For `identify_people.py`, also updated the brain fallback:
```python
# Before:
default=os.environ.get("BRAIN_OLLAMA_URL") or os.environ.get("WORKER_OLLAMA_URL", ...)
# After:
default=os.environ.get("BRAIN_API_BASE") or os.environ.get("WORKER_API_BASE", ...)
```

**Note**: The `--ollama-url` argument name and `call_ollama` function names were
left as-is for now (cosmetic, no runtime impact). These should be renamed in a
future cleanup pass.

---

## Issue 12: Brain Batch ID Mismatch with Pre-Created Search Results

**Location**: `plans/arms/research_prospector/history/`

**Symptom**: First submission attempt (`20260501_184031`) failed with all
identify_people tasks reporting:
```
Error: missing raw results file: .../history/20260501_184031/results/raw_search_results.json
```

**Root cause**: The cloud_search_plan expects `raw_search_results.json` to exist
at `{BATCH_PATH}/results/` before identify_people runs. But the brain auto-generates
`{BATCH_ID}` from the submission timestamp (e.g., `20260501_184031`), not from
a user-provided batch name. Pre-creating a batch directory with a custom name
(`gemma4_20260501_183813`) doesn't help because `{BATCH_PATH}` resolves to the
brain's auto-generated path.

**Fix applied**: After submitting, immediately watch for the brain's new batch
directory and copy `raw_search_results.json` into it:
```bash
# Submit, then:
BATCH_DIR=$(ls -dt /mnt/shared/.../history/20260501_* | head -1)
cp .../raw_search_results.json ${BATCH_DIR}/results/
```

**Better long-term fix**: Either:
1. Add a `--batch-id` override to submit.py so the brain uses the pre-created path
2. Add a `pre_run` hook in the plan format that copies prerequisite files
3. Have the brain check for a `results/raw_search_results.json` in the plan's
   `input/` directory and symlink it into the batch path during plan setup

---

## Configuration Changes Applied

### 1. Runtime Image Upgrade

**File**: `agents/config.json` line 11

```
Before: "llama_runtime_image": "llama-runtime:sm61-sm86"
After:  "llama_runtime_image": "llama-runtime:b8884-candidate"
```

This was the prerequisite for loading Gemma-4 GGUF files. The sm61-sm86 image
lacked the architecture support needed for Gemma-4 models.

### 2. Plan Model Revert

**File**: `plans/arms/research_prospector/cloud_search_plan.md`

Changed identify_people task from smoke test model back to target model:
```
Before: llm_model: qwen2.5-coder:7b
After:  llm_model: gemma-4:e2b-q8
```

Updated all plan references (header, workflow diagram, model rationale notes).

---

## Test Results

### Batch: `20260501_184528`

**Plan**: `cloud_search_plan.md` with `gemma-4:e2b-q8` for extraction,
`qwen2.5-coder:7b` for plan_searches and score_person.

**Runtime image**: `llama-runtime:b8884-candidate`

**Extraction quality**: Confirmed working. First 5 candidates:
- Sung-Gheel Jang - Director of Geospatial Center @ Stony Brook University
- Michael Minn - Teaching Assistant Professor @ UIUC
- Jeffrey L. Brewer - Program Director @ University of Cincinnati Online
- Michael Camponovo - Director of GIST Program @ University of Tennessee
- Jonathan Nelson - Professor and Director of GIS Programs @ UW-Madison

All are real GIS faculty/directors (vs discipline names with qwen2.5-coder:7b).

**Pipeline status**: 14 candidates extracted across discovery rounds. 5 initial
pipelines spawned. plan_searches, scrape_person flowing. score_person pending.
All 5 worker GPUs hot with gemma-4:e2b-q8 loaded (2.6GB VRAM each).

---

## Files Modified This Session

| File | Change |
|------|--------|
| `agents/config.json` | Runtime image: sm61-sm86 -> b8884-candidate |
| `plans/arms/research_prospector/cloud_search_plan.md` | identify_people model: qwen2.5-coder:7b -> gemma-4:e2b-q8 |
| `shoulders/research_assistant/scripts/identify_people.py` | Env var: WORKER_OLLAMA_URL -> WORKER_API_BASE |
| `shoulders/research_assistant/scripts/plan_searches.py` | Env var: WORKER_OLLAMA_URL -> WORKER_API_BASE |
| `shoulders/research_assistant/scripts/score_person.py` | Env var: WORKER_OLLAMA_URL -> WORKER_API_BASE |
| `shoulders/research_assistant/scripts/profile_scraper.py` | Fuzzy company matching in `score_identity` — token-based partial credit |

---

## Issue 13: Scraper Drops All Sources Due to Exact Company Name Matching

**Location**: `plans/shoulders/research_assistant/scripts/profile_scraper.py`, `score_identity()` (line 324)

**Symptom**: Every candidate in batch `20260501_184528` had `sources included: 0`,
`sources dropped: 10`. The scorer received empty profiles and rejected everyone.
Pipeline completed with 0 accepted, 21 rejected.

**Root cause**: `score_identity()` gives +30 for company match, but requires the
**exact full string** (e.g., "University of Illinois Urbana-Champaign"). Search
results typically use abbreviations ("UIUC", "Illinois", "UW-Madison", "Temple").
Without the company match, results score only 40% (name alone) and the 50%
inclusion threshold drops them.

Before fix — realistic snippets scored:
```
40% DROP  "Michael Minn - Teaching Assistant Professor - UIUC"
40% DROP  "Jonathan Nelson GIS Programs UW-Madison"
40% DROP  "Kyle Redican Spatial Analysis Lab Richmond"
40% DROP  "Kevin Henry - Temple"
```

**Fix applied**: Added fuzzy token matching to the company signal. Extracts
significant tokens (>=4 chars, excluding stopwords like "university", "college",
"state", "institute"), then scores based on hit ratio:
- 50%+ tokens match: +20 (strong partial match)
- 1+ tokens match: +10 (weak partial match)
- Exact full string: +30 (unchanged)

After fix:
```
50% PASS  "Michael Minn | Department of Geography | Illinois"
60% PASS  "Jonathan Nelson GIS Programs UW-Madison"
60% PASS  "Kyle Redican Spatial Analysis Lab Richmond"
60% PASS  "Kevin Henry - Temple"
 0% DROP  "Random person at some other place"  (correctly stays dropped)
```

**Note**: Pure acronyms like "UIUC" (4 chars, not in tokenized name) still don't
match. This is acceptable — enough non-acronym results should now pass to give
the scorer meaningful content.
