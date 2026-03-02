# Future: Task Decomposition and Model Placement Playbook

Purpose: capture reusable lessons about how to split work across CPU/script/7B/14B workers so future plans start from proven task shapes instead of trial-and-error.

## Why This Exists

We are learning two things at once:
- how to make workers/runtime state reliable
- how to decompose tasks so they parallelize well and fit the right model tier

The second part is reusable across plans and should become a standard design playbook.

## Core Principle

Do not ask a stronger model to compensate for bad task decomposition.

Prefer:
1. CPU/script preprocessing to reduce noise
2. complexity-balanced slicing
3. 7B extraction for broad coverage
4. 14B verify/adjudication only where needed

## Task Shape Matrix (Seed Version)

### 1) Manifest / Repo Shape / Index Building
- Best executor: `script` or `cpu`
- Model tier: none
- Split required: no
- Examples:
  - repo structure extraction
  - JSON/data summarization
  - slice complexity scoring
  - claim/evidence linking
- Notes:
  - deterministic work, cheap to parallelize
  - ideal for underutilized CPU workers

### 2) Broad Extraction / First-Pass Evidence Gathering
- Best executor: `llm`
- Model tier: `7B` (default)
- Split required: no
- Preprocess recommended:
  - repo structure summary
  - claim/evidence links
  - slice complexity ordering
  - JSON summaries
- Notes:
  - high parallelism phase
  - optimize for coverage and structured output reliability
  - JSON drift is common on 7B; preserve raw output artifacts for repair

### 3) Verification / Adjudication (Quality Gate Work)
- Best executor: `llm`
- Model tier: `14B` (often split)
- Split required: often yes on this rig
- Preprocess recommended:
  - precomputed extracts
  - contradiction candidates
  - normalized prior findings
- Notes:
  - usually the dominant wall-time phase
  - straggler slices drive batch makespan
  - requires careful complexity-based slicing, not just file-count shards

### 4) Contradiction / Gap-Focused Verification
- Best executor: `llm`
- Model tier: `14B` preferred for final verify
- Split required: often yes
- Preprocess recommended:
  - contradiction prefilter
  - claim/evidence linking
  - normalized review outputs
- Notes:
  - docs/reference-heavy slices can timeout even with modest file count
  - use stricter complexity caps than general review slices
  - avoid one giant contradictions slice

### 5) Runtime Validation / Probe Classification
- Best executor: `script` + `cpu`
- Model tier: none (or optional brain summarization only)
- Split required: no
- Notes:
  - keep deterministic
  - classify repo tests vs third-party noise (`.venv`, site-packages, etc.)

### 6) Final Synthesis / Reporting
- Best executor: `brain` LLM + `script`
- Model tier: highest available (brain)
- Split required: depends on brain placement
- Notes:
  - quality depends heavily on upstream slice quality and normalization
  - should not be carrying basic extraction burden

## Placement Strategy Guidelines (Current)

### CPU / Script First
- Push all deterministic preprocessing to CPU/script tasks
- Use LLMs for reasoning, not indexing/parsing that code can do cheaply

### 7B for Breadth
- Use `7B` for extraction and broad first-pass classification
- Keep prompts tight and schema simple
- Expect occasional JSON formatting failures; preserve artifacts and repair, don’t heuristic-fill

### 14B for Depth
- Use `14B` verify/adjudication for:
  - contradiction validation
  - high-confidence evidence checks
  - claim support adjudication
- Do not waste 14B time on low-signal raw file scanning

### Split Runtime Policy
- Splitness is runtime implementation detail for normal LLM work (capability-based routing)
- Split-specific behavior belongs to:
  - telemetry
  - load/unload meta-tasks
  - resource sequencing

## Slicing Methodology (Reusable)

## 1) Shard count is not enough
- `VERIFY_SHARDS=N` helps throughput, but does not prevent bad slice packing
- One oversized slice can dominate runtime even when total shards look reasonable

## 2) Use complexity budgets per slice
- Compute per-file complexity score (weighted by size/type/path)
- Pack slices to target complexity budget instead of only file count
- Keep a file-count cap as a guardrail

## 3) Phase-specific slice policies
- Different phases need different caps:
  - docs / contradictions: tighter caps
  - architecture / code-heavy: can be larger

## 4) Benchmark by complexity, not just file count
- Selection for benchmark manifests should prefer `complexity_hint` when available
- File count alone misses heavy docs/reference slices

## Initial Complexity Heuristic (Portable Starting Point)

Per-file score (example):
- `score = weight(path, type) * (sqrt(size_bytes) ** 1.2)`

Suggested weights (starting point):
- code files (`.py`, `.ts`, `.js`): `1.0x`
- docs (`.md`, `.txt`): `1.2x`
- JSON: `1.5x`
- `reference/`: `1.8x`
- `input/modes/`, `input/runs/`: `1.6x`
- `README*`: `1.4x`

Tune by measured runtime, not intuition.

## What to Measure (Every Benchmark)

For each slice:
- phase
- file count
- complexity hint
- runtime duration
- timeout/failure result
- repair usage (if any)

For each phase:
- wall time
- p50/p90/p95/max duration
- estimated concurrency
- timeout count
- blocked_cloud count

## `github_analyzer` Lessons (Seeded)

### What worked well
- CPU preprocessing artifacts (structure, JSON summaries, claim-evidence linking)
- 7B extract + 14B verify split pipeline (conceptually)
- worker-side task-context metadata (`WORKER_TASK_JSON_PATH`) for dynamic retry/repair hints
- complexity-balanced contradictions slicing (first pass) is the right direction

### What failed / degraded
- single giant contradictions slices (`14B` verifier/adjudicator timeouts)
- file-count-only benchmark selection
- relying on plan templates for dynamic retry metadata

### Repo-size tuning notes (current)
- `research_prospector`-sized repos: `VERIFY_SHARDS=4` often better than `8`
- large repos need:
  - reliable second split pair promotion
  - verify straggler control
  - contradiction-specific slicing caps

## Design Checklist for New Plans

Before writing LLM tasks:
1. What can be extracted deterministically on CPU/script?
2. What is the first-pass 7B task vs 14B verify task?
3. What is the complexity metric for slice packing?
4. Which phases need stricter caps (docs/contradictions)?
5. What artifacts should be preserved for repair/retry?
6. How will benchmark selection reflect actual complexity?

## Promotion Path (Later)

If this proves stable across multiple plans:
- promote to a shared workspace design doc / template
- add a canonical task-shape matrix to `workspace/PLAN_FORMAT.md` (guidance section)
- standardize metrics fields for complexity/runtime correlation

