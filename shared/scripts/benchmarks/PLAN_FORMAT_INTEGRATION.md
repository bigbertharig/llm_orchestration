# Plan Format Integration

`PLAN_FORMAT.md` already supports benchmark-driven model selection via existing llm task fields:

- `llm_model`
- `llm_min_tier`
- `llm_placement`

## Authoring Workflow

1. Classify each llm task by intent (structured extraction, deep reasoning, code generation, code review, general QA).
2. Use `model_task_library.json` recommendations for the matching profile.
3. Set explicit model fields in each plan task.
4. Keep choices measurable by linking back to benchmark test IDs in task notes or plan notes.

## Example

```markdown
### score_candidates
- **executor**: worker
- **task_class**: llm
- **llm_model**: qwen2.5-coder:14b
- **llm_min_tier**: 2
- **llm_placement**: split_gpu
- **command**: `python {PLAN_PATH}/scripts/score.py --batch-id {BATCH_ID} --item {ITEM.id}`
- **depends_on**: extract_profiles
```

## Automatic Recommendation

Use:

```bash
python3 /mnt/shared/scripts/benchmarks/recommend_plan_models.py \
  --plan /mnt/shared/plans/<plan_name>/plan.md
```

Then copy the recommendations into your plan task definitions.
