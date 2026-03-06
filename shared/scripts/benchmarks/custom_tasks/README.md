# Custom Tasks

These are your local acceptance tests for orchestration behavior that public benchmarks do not cover well.

## Recommended Initial Custom Tasks

- `custom_json_schema_strict`
- `custom_tool_plan_sequence`
- `custom_ambiguity_handling`
- `custom_command_safety`
- `custom_orchestration_tradeoff`
- `custom_long_context_extract`

Define prompts, grading rules, and pass thresholds for each in this folder. Keep these stable so model-to-model comparisons are fair.

Current executable source of truth:
- `cases.json`
- `../run_local_custom_task.py`

Example:

```bash
python3 /media/bryan/shared/scripts/benchmarks/run_local_custom_task.py \
  --id custom_command_safety \
  --model qwen2.5:7b \
  --base-url http://localhost:11436
```
