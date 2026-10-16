# llama-server Runtime Image

Every model on the rig runs on one llama-server image: orchestration loads, remote mode
(`rig` / gateway), and benchmark campaigns. The image tag lives in exactly one place, the
benchmarking repo's `model_registry.json` (`runtime_image`); `model_library.py runtime-image`
prints it. Profiles may not pin their own image (`model_library.py check` rejects it).

## Files

- `Dockerfile`: native `llama-server` image. Pins the llama.cpp tag and commit, CUDA `12.9.2`,
  and GPU architectures `61;86`. CUDA 12 is intentional: CUDA 13 removed support for the
  Pascal `sm_61` GTX 1060s. Change CUDA or the architectures only when the GPUs change.
- `build_image.sh <tag>`: build `llama-runtime:<tag>-candidate` for a llama.cpp release tag
- `smoke_runtime.py <image>`: load every model the roles and remote profiles use, one at a time
  on its normal lane, and check chat + completions; results in `/mnt/shared/logs/runtime_upgrades/`
- `upgrade_runtime.sh`: the upgrade procedure below
- `run_runtime.sh`: start one runtime container (defaults to the registry image)
- `stop_runtime.sh`, `probe_runtime.sh`, `smoke_test.sh`: single-runtime helpers
- `entrypoint.sh`: stable binary entrypoint

## Upgrading llama.cpp

On the rig, with the rig idle (`rig stop`):

```bash
cd ~/llm_orchestration/shared/scripts/llama_runtime
./upgrade_runtime.sh latest            # newest release tag, e.g. b11669
./upgrade_runtime.sh build b11669      # llama-runtime:b11669-candidate
./upgrade_runtime.sh smoke b11669      # every model in use, on its lane; all must pass
./upgrade_runtime.sh promote b11669    # tag stable, pin Dockerfile, set registry runtime_image
```

Then commit `model_registry.json` (benchmarking) and `Dockerfile` (llm_orchestration), push
through the Pi, and pull everywhere (desktop, rig clone, drive). A benchmark campaign can
certify a candidate before promotion with `run_campaign.py ... --runtime-image <candidate>`.

`promote` keeps exactly two images: the new one and the previous one as rollback. Rollback: set
`runtime_image` back to the previous image in `model_registry.json`.

Smoke covers the models in use (every model a role or remote profile names), not every tested
model. When a new build changes a model's behaviour, fix that model's settings in the registry
(example: b11669 needed `--reasoning off` instead of `--reasoning-budget 0` for Gemma 4).
