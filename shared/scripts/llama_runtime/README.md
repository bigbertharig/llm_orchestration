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
- `smoke_runtime.py <image>`: load every model the roles and remote profiles use plus six-card Nemotron
  and split Mellum, one at a time; check chat + completions and record Mellum decode; results in `/mnt/shared/logs/runtime_upgrades/`
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
./upgrade_runtime.sh smoke bNNNNN --baseline /mnt/shared/logs/runtime_upgrades/b11669_20261025_192107.json
./upgrade_runtime.sh promote b11669    # tag stable, pin Dockerfile, set registry runtime_image
```

Then commit `model_registry.json` (benchmarking) and `Dockerfile` (llm_orchestration), push
through the Pi, and pull everywhere (desktop, rig clone, drive). A benchmark campaign can
certify a candidate before promotion with `run_campaign.py ... --runtime-image <candidate>`.

`promote` keeps exactly two images: the new one and the previous one as rollback. Rollback: set
`runtime_image` back to the previous image in `model_registry.json`.

Smoke covers the models in use plus promotion-critical six-card Nemotron and split Mellum.
For Mellum, record a baseline with the current stable image and pass it to the candidate
smoke as `--baseline /mnt/shared/logs/runtime_upgrades/<stable-receipt>.json`. The
candidate requires this baseline and fails if Mellum decode is below 70% of it. This guards
against the upstream Mellum CUDA regression reported in
https://github.com/ggml-org/llama.cpp/issues/30704. The b11669 baseline on
2026-10-25 passed both: Nemotron six cards 201.8 s load, Mellum two 1060s
98.6 s load and 49.62 decode tok/s. Receipt:
`/mnt/shared/logs/runtime_upgrades/b11669_20261025_192107.json`. When a new build changes a model's behaviour, fix that model's settings in the registry
(example: b11669 needed `--reasoning off` instead of `--reasoning-budget 0` for Gemma 4).
