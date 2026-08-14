# llama-server Runtime Helpers

This directory holds the concrete runtime artifacts for the Ollama to
`llama-server` migration:

- a dedicated CUDA image build
- a deterministic container entrypoint
- host-side wrappers for build, run, stop, and readiness probe

Defaults are explicit and can be overridden per invocation. The intent is that
agent-side plumbing should call the same commands instead of embedding ad hoc
`docker run` strings.

## Files

- `Dockerfile`: native `llama-server` image for `sm_61` and `sm_86`
- `entrypoint.sh`: stable binary entrypoint
- `build_image.sh`: build the runtime image
- `run_runtime.sh`: start one runtime container
- `stop_runtime.sh`: stop one runtime container
- `probe_runtime.sh`: readiness probe via `/v1/models`
- `smoke_test.sh`: readiness + chat/completions smoke test
- `build_and_smoke_test.sh`: build image, run one runtime, smoke test, then stop

## Default Image Name

`llama-runtime:b10333`

The Dockerfile currently pins llama.cpp `b10333` at commit
`08659901c43b51de735740f1cf61bb82fbe0c4e4` and CUDA `12.9.2`. CUDA 12 is
intentional: CUDA 13 removed offline compilation and library support for the
Pascal `sm_61` worker GPUs.

Build upgrades as a candidate before moving the stable tag:

```bash
IMAGE_TAG=llama-runtime:b10333-candidate ./build_image.sh
```

Promote the candidate only after single-GPU smoke tests pass on both `sm_61`
and `sm_86`, followed by the brain-model runtime smoke test.

Override paths:
- build time: `IMAGE_TAG=...`
- direct runtime launch: `--image ...`
- orchestrator-managed runtime launch: set `llama_runtime_image` in the active agent config

Normal operation should prefer the config-driven path so single-worker and split-worker
loads use the same promoted runtime image without code edits.
