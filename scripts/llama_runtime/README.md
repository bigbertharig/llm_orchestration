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

## Default Image Name

`llama-runtime:sm61-sm86`

Override with `IMAGE_TAG=...` or `--image ...`.
