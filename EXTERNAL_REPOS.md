# External repositories

The [four-repo map](README.md#the-four-repos) covers our own projects. This
page tracks outside runtime code used on the rig and research sources that
inform placement experiments. Last checked: 2026-11-15.

| Upstream | License of upstream code | Role here | Current integration / source location |
|---|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) | `llama-server` runtime used by rig loads and workers | Build and image procedure: [llama runtime README](shared/scripts/llama_runtime/README.md). Check the image digest and source commit recorded by a campaign before replacing a runtime. |
| [Niko1221/Strata](https://github.com/Niko1221/Strata) | [MIT](https://github.com/Niko1221/Strata/blob/main/LICENSE) | Optional `backend: strata` in the rig's managed remote load path | Rig release bundle `/data/bench-staging/strata-v0.2.10` (Control/engine 0.1.56); Pi source checkout `~/Strata` at `baf16363` on 2026-11-15. The new source checkout does not alter rig loads. See [managed runtime](README.md#managed-strata-runtime). |
| [Helldez/BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge) | [Apache-2.0](https://github.com/Helldez/BigMoeOnEdge/blob/main/LICENSE) | Research source for expert placement; not the production runtime | Local research checkout at `930b2c9` on 2026-11-15. Experimental ownership and evidence are in [benchmarking](https://github.com/bigbertharig/benchmarking/blob/master/EXPERT_TENSOR_SHARDING_RESEARCH.md). |

These are code licenses for the named upstream repositories. Model weights, quantized packs, and any separately downloaded components carry their own terms; record those terms with the exact model artifact before redistributing or publishing a recipe.

## Checking for updates

The source links above are the upstream projects. A local checkout or release
pin is a snapshot, not an automatic update. For a checkout, run
`git -C PATH status --short --branch`, then `git -C PATH fetch origin` and
`git -C PATH log --oneline HEAD..origin/BRANCH` (replace PATH and BRANCH from
the table or the checkout's `git branch -r`). Review release notes and changes
before changing a deployed binary, model recipe, or benchmark baseline. Record
the new source revision and the validation receipt here when adopting it.
