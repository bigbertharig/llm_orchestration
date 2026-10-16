#!/bin/bash
# Upgrade the one llama-server image every model runs on.
#
#   upgrade_runtime.sh latest           newest llama.cpp release tag
#   upgrade_runtime.sh build   bNNNNN   build llama-runtime:bNNNNN-candidate
#   upgrade_runtime.sh smoke   bNNNNN   load every model the roles/remote profiles use on the candidate
#   upgrade_runtime.sh promote bNNNNN   tag it llama-runtime:bNNNNN, pin the Dockerfile, set the
#                                       registry runtime_image, keep only the new image and the
#                                       previous one (rollback); then commit both repos
#
# Run on the rig while it is idle (`rig stop`). Smoke covers the models in use (roles and remote
# profiles), not every tested model; fix settings a new build changes in model_registry.json.
# Roll back by setting runtime_image to the previous image in model_registry.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "${HERE}/../.." && pwd)"
BENCH="${TREE}/plans/shoulders/benchmarking"
PY="${PYTHON:-/home/bryan/llm-orchestration-venv/bin/python}"
REPO="https://github.com/ggml-org/llama.cpp.git"

cmd="${1:-}"
version="${2:-}"
need_version() {
  [[ "${version}" =~ ^b[0-9]+$ ]] || { echo "need a llama.cpp tag like b11669" >&2; exit 64; }
}

case "${cmd}" in
  latest)
    curl -fsS https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
      | "${PY}" -c 'import json,sys; r=json.load(sys.stdin); print(r["target_commitish"], r["tag_name"], r["published_at"])' \
      | while read -r commit name published; do
          tag="$(git ls-remote --tags "${REPO}" | awk -v c="${commit}" '$1==c && $2 ~ /refs\/tags\/b[0-9]+$/ {sub("refs/tags/","",$2); print $2}')"
          echo "${tag:-?} (release ${name}, ${published}, ${commit})"
        done
    ;;
  build)
    need_version
    "${HERE}/build_image.sh" "${version}"
    ;;
  smoke)
    need_version
    "${PY}" "${HERE}/smoke_runtime.py" "llama-runtime:${version}-candidate"
    ;;
  promote)
    need_version
    docker image inspect "llama-runtime:${version}-candidate" >/dev/null
    ref="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "llama-runtime:${version}-candidate")"
    previous="$("${PY}" "${BENCH}/scripts/active/model_library.py" runtime-image)"
    docker tag "llama-runtime:${version}-candidate" "llama-runtime:${version}"
    sed -i -E "s|^ARG LLAMA_CPP_REF=.*|ARG LLAMA_CPP_REF=${ref}|; s|^ARG LLAMA_CPP_VERSION=.*|ARG LLAMA_CPP_VERSION=${version}|" "${HERE}/Dockerfile"
    "${PY}" - "${BENCH}/model_registry.json" "llama-runtime:${version}" <<'EOF'
import json, sys
path, image = sys.argv[1], sys.argv[2]
raw = open(path, encoding="utf-8").read()
reg = json.loads(raw)
print(f"runtime_image: {reg['runtime_image']} -> {image}")
reg["runtime_image"] = image
open(path, "w", encoding="utf-8", newline="\n").write(json.dumps(reg, indent=2, ensure_ascii=True) + "\n")
EOF
    "${PY}" "${BENCH}/scripts/active/model_library.py" check
    # Keep exactly two images: the new one and the previous one as rollback.
    for tag in $(docker images --format '{{.Repository}}:{{.Tag}}' llama-runtime); do
      if [ "${tag}" != "llama-runtime:${version}" ] && [ "${tag}" != "${previous}" ]; then
        docker rmi "${tag}" >/dev/null && echo "removed ${tag}"
      fi
    done
    echo "promoted llama-runtime:${version} (rollback: ${previous})."
    echo "Commit: benchmarking model_registry.json, llm_orchestration Dockerfile"
    ;;
  *)
    sed -n '2,12p' "$0"
    exit 64
    ;;
esac
