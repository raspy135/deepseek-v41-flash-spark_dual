#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# code_digest.sh -- 16-hex digest of the engine code a serving image runs: the sources under
# engine/, tools/ and server/ (what the Dockerfile's COPY puts in /app and what dual-dev.sh
# bind-mounts instead). dual-build.sh stamps it on the image as the label dsv41.code_sha;
# dual-up.sh compares the label with the checkout so a plain restart cannot silently boot code
# older than the checkout (2026-10-08: an image from before the EXL3 CUDA kernel came up
# "healthy" on the torch reference MoE, minutes per request).
#
# Usage: scripts/code_digest.sh [repo-root]
# ---------------------------------------------------------------------------------------
set -euo pipefail
cd "${1:-$(dirname "$0")/..}"
export LC_ALL=C
find engine tools server -type f \
    \( -name '*.py' -o -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' \
       -o -name '*.html' -o -name '*.js' \) \
    -not -path '*/__pycache__/*' -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16
