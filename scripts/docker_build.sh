#!/usr/bin/env bash
# Build the canonical SPT LAZ Docker image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/docker_defaults.sh
source "${HERE}/scripts/docker_defaults.sh"

docker build -t "${SPT_IMAGE}" "${HERE}"
echo "Built ${SPT_IMAGE}"
