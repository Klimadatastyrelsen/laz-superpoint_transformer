#!/usr/bin/env bash
# Pull the published image from Docker Hub and tag it locally.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/docker_defaults.sh
source "${HERE}/scripts/docker_defaults.sh"

HUB_IMAGE="${SPT_IMAGE_HUB:-rasmuspjohansson/kds_spt_laz_pytorch:latest}"

docker pull "${HUB_IMAGE}"
docker tag "${HUB_IMAGE}" "${SPT_IMAGE}"
echo "Pulled ${HUB_IMAGE} -> ${SPT_IMAGE}"
