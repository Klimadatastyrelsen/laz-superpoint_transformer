#!/usr/bin/env bash
# Local image name used by build, run, and verify scripts.
# Override with SPT_IMAGE if needed.
SPT_IMAGE="${SPT_IMAGE:-kds_spt_laz_pytorch:latest}"
export SPT_IMAGE
