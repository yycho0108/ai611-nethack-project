#!/usr/bin/env bash

set -ex

# NOTE(ycho): see github.com/facebookresearch/nle
IMAGE_TAG='fairnle/challenge:dev'

# Figure out repository root.
SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"


# Pull docker image from hub
docker pull ${IMAGE_TAG}

# Launch simple docker container with
# * Network enabled (passthrough to host)
# * Privileged permissions
# * All GPU devices visible
# * Current working git repository mounted at /root

docker run -it --rm \
    --mount type=bind,source=${REPO_ROOT},target="/home/aicrowd/$(basename ${REPO_ROOT})" \
    --network host \
    --privileged \
    --gpus all \
    "$@" \
    "${IMAGE_TAG}" \
    bash
