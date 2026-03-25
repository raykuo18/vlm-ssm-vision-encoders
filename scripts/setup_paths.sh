#!/usr/bin/env bash

set -euo pipefail

# Detect server type and specific server
if [[ -n "${SLURM_JOB_ID:-}" ]] || command -v squeue >/dev/null 2>&1; then
    SERVER_TYPE="slurm"
    # Check if we're on nvwulf server
    if [[ "$(hostname)" == *"nvwulf"* ]] || [[ "${PWD}" == *"/lustre/nvwulf/"* ]]; then
        SERVER_NAME="nvwulf"
    else
        SERVER_NAME="other-slurm"
    fi
else
    SERVER_TYPE="gpu"
    # Check if we're on nvwulf server
    if [[ "${PWD}" == *"/lustre/nvwulf/"* ]]; then
        SERVER_NAME="nvwulf"
    else
        SERVER_NAME="other-gpu"
    fi
fi

# Set paths based on server
if [[ "${SERVER_NAME}" == "nvwulf" ]]; then
    # NVWULF server paths
    export RUN_ROOT_DIR="/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_runs"
    export DATASET_ROOT="/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_dataset/cobra_dataset"
    export VLM_EVAL_DATA_ROOT="/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data"

    # Dataset paths for NVWULF
    export DATASET_LLAVA_V15_INSTRUCT="${DATASET_ROOT}/download/llava-v1.5-instruct"
    export DATASET_LLAVA_V15_IMAGE_ONLY="${DATASET_ROOT}/download/llava-v1.5-instruct"
    export DATASET_LLAVA_V15_VQA_GQA_AOK="${DATASET_ROOT}/download/llava-v1.5-instruct"

else
    # Other server paths (update these as needed)
    export RUN_ROOT_DIR="./runs"
    export DATASET_ROOT="./datasets"
    export VLM_EVAL_DATA_ROOT="./datasets/vlm-evaluation"

    # Dataset paths (update these paths for your other servers)
    export DATASET_LLAVA_V15_INSTRUCT="${DATASET_ROOT}/llava-v1.5-instruct"
    export DATASET_LLAVA_V15_IMAGE_ONLY="${DATASET_ROOT}/llava-v1.5-instruct"
    export DATASET_LLAVA_V15_VQA_GQA_AOK="${DATASET_ROOT}/llava-v1.5-instruct"
fi

# Export server name for wandb tags
export WANDB_SERVER_TAG="${SERVER_NAME}"

# Print configuration
echo "Server: ${SERVER_NAME} | Run Dir: ${RUN_ROOT_DIR} | Train Data: ${DATASET_ROOT} | Eval Data: ${VLM_EVAL_DATA_ROOT}"
