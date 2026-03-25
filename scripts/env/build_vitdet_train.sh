#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/versions.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/build_steps.sh"

parse_common_args "vlm-backbones-train-vitdet" "true" "$@"
ensure_conda_ready
require_nvcc_if_flash_attn "${WITH_FLASH_ATTN}"
remove_env_if_requested "${ENV_NAME}" "${RECREATE}"
create_env_from_scratch "${ENV_NAME}" "${PYTHON_VERSION_DEFAULT}"
install_base_train_stack "${ENV_NAME}" "${WITH_FLASH_ATTN}"
install_vitdet_extras "${ENV_NAME}"
finish_train_env_setup "${ENV_NAME}"
log "Ready: ${ENV_NAME}"
