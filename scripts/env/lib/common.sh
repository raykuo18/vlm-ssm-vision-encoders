#!/usr/bin/env bash

set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT_DIR="$(cd "${COMMON_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${COMMON_DIR}/../../.." && pwd)"

log() {
    echo "[env] $*"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

ensure_command() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || die "Missing required command: ${cmd}"
}

discover_nvcc() {
    local candidate=""
    local candidates=()

    if candidate="$(command -v nvcc 2>/dev/null)"; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    if [[ -n "${CUDA_HOME:-}" ]]; then
        candidates+=("${CUDA_HOME}/bin/nvcc")
    fi

    candidates+=(
        "/usr/local/cuda/bin/nvcc"
        "/cm/shared/apps/cuda12.8/toolkit/12.8.0/bin/nvcc"
        "/cm/shared/apps/cuda12.8/bin/nvcc"
    )

    local wildcard_candidate=""
    for wildcard_candidate in /usr/local/cuda-*/bin/nvcc; do
        candidates+=("${wildcard_candidate}")
    done

    for candidate in "${candidates[@]}"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

ensure_nvcc_ready() {
    local nvcc_path=""
    if nvcc_path="$(discover_nvcc)"; then
        export CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "${nvcc_path}")/.." && pwd)}"
        export PATH="$(dirname "${nvcc_path}"):${PATH}"
        return 0
    fi
    die "Missing required command: nvcc"
}

ensure_git_ready() {
    ensure_command git
}

ensure_conda_ready() {
    if ! command -v conda >/dev/null 2>&1; then
        if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
            export PATH="$(dirname "${CONDA_EXE}"):${PATH}"
        elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
            export PATH="${HOME}/anaconda3/bin:${PATH}"
        fi
    fi
    ensure_command conda
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
}

run_in_env() {
    local env_name="$1"
    shift
    local cmd="$*"
    local conda_base=""
    conda_base="$(conda info --base)"
    bash -lc 'set -eo pipefail; source "$1/etc/profile.d/conda.sh"; conda activate "$2"; eval "$3"' bash "${conda_base}" "${env_name}" "${cmd}"
}

remove_env_if_requested() {
    local env_name="$1"
    local recreate="$2"
    if [[ "${recreate}" == "true" ]]; then
        if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${env_name}"; then
            log "Removing conda env ${env_name}"
            conda env remove -y -n "${env_name}"
        fi
    fi
}

create_env_from_scratch() {
    local env_name="$1"
    local python_version="$2"
    if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${env_name}"; then
        log "Creating conda env ${env_name} (python=${python_version})"
        conda create -y -n "${env_name}" "python=${python_version}"
    else
        log "Reusing conda env ${env_name}"
    fi
}

ensure_submodule_initialized() {
    local rel_path="$1"
    local abs_path="${REPO_ROOT}/${rel_path}"
    [[ -d "${abs_path}" ]] || die "Missing submodule path: ${rel_path}"
    if ! find "${abs_path}" -mindepth 1 -maxdepth 1 | read -r _; then
        die "Submodule appears empty: ${rel_path}. Run: git submodule update --init --recursive ${rel_path}"
    fi
}

ensure_repo_checkout() {
    local rel_path="$1"
    local repo_url="$2"
    local repo_ref="$3"
    local abs_path="${REPO_ROOT}/${rel_path}"
    local parent_dir=""

    ensure_git_ready
    parent_dir="$(dirname "${abs_path}")"
    mkdir -p "${parent_dir}"

    if [[ -d "${abs_path}/.git" ]]; then
        :
    elif [[ -d "${abs_path}" ]] && find "${abs_path}" -mindepth 1 -maxdepth 1 | read -r _; then
        die "Path ${rel_path} already exists and is not an empty git checkout. Remove it or initialize the repo manually."
    else
        rm -rf "${abs_path}"
        log "Cloning ${repo_url} -> ${rel_path}"
        git clone "${repo_url}" "${abs_path}"
    fi

    if ! git -C "${abs_path}" rev-parse --verify --quiet "${repo_ref}^{commit}" >/dev/null; then
        log "Fetching ${rel_path}"
        git -C "${abs_path}" fetch --tags --force origin
    fi

    if ! git -C "${abs_path}" rev-parse --verify --quiet "${repo_ref}^{commit}" >/dev/null; then
        die "Git ref ${repo_ref} is not available in ${repo_url}. Update the pinned ref in scripts/env/lib/versions.sh."
    fi

    local current_commit=""
    current_commit="$(git -C "${abs_path}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${current_commit}" != "${repo_ref}" ]]; then
        log "Checking out ${rel_path} @ ${repo_ref}"
        git -C "${abs_path}" checkout --detach "${repo_ref}"
    fi
}

install_cuda_runtime_activate_hook() {
    local env_name="$1"
    local env_prefix=""
    local activate_dir=""
    local deactivate_dir=""
    local activate_hook=""
    local deactivate_hook=""

    env_prefix="$(run_in_env "${env_name}" 'printf "%s" "${CONDA_PREFIX}"')"
    [[ -n "${env_prefix}" ]] || die "Unable to resolve CONDA_PREFIX for ${env_name}"

    activate_dir="${env_prefix}/etc/conda/activate.d"
    deactivate_dir="${env_prefix}/etc/conda/deactivate.d"
    activate_hook="${activate_dir}/vlm_backbones_cuda.sh"
    deactivate_hook="${deactivate_dir}/vlm_backbones_cuda.sh"

    mkdir -p "${activate_dir}" "${deactivate_dir}"

    cat > "${activate_hook}" <<'EOF'
#!/usr/bin/env bash
_vlm_backbones_torch_lib="$(python - <<'PY'
import site
print(f"{site.getsitepackages()[0]}/torch/lib")
PY
)"
_vlm_backbones_cuda_target_lib="${CONDA_PREFIX}/targets/x86_64-linux/lib"
_vlm_backbones_cuda_target_stub_lib="${_vlm_backbones_cuda_target_lib}/stubs"
_vlm_backbones_new_ld_library_path="${CONDA_PREFIX}/lib:${_vlm_backbones_torch_lib}"

if [[ -d "${_vlm_backbones_cuda_target_lib}" ]]; then
    _vlm_backbones_new_ld_library_path="${_vlm_backbones_new_ld_library_path}:${_vlm_backbones_cuda_target_lib}"
fi

if [[ -d "${_vlm_backbones_cuda_target_stub_lib}" ]]; then
    _vlm_backbones_new_ld_library_path="${_vlm_backbones_new_ld_library_path}:${_vlm_backbones_cuda_target_stub_lib}"
fi

export _VLM_BACKBONES_PREV_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
if [[ -n "${_VLM_BACKBONES_PREV_LD_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${_vlm_backbones_new_ld_library_path}:${_VLM_BACKBONES_PREV_LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${_vlm_backbones_new_ld_library_path}"
fi

unset _vlm_backbones_cuda_target_lib
unset _vlm_backbones_cuda_target_stub_lib
unset _vlm_backbones_new_ld_library_path
unset _vlm_backbones_torch_lib
EOF

    cat > "${deactivate_hook}" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${_VLM_BACKBONES_PREV_LD_LIBRARY_PATH+x}" ]]; then
    if [[ -n "${_VLM_BACKBONES_PREV_LD_LIBRARY_PATH}" ]]; then
        export LD_LIBRARY_PATH="${_VLM_BACKBONES_PREV_LD_LIBRARY_PATH}"
    else
        unset LD_LIBRARY_PATH
    fi
    unset _VLM_BACKBONES_PREV_LD_LIBRARY_PATH
else
    unset LD_LIBRARY_PATH
fi
EOF

    chmod +x "${activate_hook}" "${deactivate_hook}"
}

require_nvcc_for_build() {
    ensure_nvcc_ready
}

require_nvcc_if_flash_attn() {
    local with_flash_attn="$1"
    if [[ "${with_flash_attn}" == "true" ]]; then
        ensure_nvcc_ready
    fi
}

detect_torch_cuda_arch_list() {
    if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
        printf '%s\n' "${TORCH_CUDA_ARCH_LIST}"
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi

    nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print $0}' \
        | sort -u \
        | paste -sd';' -
}

parse_common_args() {
    ENV_NAME_DEFAULT="$1"
    WITH_FLASH_ATTN_DEFAULT="${2:-false}"
    ENV_NAME="${ENV_NAME_DEFAULT}"
    RECREATE="false"
    WITH_FLASH_ATTN="${WITH_FLASH_ATTN_DEFAULT}"

    shift
    if [[ $# -gt 0 && ( "$1" == "true" || "$1" == "false" ) ]]; then
        shift
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --env-name)
                ENV_NAME="$2"
                shift 2
                ;;
            --recreate)
                RECREATE="true"
                shift
                ;;
            --with-flash-attn)
                WITH_FLASH_ATTN="true"
                shift
                ;;
            --without-flash-attn)
                WITH_FLASH_ATTN="false"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
}
