#!/usr/bin/env bash

set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/versions.sh"

DETECTED_BUILD_JOBS="$(nproc 2>/dev/null || echo 8)"
if (( DETECTED_BUILD_JOBS > 8 )); then
    DEFAULT_BUILD_JOBS=8
else
    DEFAULT_BUILD_JOBS="${DETECTED_BUILD_JOBS}"
fi
ENV_BUILD_JOBS="${ENV_BUILD_JOBS:-${DEFAULT_BUILD_JOBS}}"

cuda_build_exports() {
    local build_jobs="${1:-${ENV_BUILD_JOBS}}"
    local arch_list=""
    arch_list="$(detect_torch_cuda_arch_list)"
    if [[ -n "${arch_list}" ]]; then
        printf "export TORCH_CUDA_ARCH_LIST='%s'; " "${arch_list}"
    fi
    printf 'export CUDA_HOME="${CONDA_PREFIX}"; export PATH="${CUDA_HOME}/bin:${PATH}"; export MAX_JOBS=%s; export CMAKE_BUILD_PARALLEL_LEVEL=%s;' "${build_jobs}" "${build_jobs}"
}

flash_attn_build_jobs() {
    if [[ -n "${FLASH_ATTN_BUILD_JOBS:-}" ]]; then
        printf '%s\n' "${FLASH_ATTN_BUILD_JOBS}"
        return 0
    fi

    if (( ENV_BUILD_JOBS > 4 )); then
        printf '4\n'
    else
        printf '%s\n' "${ENV_BUILD_JOBS}"
    fi
}

flash_attn_cuda_archs() {
    local torch_archs=""
    local flash_archs=()
    local item=""
    local torch_arch_items=()

    torch_archs="$(detect_torch_cuda_arch_list)"
    if [[ -z "${torch_archs}" ]]; then
        return 0
    fi

    IFS=';' read -r -a torch_arch_items <<< "${torch_archs}"
    for item in "${torch_arch_items[@]}"; do
        item="${item//[[:space:]]/}"
        item="${item/./}"
        if [[ -n "${item}" ]]; then
            flash_archs+=("${item}")
        fi
    done

    local joined=""
    joined="$(IFS=';'; printf '%s' "${flash_archs[*]}")"
    printf '%s\n' "${joined}"
}

conda_install_nvidia_pkg() {
    local env_name="$1"
    local pkg_spec="$2"
    bash -lc "set -eo pipefail; source \"$(conda info --base)/etc/profile.d/conda.sh\"; conda install -y -n \"${env_name}\" -c nvidia \"${pkg_spec}\""
}

ensure_cuda_nvcc_toolchain() {
    local env_name="$1"
    local current_release=""
    current_release="$(run_in_env "${env_name}" 'if [[ -x "${CONDA_PREFIX}/bin/nvcc" ]]; then "${CONDA_PREFIX}/bin/nvcc" --version 2>/dev/null | sed -n '"'"'s/.*release \([0-9]\+\.[0-9]\+\).*/\1/p'"'"' | head -n1; fi' || true)"
    if [[ "${current_release}" != "${CUDA_NVCC_VERSION}" ]]; then
        log "Installing cuda-nvcc=${CUDA_NVCC_VERSION} in ${env_name}"
        conda_install_nvidia_pkg "${env_name}" "cuda-nvcc=${CUDA_NVCC_VERSION}"
    fi
}

ensure_cuda_dev_libraries() {
    local env_name="$1"
    local cublas_header_path=""
    cublas_header_path="$(run_in_env "${env_name}" 'ls "${CONDA_PREFIX}/targets/x86_64-linux/include/cublas_v2.h" 2>/dev/null || true')"
    if [[ -z "${cublas_header_path}" ]]; then
        log "Installing cuda-libraries-dev=${CUDA_NVCC_VERSION} in ${env_name}"
        conda_install_nvidia_pkg "${env_name}" "cuda-libraries-dev=${CUDA_NVCC_VERSION}"
    fi
}

install_common_python_stack() {
    local env_name="$1"
    run_in_env "${env_name}" "python -m pip install -U pip setuptools wheel"
    run_in_env "${env_name}" "python -m pip install --upgrade --index-url '${TORCH_INDEX_URL}' 'torch==${TORCH_VERSION}' 'torchvision==${TORCHVISION_VERSION}' 'torchaudio==${TORCHAUDIO_VERSION}'"
    run_in_env "${env_name}" "python -m pip uninstall -y chardet || true"
    run_in_env "${env_name}" "python -m pip install packaging ninja"
    install_cuda_runtime_activate_hook "${env_name}"
}

install_train_python_stack() {
    local env_name="$1"
    install_common_python_stack "${env_name}"
    run_in_env "${env_name}" "python -m pip install 'accelerate==${ACCELERATE_VERSION}' 'certifi==${CERTIFI_VERSION}' 'einops==${EINOPS_VERSION}' 'huggingface_hub==${HUGGINGFACE_HUB_VERSION}' 'jsonlines==${JSONLINES_VERSION}' 'Pillow==${PILLOW_VERSION}' 'protobuf==${PROTOBUF_VERSION}' 'PyYAML==${PYYAML_VERSION}' 'requests==${REQUESTS_VERSION}' 'rich==${RICH_VERSION}' 'safetensors==${SAFETENSORS_VERSION}' 'sentencepiece==${SENTENCEPIECE_VERSION}' 'timm==${TIMM_VERSION}' 'tokenizers==${TOKENIZERS_VERSION}' 'transformers==${TRANSFORMERS_VERSION}'"
}

install_eval_python_stack() {
    local env_name="$1"
    install_common_python_stack "${env_name}"
    run_in_env "${env_name}" "python -m pip install 'accelerate==${EVAL_ACCELERATE_VERSION}' 'certifi==${EVAL_CERTIFI_VERSION}' 'einops==${EINOPS_VERSION}' 'huggingface_hub==${HUGGINGFACE_HUB_VERSION}' 'jsonlines==${JSONLINES_VERSION}' 'Pillow==${PILLOW_VERSION}' 'protobuf==${PROTOBUF_VERSION}' 'PyYAML==${PYYAML_VERSION}' 'requests==${EVAL_REQUESTS_VERSION}' 'rich==${RICH_VERSION}' 'safetensors==${SAFETENSORS_VERSION}' 'sentencepiece==${SENTENCEPIECE_VERSION}' 'timm==${EVAL_TIMM_VERSION}' 'tokenizers==${TOKENIZERS_VERSION}' 'transformers==${TRANSFORMERS_VERSION}' 'wandb==${EVAL_WANDB_VERSION}'"
}

install_flash_attn_if_requested() {
    local env_name="$1"
    local with_flash_attn="$2"
    local arch_list=""
    local build_exports=""
    local flash_jobs=""
    local flash_archs=""
    if [[ "${with_flash_attn}" != "true" ]]; then
        return 0
    fi
    ensure_command nvcc
    ensure_cuda_nvcc_toolchain "${env_name}"
    ensure_cuda_dev_libraries "${env_name}"
    arch_list="$(detect_torch_cuda_arch_list)"
    if [[ -n "${arch_list}" ]]; then
        log "Using TORCH_CUDA_ARCH_LIST=${arch_list}"
    fi
    flash_archs="$(flash_attn_cuda_archs)"
    if [[ -n "${flash_archs}" ]]; then
        log "Using FLASH_ATTN_CUDA_ARCHS=${flash_archs}"
    fi
    flash_jobs="$(flash_attn_build_jobs)"
    if [[ "${flash_jobs}" != "${ENV_BUILD_JOBS}" ]]; then
        log "Using FLASH_ATTN_BUILD_JOBS=${flash_jobs}"
    fi
    build_exports="$(cuda_build_exports "${flash_jobs}")"
    if [[ -n "${flash_archs}" ]]; then
        build_exports="${build_exports} export FLASH_ATTN_CUDA_ARCHS='${flash_archs}';"
    fi
    run_in_env "${env_name}" "python -m pip uninstall -y flash-attn || true"
    run_in_env "${env_name}" "${build_exports} python -m pip install --prefer-binary --no-deps --no-build-isolation 'flash-attn==${FLASH_ATTN_VERSION}'"
}

install_base_inference_stack() {
    local env_name="$1"
    local with_flash_attn="$2"
    install_train_python_stack "${env_name}"
    if [[ "${with_flash_attn}" == "true" ]]; then
        install_flash_attn_if_requested "${env_name}" "${with_flash_attn}"
    else
        log "Skipping flash-attn (not required for backbone inference). Pass --with-flash-attn to install it."
    fi
}

install_train_python_extras() {
    local env_name="$1"
    run_in_env "${env_name}" "python -m pip uninstall -y draccus || true"
    run_in_env "${env_name}" "python -m pip install 'draccus @ ${DRACCUS_GIT_REF}' 'wandb==${WANDB_VERSION}'"
}

install_eval_python_extras() {
    local env_name="$1"
    run_in_env "${env_name}" "python -m pip uninstall -y draccus || true"
    run_in_env "${env_name}" "python -m pip install 'accelerate==${EVAL_ACCELERATE_VERSION}' 'draccus @ ${DRACCUS_GIT_REF}' 'ascii_magic==${ASCII_MAGIC_VERSION}' 'gradio==${GRADIO_VERSION}' 'gradio_client==${GRADIO_CLIENT_VERSION}' 'pydantic==${PYDANTIC_VERSION}' 'jinja2==${JINJA2_VERSION}' 'mosaicml-streaming==${MOSAICML_STREAMING_VERSION}' 'openai==${OPENAI_VERSION}' 'pycocotools==${PYCOCOTOOLS_VERSION}' 'scikit-image==${SCIKIT_IMAGE_VERSION}' 'scikit-learn==${SCIKIT_LEARN_VERSION}' 'webdataset==${WEBDATASET_VERSION}' 'pymongo==${PYMONGO_VERSION}' 'spacy==${SPACY_VERSION}'"
}

install_base_train_stack() {
    local env_name="$1"
    local with_flash_attn="$2"
    install_train_python_stack "${env_name}"
    install_train_python_extras "${env_name}"
    if [[ "${with_flash_attn}" == "true" ]]; then
        install_flash_attn_if_requested "${env_name}" "${with_flash_attn}"
    else
        log "Skipping flash-attn. Pass --with-flash-attn to install it for train/eval parity."
    fi
}

install_base_eval_stack() {
    local env_name="$1"
    local with_flash_attn="$2"
    install_eval_python_stack "${env_name}"
    install_eval_python_extras "${env_name}"
    if [[ "${with_flash_attn}" == "true" ]]; then
        install_flash_attn_if_requested "${env_name}" "${with_flash_attn}"
    else
        log "Skipping flash-attn for evaluation. Pass --with-flash-attn to enable it."
    fi
}

install_vmamba_extras() {
    local env_name="$1"
    local arch_list=""
    local build_exports=""
    ensure_repo_checkout "third_party/VMamba" "${VMAMBA_REPO_URL}" "${VMAMBA_REPO_REF}"
    ensure_command nvcc
    ensure_cuda_nvcc_toolchain "${env_name}"
    ensure_cuda_dev_libraries "${env_name}"
    arch_list="$(detect_torch_cuda_arch_list)"
    if [[ -n "${arch_list}" ]]; then
        log "Using TORCH_CUDA_ARCH_LIST=${arch_list}"
    fi
    build_exports="$(cuda_build_exports)"
    run_in_env "${env_name}" "python -m pip install --no-deps -r '${REPO_ROOT}/third_party/VMamba/runtime_requirements.txt'"
    run_in_env "${env_name}" "python -m pip install iopath"
    run_in_env "${env_name}" "python -m pip uninstall -y mamba-ssm mamba_ssm || true"
    run_in_env "${env_name}" "${build_exports} python -m pip install --no-build-isolation 'mamba-ssm @ ${MAMBA_SSM_GIT_REF}'"
    run_in_env "${env_name}" "${build_exports} if ! python -m pip install --no-build-isolation -e '${REPO_ROOT}/third_party/VMamba/kernels/selective_scan'; then cd '${REPO_ROOT}/third_party/VMamba/kernels/selective_scan' && python setup.py build_ext --inplace && python setup.py install --no-deps; fi"
}

install_mambavision_extras() {
    local env_name="$1"
    local arch_list=""
    local build_exports=""
    ensure_repo_checkout "third_party/MambaVision" "${MAMBAVISION_REPO_URL}" "${MAMBAVISION_REPO_REF}"
    ensure_command nvcc
    ensure_cuda_nvcc_toolchain "${env_name}"
    arch_list="$(detect_torch_cuda_arch_list)"
    if [[ -n "${arch_list}" ]]; then
        log "Using TORCH_CUDA_ARCH_LIST=${arch_list}"
    fi
    build_exports="$(cuda_build_exports)"
    run_in_env "${env_name}" "python -m pip install 'einops==0.8.1' 'requests==2.32.3' 'Pillow==11.1.0'"
    run_in_env "${env_name}" "python -m pip uninstall -y mamba-ssm mamba_ssm || true"
    run_in_env "${env_name}" "${build_exports} python -m pip install --no-build-isolation 'mamba-ssm @ ${MAMBA_SSM_GIT_REF}'"
    run_in_env "${env_name}" "python -m pip install -e '${REPO_ROOT}/third_party/MambaVision' --no-deps"
}

install_vitdet_extras() {
    local env_name="$1"
    ensure_repo_checkout "third_party/detectron2" "${DETECTRON2_REPO_URL}" "${DETECTRON2_REPO_REF}"
    run_in_env "${env_name}" "python -m pip install cloudpickle fvcore iopath yacs omegaconf hydra-core termcolor tabulate tqdm pyyaml packaging pycocotools"
}

install_vit_adapter_extras() {
    local env_name="$1"
    local arch_list=""
    local build_exports=""
    ensure_repo_checkout "third_party/ViT-Adapter" "${VIT_ADAPTER_REPO_URL}" "${VIT_ADAPTER_REPO_REF}"
    ensure_command nvcc
    ensure_cuda_nvcc_toolchain "${env_name}"
    ensure_cuda_dev_libraries "${env_name}"
    arch_list="$(detect_torch_cuda_arch_list)"
    if [[ -n "${arch_list}" ]]; then
        log "Using TORCH_CUDA_ARCH_LIST=${arch_list}"
    fi
    build_exports="$(cuda_build_exports)"
    run_in_env "${env_name}" "cd '${REPO_ROOT}/third_party/ViT-Adapter/segmentation' && [ -e ops ] || ln -s ../detection/ops ./ops"
    run_in_env "${env_name}" "${build_exports} export MAKEFLAGS='-j${ENV_BUILD_JOBS}'; cd '${REPO_ROOT}/third_party/ViT-Adapter/segmentation/ops' && sh make.sh"
}

finish_env_setup() {
    local env_name="$1"
    run_in_env "${env_name}" "python -m pip install -e '${REPO_ROOT}' --no-deps"
}

finish_train_env_setup() {
    local env_name="$1"
    finish_env_setup "${env_name}"
}

finish_eval_env_setup() {
    local env_name="$1"
    run_in_env "${env_name}" "python -m pip install -e '${REPO_ROOT}' --no-deps"
    run_in_env "${env_name}" "python -m pip install -e '${REPO_ROOT}/third_party/vlm-evaluation' --no-deps"
}
