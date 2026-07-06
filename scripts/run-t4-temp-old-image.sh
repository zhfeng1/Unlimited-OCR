#!/usr/bin/env bash
set -euo pipefail

GPU_DEVICE="${GPU_DEVICE:-1}"
HOST_PORT="${HOST_PORT:-8011}"
CONTAINER_PORT="${CONTAINER_PORT:-10000}"
IMAGE="${IMAGE:-ghcr.io/zhfeng1/unlimited-ocr:sha-50fad0c}"
MODEL_DIR="${MODEL_DIR:-$PWD/model}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"

mkdir -p data outputs log .cache/huggingface

docker run --rm \
  --user root \
  --gpus "\"device=${GPU_DEVICE}\"" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  --ipc host \
  -v "${MODEL_DIR}:/model:ro" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/log:/app/log" \
  -v "$PWD/.cache/huggingface:/home/unlimited/.cache/huggingface" \
  "${IMAGE}" \
  bash -lc "apt-get update && apt-get install -y --no-install-recommends libnuma1 && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip uninstall -y kernels-data || true && \
    python -m pip install --no-cache-dir --no-deps --force-reinstall 'kernels==0.11.7' && \
    python -m pip install --no-cache-dir 'addict==2.4.0' 'matplotlib==3.10.8' && \
    exec python -m sglang.launch_server \
      --model /model \
      --trust-remote-code \
      --served-model-name Unlimited-OCR \
      --attention-backend triton \
      --mm-attention-backend triton_attn \
      --page-size 1 \
      --mem-fraction-static '${MEM_FRACTION_STATIC}' \
      --context-length '${CONTEXT_LENGTH}' \
      --enable-custom-logit-processor \
      --disable-overlap-schedule \
      --skip-server-warmup \
      --host 0.0.0.0 \
      --port '${CONTAINER_PORT}'"
