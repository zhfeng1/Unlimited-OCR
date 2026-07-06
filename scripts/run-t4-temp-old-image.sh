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
  bash -s <<CONTAINER_SCRIPT
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends gcc g++ python3.12-dev libnuma1
rm -rf /var/lib/apt/lists/*

python -m pip uninstall -y kernels-data || true
python -m pip install --no-cache-dir --no-deps --force-reinstall 'kernels==0.11.7'
python -m pip install --no-cache-dir 'addict==2.4.0' 'matplotlib==3.10.8'

python - <<'PY'
from pathlib import Path

path = Path("/opt/venv/lib/python3.12/site-packages/sglang/srt/layers/layernorm.py")
text = path.read_text()
needle = "        if _use_aiter:\n            self._forward_method = self.forward_aiter\n"
patch = (
    "        if _is_cuda and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:\n"
    "            self._forward_method = self.forward_native\n"
    "        elif _use_aiter:\n"
    "            self._forward_method = self.forward_aiter\n"
)
if patch not in text:
    if needle not in text:
        raise SystemExit("Unable to patch SGLang RMSNorm for sm75")
    path.write_text(text.replace(needle, patch))
PY

export SGLANG_IS_FLASHINFER_AVAILABLE=false

exec python -m sglang.launch_server \
  --model /model \
  --trust-remote-code \
  --served-model-name Unlimited-OCR \
  --dtype float16 \
  --attention-backend triton \
  --mm-attention-backend triton_attn \
  --sampling-backend pytorch \
  --page-size 1 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --context-length "${CONTEXT_LENGTH}" \
  --disable-cuda-graph \
  --enable-custom-logit-processor \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --host 0.0.0.0 \
  --port "${CONTAINER_PORT}"
CONTAINER_SCRIPT
