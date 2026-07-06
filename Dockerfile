# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive

ENV PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-sglang.txt ./
COPY wheel/ ./wheel/
RUN python3.12 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && mkdir /wheelhouse \
    && python -m pip wheel --wheel-dir /wheelhouse -r requirements-sglang.txt

FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV HF_HOME=/home/unlimited/.cache/huggingface \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SGLANG_IS_FLASHINFER_AVAILABLE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        gcc \
        git \
        libgl1 \
        libglib2.0-0 \
        libnuma1 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /wheelhouse /wheelhouse
RUN python3.12 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-index --find-links=/wheelhouse \
        "sglang==0.0.0.dev11416+g92e8bb79e" \
        "addict==2.4.0" \
        "matplotlib==3.10.8" \
        "pymupdf==1.27.2.2" \
    && rm -rf /wheelhouse

RUN python - <<'PY'
from pathlib import Path

layernorm_path = Path("/opt/venv/lib/python3.12/site-packages/sglang/srt/layers/layernorm.py")
text = layernorm_path.read_text()
layernorm_needle = "        if _use_aiter:\n            self._forward_method = self.forward_aiter\n"
layernorm_patch = (
    "        if _is_cuda and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:\n"
    "            self._forward_method = self.forward_native\n"
    "        elif _use_aiter:\n"
    "            self._forward_method = self.forward_aiter\n"
)
if layernorm_patch not in text:
    if layernorm_needle not in text:
        raise SystemExit("Unable to patch SGLang RMSNorm for sm75")
    layernorm_path.write_text(text.replace(layernorm_needle, layernorm_patch))

ocr_path = Path("/opt/venv/lib/python3.12/site-packages/sglang/srt/models/unlimited_ocr.py")
text = ocr_path.read_text()
dtype_needle = "        target_dtype = self.vision_model.dtype\n"
dtype_patch = "        target_dtype = next(self.sam_model.parameters()).dtype\n"
if dtype_patch not in text:
    if dtype_needle not in text:
        raise SystemExit("Unable to patch Unlimited-OCR target dtype")
    text = text.replace(dtype_needle, dtype_patch)
ocr_needle = "                patches = images_crop[jdx][0].to(torch.bfloat16)\n"
ocr_patch = "                patches = images_crop[jdx][0].to(dtype=next(self.sam_model.parameters()).dtype)\n"
if ocr_patch not in text:
    if ocr_needle not in text:
        raise SystemExit("Unable to patch Unlimited-OCR vision dtype")
    text = text.replace(ocr_needle, ocr_patch)
ocr_path.write_text(text)
PY

WORKDIR /app

COPY infer.py README.md LICENSE ./

RUN if getent group "${GROUP_ID}" >/dev/null; then \
        groupmod --new-name unlimited "$(getent group "${GROUP_ID}" | cut -d: -f1)"; \
    else \
        groupadd --gid "${GROUP_ID}" unlimited; \
    fi \
    && if getent passwd "${USER_ID}" >/dev/null; then \
        usermod --login unlimited --move-home --home /home/unlimited --gid "${GROUP_ID}" "$(getent passwd "${USER_ID}" | cut -d: -f1)"; \
    else \
        useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash unlimited; \
    fi \
    && mkdir -p /app/log /app/outputs "${HF_HOME}" \
    && chown -R unlimited:unlimited /app /home/unlimited

USER unlimited

EXPOSE 10000
VOLUME ["/data", "/app/outputs", "/app/log", "/home/unlimited/.cache/huggingface"]

CMD ["python", "-m", "sglang.launch_server", \
    "--model", "baidu/Unlimited-OCR", \
    "--trust-remote-code", \
    "--served-model-name", "Unlimited-OCR", \
    "--dtype", "float16", \
    "--attention-backend", "triton", \
    "--mm-attention-backend", "triton_attn", \
    "--sampling-backend", "pytorch", \
    "--page-size", "1", \
    "--mem-fraction-static", "0.8", \
    "--context-length", "8192", \
    "--disable-cuda-graph", \
    "--enable-custom-logit-processor", \
    "--disable-overlap-schedule", \
    "--skip-server-warmup", \
    "--host", "0.0.0.0", \
    "--port", "10000"]
