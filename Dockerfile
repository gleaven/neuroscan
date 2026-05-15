ARG CUDA_VERSION=13.0.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_ARCH=12.1
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.9.1+cu130 — only stable with aarch64 + sm_121 (GB10 Blackwell)
RUN pip3 install --no-cache-dir \
    torch==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python3", "server.py"]
