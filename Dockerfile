FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System dependencies for running the API server
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (used for all Python dependency installs)
RUN python3 -m pip install --no-cache-dir "uv==0.9.26"

# Create and use a dedicated virtualenv
RUN uv venv "/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"

# Install a driver-compatible PyTorch CUDA wheel (avoid cu129+)
ARG TORCH_VERSION="2.5.1"
ARG TORCH_CUDA="cu121"
ARG TORCHVISION_VERSION="0.20.1"
ARG TRITON_VERSION="3.1.0"
ARG LMDEPLOY_VERSION="0.11.1"
RUN uv pip install --no-cache \
    "setuptools" \
    "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
    "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}" \
    "torchaudio==${TORCH_VERSION}+${TORCH_CUDA}" \
    "triton==${TRITON_VERSION}" \
    "lmdeploy==${LMDEPLOY_VERSION}" \
    --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# Install application dependencies (without torch to avoid overrides)
COPY "pyproject.toml" "/workspace/pyproject.toml"
COPY "README.md" "/workspace/README.md"
COPY "app" "/workspace/app"
RUN uv pip install --no-cache -e "/workspace"

# Copy code for standalone usage (compose may still mount it)
COPY "scripts" "/workspace/scripts"
RUN chmod +x "/workspace/scripts/entrypoint.sh"

EXPOSE 8000
ENTRYPOINT ["/bin/bash"]
CMD ["/workspace/scripts/entrypoint.sh"]
