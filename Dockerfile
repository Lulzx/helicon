# Helicon — reproducible build container
#
# Usage:
#   docker build -t helicon .
#   docker run --rm -v $(pwd)/results:/app/results helicon \
#       helicon run --preset dfd --output results/dfd
#
# WarpX is NOT included in this image (it requires platform-specific
# compilation for CUDA/OpenMP). Mount a pre-built WarpX install or
# install pywarpx inside the container for simulation runs.
# This image is sufficient for:
#   - Field computation (Biot-Savart)
#   - Post-processing
#   - Parameter scans (dry_run=True)
#   - Optimization and analytics

FROM python:3.12-slim

LABEL maintainer="Helicon Contributors"
LABEL description="GPU-Accelerated Magnetic Nozzle Simulation & Detachment Analysis Toolkit"
LABEL org.opencontainers.image.source="https://github.com/helicon/helicon"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY helicon/ ./helicon/

# Install helicon and dependencies (no MLX — CPU fallback only in Docker)
RUN uv pip install --system --no-cache \
    numpy scipy pydantic pyyaml click h5py \
    scikit-learn \
    && uv pip install --system --no-cache -e .

# Optional: install WarpX Python bindings if pre-built wheel is available
# COPY warpx-*.whl /tmp/
# RUN uv pip install --system /tmp/warpx-*.whl

# Create results directory
RUN mkdir -p /app/results

VOLUME ["/app/results"]

ENTRYPOINT ["helicon"]
CMD ["--help"]
