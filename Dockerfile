# ==============================================================================
# Multi-Arch Dockerfile for kulyk-rust
# Supports building for linux/amd64 and linux/arm64 platforms.
#
# Usage:
#
#   docker buildx create --use  # (if not already set up)
#   docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/egorsmkv/kulyk-rust:latest --push .
#
# Load built images to localhost for both architectures:
#
#   docker buildx build --platform linux/amd64 -t kulyk-rust:latest-amd64 . --load
#   docker buildx build --platform linux/arm64 -t kulyk-rust:latest-arm64 . --load
#
# Run:
#
#   docker run --rm --platform linux/amd64 -p 3000:3000 kulyk-rust:latest-amd64
#   docker run --rm --platform linux/arm64 -p 3000:3000 kulyk-rust:latest-arm64
#
# This produces a multi-arch manifest that can be pushed to a registry like GHCR.
# For single-platform builds (e.g., `docker build -t kulyk-rust:latest .`), it
# automatically detects the host architecture via `uname -m`.
# ==============================================================================

# ==============================================================================
# Stage 1: Downloader
# Downloads the GGUF models using temporary dependencies (wget, ca-certificates).
# ==============================================================================
FROM debian:bookworm-slim as downloader

# Install temporary dependencies for downloading models.
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download models to a staging directory.
RUN mkdir -p /download/models && \
    wget -O /download/models/kulyk-uk-en.gguf "https://huggingface.co/mradermacher/kulyk-uk-en-GGUF/resolve/main/kulyk-uk-en.Q8_0.gguf" && \
    wget -O /download/models/kulyk-en-uk.gguf "https://huggingface.co/mradermacher/kulyk-en-uk-GGUF/resolve/main/kulyk-en-uk.Q8_0.gguf"

# ==============================================================================
# Stage 2: Runner
# Creates the final minimal image with runtime dependencies, models, and the
# architecture-specific binary. Runs as non-root user for security.
# ==============================================================================
FROM debian:bookworm-slim as runner

# Add metadata labels for better image introspection.
LABEL maintainer="egorsmkv"
LABEL org.opencontainers.image.source="https://github.com/egorsmkv/kulyk-rust"
LABEL org.opencontainers.image.description="A translator application using local models, supporting Ukrainian and English."

# Install runtime dependencies (libgomp1 for OpenMP support in llama.cpp).
# ca-certificates omitted as the app uses local models and has no outbound HTTPS needs.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy models from the downloader stage.
COPY --from=downloader --chown=1000:1000 /download/models /app/models

# Create a non-root user for running the container (principle of least privilege).
# UID 1000 is standard for non-root; home dir set to /app for simplicity.
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -d /app appuser && \
    chown -R appuser:appuser /app

# Copy pre-compiled binaries for both architectures into temporary locations.
# These are selected based on the build-time architecture detection during the build.
COPY ./dist/kulyk_x86_64-unknown-linux-gnu/kulyk /tmp/amd64/kulyk
COPY ./dist/kulyk_aarch64-unknown-linux-gnu/kulyk /tmp/arm64/kulyk

# Select the appropriate binary based on the target architecture, set permissions,
# and chown to non-root user. Uses `uname -m` for architecture detection, which
# works in both multi-platform (Buildx) and single-platform builds.
# Maps "x86_64" to amd64 binary and "aarch64" to arm64 binary.
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        cp /tmp/amd64/kulyk /app/kulyk-translator; \
    elif [ "$arch" = "aarch64" ]; then \
        cp /tmp/arm64/kulyk /app/kulyk-translator; \
    else \
        echo "Unsupported architecture: $arch" && exit 1; \
    fi && \
    chown appuser:appuser /app/kulyk-translator && \
    chmod 755 /app/kulyk-translator && \
    rm -rf /tmp/amd64 /tmp/arm64

# Switch to non-root user.
USER appuser

# Expose the port the server listens on.
EXPOSE 3000

# Set the entrypoint to run the translation server.
CMD ["./kulyk-translator", "--verbose", "--n-len", "1024", "--model-path-ue", "/app/models/kulyk-uk-en.gguf", "--model-path-eu", "/app/models/kulyk-en-uk.gguf"]
