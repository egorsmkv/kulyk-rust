# syntax=docker/dockerfile:1

# Dockerfile for building a multi-architecture (amd64, arm64) Rust application.
#
# ==============================================================================
# Build and Push Commands (using docker buildx):
#
# 1. (One-time setup) Create and switch to a new builder instance:
#    docker buildx create --use --name multi-arch-builder
#
# 2. Build for both platforms and load locally:
#    docker buildx build --platform linux/amd64 -t kulyk-rust:latest --load .
#    docker buildx build --platform linux/arm64 -t kulyk-rust:latest --load .
#
# 3. Build for both platforms and push to a registry:
#    docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/egorsmkv/kulyk-rust:latest --push .
#
# ==============================================================================
# Local Build Command (for your machine's native architecture):
#
#    docker build -t kulyk-rust:latest .
#
# Check linked shared libraries:
#
#    docker run --platform linux/amd64 --rm -it --entrypoint /bin/bash kulyk-rust:latest
#    > ldd /app/kulyk-translator
#
# ==============================================================================
# Run Command:
#
#    docker run -p 3000:3000 --rm kulyk-rust:latest
# ==============================================================================


# ==============================================================================
# Stage 1: Builder
# This stage compiles the Rust application for the target architecture.
# ==============================================================================
FROM rust:1.90-slim as builder

# --- Build arguments for multi-arch support ---
# TARGETARCH is an automatic platform ARG provided by Docker buildx.
# It will be 'amd64' or 'arm64'.
ARG TARGETARCH

# Set the Rust target triple based on the architecture and install the toolchain.
# This ensures we are compiling for the correct platform.
RUN <<EOF
set -e
case "$TARGETARCH" in \
    "amd64") RUST_TARGET="x86_64-unknown-linux-gnu";; \
    "arm64") RUST_TARGET="aarch64-unknown-linux-gnu";; \
    *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1;; \
esac
echo "Building for architecture: $TARGETARCH, Rust target: $RUST_TARGET"
rustup target add $RUST_TARGET
EOF

# --- Install build dependencies ---
# Dependencies required by llama-cpp-sys-2 for CPU-based builds.
# Combining update, install, and cleanup into a single layer for efficiency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    pkg-config \
    llvm \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Cache dependencies ---
# Copy manifests and build dependencies to cache them in a separate layer.
COPY .cargo ./.cargo
COPY Cargo.toml Cargo.lock ./

# Create a dummy project to build and cache only the dependencies.
# This prevents re-downloading/re-compiling deps on every source code change.
RUN <<EOF
set -e
case "$TARGETARCH" in \
    "amd64") RUST_TARGET="x86_64-unknown-linux-gnu";; \
    "arm64") RUST_TARGET="aarch64-unknown-linux-gnu";; \
esac
mkdir src
echo "fn main() {}" > src/main.rs
cargo build --target $RUST_TARGET --release --locked
EOF


# --- Build the application ---
# Remove the dummy source and copy the actual application source code.
RUN rm -rf src
COPY src ./src

# Build the application binary.
# The --locked flag ensures reproducible builds using Cargo.lock.
RUN <<EOF
set -e
case "$TARGETARCH" in \
    "amd64") RUST_TARGET="x86_64-unknown-linux-gnu";; \
    "arm64") RUST_TARGET="aarch64-unknown-linux-gnu";; \
esac
cargo build --target $RUST_TARGET --release --locked
# Copy the final binary to a known location to simplify the next stage.
cp "/app/target/$RUST_TARGET/release/kulyk" /app/kulyk-translator
EOF


# ==============================================================================
# Stage 2: Runner
# This stage creates the final, smaller image with the compiled binary.
# ==============================================================================
FROM ubuntu:24.04 as runner

# Using Ubuntu 24.04 LTS for better long-term stability.

# Add metadata labels for better image introspection.
LABEL maintainer="egorsmkv"
LABEL org.opencontainers.image.source="https://github.com/egorsmkv/kulyk-rust"
LABEL org.opencontainers.image.description="A translator application using local models, supporting Ukrainian and English."

WORKDIR /app

# --- Install runtime dependencies ---
# wget for downloading models, libgomp1 for OpenMP support required by llama.cpp.
RUN apt-get update && apt-get install -y \
    wget \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Download models ---
# Create a directory for models and download the GGUF models from Hugging Face.
RUN mkdir -p /app/models
RUN wget -O /app/models/kulyk-uk-en.gguf "https://huggingface.co/mradermacher/kulyk-uk-en-GGUF/resolve/main/kulyk-uk-en.Q8_0.gguf"
RUN wget -O /app/models/kulyk-en-uk.gguf "https://huggingface.co/mradermacher/kulyk-en-uk-GGUF/resolve/main/kulyk-en-uk.Q8_0.gguf"

# --- Copy the compiled binary ---
# Copy the compiled binary from the builder stage's known location.
COPY --from=builder /app/kulyk-translator .

# --- Configure and Run ---
# Expose the port the server listens on.
EXPOSE 3000

# Set the entrypoint to run the translation server.
CMD ["./kulyk-translator", "--verbose", "--n-len", "1024", "--model-path-ue", "/app/models/kulyk-uk-en.gguf", "--model-path-eu", "/app/models/kulyk-en-uk.gguf"]
