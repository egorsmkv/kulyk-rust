# docker build -t kulyk-translator .
# docker run -p 3000:3000 --rm kulyk-translator


# Stage 1: Builder
# This stage compiles the Rust application.
FROM rust:1.90-slim as builder

# Install build dependencies required by llama-cpp-sys-2
RUN apt-get update && apt-get install -y build-essential cmake pkg-config llvm libclang-dev

WORKDIR /app

# Copy manifests and build dependencies to cache them
COPY Cargo.toml Cargo.lock ./

# Create a dummy project to build only dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release --locked

# Remove dummy source and copy the actual source code
COPY src ./src

# Build the application
RUN cargo build --release --locked

# Stage 2: Runner
# This stage creates the final, smaller image.
FROM ubuntu:25.10 as runner

WORKDIR /app

# Install runtime dependencies (e.g., for GPU support if needed, though this setup is CPU-only)
# and tools to download models.
RUN apt-get update && apt-get install -y wget libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*

# Create a directory for models
RUN mkdir -p /app/models

# Download the GGUF models from Hugging Face.
# Using Q4_K_M as a good balance of size and quality.
RUN wget -O /app/models/kulyk-uk-en.gguf "https://huggingface.co/mradermacher/kulyk-uk-en-GGUF/resolve/main/kulyk-uk-en.Q8_0.gguf"
RUN wget -O /app/models/kulyk-en-uk.gguf "https://huggingface.co/mradermacher/kulyk-en-uk-GGUF/resolve/main/kulyk-en-uk.Q8_0.gguf"

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/kulyk .

# Expose the port the server listens on
EXPOSE 3000

# Set the entrypoint to run the translation server
CMD ["./kulyk", "--verbose", "--n-len", "1024", "--model-path-ue", "/app/models/kulyk-uk-en.gguf", "--model-path-eu", "/app/models/kulyk-en-uk.gguf"]
