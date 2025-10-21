# Kulyk Translator using Rust

This is a Rust-based translator that utilizes the `llama-cpp-2` crate to inference Kulyk models.

Created for the Rustcamp 2025 Summer: https://github.com/rust-lang-ua/rustcamp_ml

## Demo

<a href="./screenshot.png"><img src="./screenshot.png" width="700px"/></a>

## Models

- Models: 
    - https://huggingface.co/Yehor/kulyk-uk-en
    - https://huggingface.co/Yehor/kulyk-en-uk

## Run using Docker (CPU)

```shell
docker run -p 3000:3000 --rm ghcr.io/egorsmkv/kulyk-rust:latest
```

**Run using Apptainer (CUDA)**:

1. Run it using shell:

```
apptainer shell --nv ./kulyk.sif

Apptainer> /opt/entrypoints/kulyk --verbose --n-len 1024 --model-path-ue /project/models/kulyk-uk-en.gguf --model-path-eu /project/models/kulyk-en-uk.gguf
```

2. Run it as a webservice:

```
apptainer instance run --nv ./kulyk.sif kulyk-ws

# go to http://localhost:3000
```

## Build

```shell
cargo build --release
```

### Cross build

#### Build images

```
docker build --platform=linux/amd64 -f dockerfiles/Dockerfile.aarch64-unknown-linux-gnu -t aarch64-unknown-linux-gnu:my-edge .
docker build --platform=linux/amd64 -f dockerfiles/Dockerfile.x86_64-unknown-linux-gnu -t x86_64-unknown-linux-gnu:my-edge .
```

#### Build binaries

```
cargo install cross --git https://github.com/cross-rs/cross

goreleaser build --clean --snapshot --id kulyk --timeout 60m
```

## Usage

```shell
# this command will download quantized GGUF models
just download_models

# start web server and navigate to http://localhost:3000 in your browser
just run
```

## High-Load test

- Test set: https://huggingface.co/datasets/speech-uk/text-to-speech-sentences
- Threads: 16
- GPU: NVIDIA GeForce RTX 3090
- Memory usage: 1190MiB
- Average inference speed per sentence: 0.0537 sec = 53.7 ms

## Acknowledgements

This project is based on the following repositories:

- [`simple` example from llama-cpp-2](https://github.com/utilityai/llama-cpp-rs/tree/main/examples/simple)
