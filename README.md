# Kulyk Translator using Rust

This is a Rust-based translator that utilizes the `llama-cpp-2` crate to inference Kulyk models.

Created for the Rustcamp 2025 Summer: https://github.com/rust-lang-ua/rustcamp_ml

## Models

- Model with hyperparameters: https://huggingface.co/Yehor/kulyk-uk-en
- GGUF Models: https://huggingface.co/mradermacher/kulyk-uk-en-GGUF

## Build

```shell
cargo build --release
```

## Usage

```shell
# this command will download a quantized GGUF model
just download_model

# start web server
./target/release/kulyk --verbose --threads 1 --threads-batch 1 --n-len 1024 models/kulyk-uk-en.Q8_0.gguf

# open ui.html in your browser
```
