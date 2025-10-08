# Kulyk Translator using Rust

This is a Rust-based translator that utilizes the `llama-cpp-2` crate to inference Kulyk models.

Created for the Rustcamp 2025 Summer: https://github.com/rust-lang-ua/rustcamp_ml

## Demo

<a href="./screenshot.png"><img src="./screenshot.png" width="700px"/></a>

## Models

- Models: 
    - https://huggingface.co/Yehor/kulyk-uk-en
    - https://huggingface.co/Yehor/kulyk-en-uk
- GGUF Models:
    - https://huggingface.co/mradermacher/kulyk-uk-en-GGUF
    - https://huggingface.co/mradermacher/kulyk-en-uk-GGUF

## Build

```shell
cargo build --release
```

## Usage

```shell
# this command will download a quantized GGUF model
just download_model

# start web server
just run

# open the ui.html file in your browser
```

## Acknowledgements

This project is based on the following repositories:

- [`simple` example from llama-cpp-2](https://github.com/utilityai/llama-cpp-rs/tree/main/examples/simple)
