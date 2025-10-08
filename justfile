download_model:
    mkdir -p models/
    wget -O "models/kulyk-uk-en.Q8_0.gguf" "https://huggingface.co/mradermacher/kulyk-uk-en-GGUF/resolve/main/kulyk-uk-en.Q8_0.gguf"

build_release:
    cargo build --release

run: build_release
    ./target/release/kulyk --verbose --threads 1 --threads-batch 1 --n-len 1024 models/kulyk-uk-en.Q8_0.gguf
