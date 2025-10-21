download_models:
    mkdir -p models/
    wget -O "models/kulyk-uk-en.Q8_0.gguf" "https://huggingface.co/Yehor/kulyk-uk-en/resolve/main/kulyk-uk-en-q8_0.gguf"
    wget -O "models/kulyk-en-uk.Q8_0.gguf" "https://huggingface.co/Yehor/kulyk-en-uk/resolve/main/kulyk-en-uk-q8_0.gguf"

build_release:
    cargo build --release

run: build_release
    ./target/release/kulyk --verbose --threads 1 --threads-batch 1 --n-len 1024 --model-path-ue models/kulyk-uk-en.Q8_0.gguf --model-path-eu models/kulyk-en-uk.Q8_0.gguf
