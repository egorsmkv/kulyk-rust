use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tracing::{error, info};

use anyhow::{bail, Context, Result};
use clap::Parser;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The path to the kulyk-uk-en model
    #[arg(long, help = "Path to kulyk-uk-en model")]
    model_path_ue: PathBuf,

    /// The path to the kulyk-en-uk model
    #[arg(long, help = "Path to kulyk-en-uk model")]
    model_path_eu: PathBuf,

    /// Set the length of the prompt + output in tokens
    #[arg(long, default_value_t = 32)]
    n_len: i32,

    /// Disable offloading layers to the gpu
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[clap(long)]
    disable_gpu: bool,

    /// Seed value
    #[arg(short = 's', long, help = "RNG seed (default: 1234)")]
    seed: Option<u32>,

    /// Number of threads
    #[arg(
        short = 't',
        long,
        help = "number of threads to use during generation (default: use all available threads)"
    )]
    threads: Option<i32>,

    /// Number of threads for batching
    #[arg(
        long,
        help = "number of threads to use during batch and prompt processing (default: use all available threads)"
    )]
    threads_batch: Option<i32>,

    /// Context size
    #[arg(
        short = 'c',
        long,
        help = "size of the prompt context (default: loaded from themodel)"
    )]
    ctx_size: Option<NonZeroU32>,

    /// Enable verbose llama.cpp logs
    #[arg(short = 'v', long, help = "enable verbose llama.cpp logs")]
    verbose: bool,

    /// Port number for the server
    #[arg(short = 'p', long, help = "Port number for the server (default: 3000)")]
    port: Option<u16>,
}

struct TranslationModel {
    args: Args,
    model_ue: LlamaModel,
    backend: LlamaBackend,
    model_eu: LlamaModel,
}

impl TranslationModel {
    fn new(args: Args) -> Result<Self> {
        let backend = LlamaBackend::init()?;

        let model_ue = {
            let model_params = {
                #[cfg(any(feature = "cuda", feature = "vulkan"))]
                if !args.disable_gpu {
                    LlamaModelParams::default().with_n_gpu_layers(1000)
                } else {
                    LlamaModelParams::default()
                }
                #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
                LlamaModelParams::default()
            };
            LlamaModel::load_from_file(&backend, &args.model_path_ue, &model_params)
                .with_context(|| "unable to load uk-en model")?
        };

        let model_eu = {
            let model_params = {
                #[cfg(any(feature = "cuda", feature = "vulkan"))]
                if !args.disable_gpu {
                    LlamaModelParams::default().with_n_gpu_layers(1000)
                } else {
                    LlamaModelParams::default()
                }
                #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
                LlamaModelParams::default()
            };
            LlamaModel::load_from_file(&backend, &args.model_path_eu, &model_params)
                .with_context(|| "unable to load en-uk model")?
        };

        Ok(Self {
            args,
            model_ue,
            backend,
            model_eu,
        })
    }

    fn translate(&self, text: &str, source_lang: &str, target_lang: &str) -> Result<String> {
        info!(
            "Translating text from '{}' to '{}'",
            source_lang, target_lang
        );

        let prompt = fill_prompt(text, target_lang);
        info!("prompt: {}", prompt);

        let model_to_use = if source_lang == "uk" && target_lang == "en" {
            &self.model_ue
        } else if source_lang == "en" && target_lang == "uk" {
            &self.model_eu
        } else {
            bail!(
                "Unsupported translation direction: {} to {}",
                source_lang,
                target_lang
            );
        };

        self.translate_text(model_to_use, prompt)
    }

    fn translate_text(&self, model: &LlamaModel, prompt: String) -> Result<String> {
        let args = &self.args;

        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(args.ctx_size.or(Some(NonZeroU32::new(2048).unwrap())));
        if let Some(threads) = args.threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = args.threads_batch.or(args.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        let mut ctx = model
            .new_context(&self.backend, ctx_params)
            .with_context(|| "unable to create the llama_context")?;

        let tokens_list = model
            .str_to_token(&prompt, AddBos::Always)
            .with_context(|| format!("failed to tokenize {prompt}"))?;

        let n_cxt = ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (args.n_len - tokens_list.len() as i32);

        if n_kv_req > n_cxt {
            bail!(
                "n_kv_req > n_ctx, the required kv cache size is not big enough; either reduce n_len or increase n_ctx"
            );
        }
        if tokens_list.len() >= usize::try_from(args.n_len)? {
            bail!("the prompt is too long, it has more tokens than n_len");
        }

        let mut batch = LlamaBatch::new(512, 1);
        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        let mut n_cur = batch.n_tokens();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::dist(args.seed.unwrap_or(1234)),
            LlamaSampler::greedy(),
        ]);
        let mut output = String::new();

        while n_cur <= args.n_len {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if model.is_eog_token(token) {
                break;
            }
            let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
            let mut output_string = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            output.push_str(&output_string);
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch).with_context(|| "failed to eval")?;
        }

        Ok(output.trim().to_string())
    }
}

#[derive(Deserialize, Debug)]
struct TranslateRequest {
    text: String,
    source_lang: String,
    target_lang: String,
}

#[derive(Serialize, Debug)]
struct TranslateResponse {
    translated_text: String,
    source_lang: String,
    target_lang: String,
}

fn fill_prompt(text: &str, target: &str) -> String {
    if target == "en" {
        format!(
            "<|im_start|>user\nTranslate the text to English:\n{text}<|im_end|>\n<|im_start|>assistant"
        )
    } else if target == "uk" {
        format!(
            "<|im_start|>user\nTranslate the text to Ukrainian:\n{text}<|im_end|>\n<|im_start|>assistant"
        )
    } else {
        panic!("Unsupported target language: {}", target);
    }
}

/// Handler to serve the HTML frontend.
async fn root_handler() -> Html<&'static str> {
    Html(include_str!("../ui.html"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        tracing_subscriber::fmt::init();
    }
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(args.verbose));

    let port: u16 = args.port.unwrap_or(3000);

    let model = Arc::new(TranslationModel::new(args)?);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(root_handler)) // Serve the frontend
        .route("/translate", post(translate_handler))
        .with_state(model)
        .layer(cors);

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await.unwrap();
    info!("Server listening on http://{}", addr);

    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();

    Ok(())
}

/// API handler for translation requests.
async fn translate_handler(
    State(model): State<Arc<TranslationModel>>,
    Json(payload): Json<TranslateRequest>,
) -> impl IntoResponse {
    info!("Received translation request: {:?}", payload);

    match model.translate(&payload.text, &payload.source_lang, &payload.target_lang) {
        Ok(translated_text) => {
            let response = TranslateResponse {
                translated_text,
                source_lang: payload.source_lang,
                target_lang: payload.target_lang,
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            error!("Translation failed: {:?}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Translation failed: {}", e),
            )
                .into_response()
        }
    }
}
