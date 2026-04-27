# textclap

[![crates.io](https://img.shields.io/crates/v/textclap.svg)](https://crates.io/crates/textclap)
[![docs.rs](https://docs.rs/textclap/badge.svg)](https://docs.rs/textclap)
[![CI](https://github.com/Findit-AI/textclap/actions/workflows/ci.yml/badge.svg)](https://github.com/Findit-AI/textclap/actions/workflows/ci.yml)

Rust ONNX-inference library for the LAION
[CLAP-HTSAT-unfused](https://huggingface.co/laion/clap-htsat-unfused) model. Ships the audio (HTSAT)
and text (RoBERTa) encoders behind one library, plus a zero-shot classification helper. Designed for
the indexing-and-query pipeline shape described below.

## Pipeline

```text
                                  ┌─ audio path (fixed 10 s windows of source audio) ─┐
                                  │   decoder → resample to 48 kHz → buffer 10 s      │
                                  │       → AudioEncoder::embed → 512-dim embedding   │
audio frames (native rate) ──────►│                                                    │
                                  │                                                    │
                                  │   (text path, query-time only):                    │
                                  │       user query → TextEncoder::embed → 512-dim    │
                                  │       → cosine search against audio_embedding col │
                                  └────────────────────────────────────────────────────┘
                                                          ↓
                                                  lancedb / vector store
```

The audio encoder runs every 10 s of input. The text encoder runs **once per user search query** —
it does not embed Whisper transcripts or any STT output. See the spec at
`docs/superpowers/specs/2026-04-25-textclap-clap-inference-design.md` for the full architecture.

## Quick start

The simplest path uses the crate's bundled tokenizer — supply only the two ONNX files:

```rust,no_run
use textclap::{Clap, Options};

async fn run(user_query: &str, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut clap = Clap::from_onnx_files(
        "audio_model_quantized.onnx",
        "text_model_quantized.onnx",
        Options::new(),
    )?;
    clap.warmup()?;
    let labels = ["a dog barking", "rain", "music"];
    let scores = clap.classify(samples, &labels, 3)?;
    let _q = clap.text_mut().embed(user_query)?;
    Ok(())
}
```

Indexing-only and query-only workers can skip the unused encoder by going through
`AudioEncoder::from_file` or `TextEncoder::from_onnx_file` directly:

```rust,no_run
use textclap::{AudioEncoder, Options, TextEncoder};

// Indexing-only worker (saves ~120 MB resident vs Clap):
fn run_indexer() -> Result<(), Box<dyn std::error::Error>> {
    let mut audio = AudioEncoder::from_file("audio_model_quantized.onnx", Options::new())?;
    audio.warmup()?;
    loop {
        let pcm = decoder.next_10s_at_48khz()?;          // caller-supplied
        let emb = audio.embed(&pcm)?;
        lancedb_writer.push(emb.as_slice(), ts_start, ts_end)?;
    }
}

// Query-only worker (uses the bundled tokenizer; no separate file needed):
async fn run_query(user_query: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = TextEncoder::from_onnx_file(
        "text_model_quantized.onnx", Options::new(),
    )?;
    text.warmup()?;
    let q = text.embed(user_query)?;
    let _hits = table.search(q.to_vec()).limit(10).execute().await?;
    Ok(())
}
```

The `Box<dyn std::error::Error>` wrapper lets `?` mix textclap errors with caller-supplied decoder /
lancedb errors without writing `From` impls. In library code, wrap textclap's `Error` in your own
concrete error type.

### Advanced: bring your own tokenizer

If you need a different tokenizer revision, use the 3-path `Clap::from_files` (or the matching
`TextEncoder::from_files`) and supply your own `tokenizer.json`:

```rust,no_run
use textclap::{Clap, Options};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let _clap = Clap::from_files(
        "audio_model_quantized.onnx",
        "text_model_quantized.onnx",
        "tokenizer.json",
        Options::new(),
    )?;
    Ok(())
}
```

The pinned tokenizer bytes shipped with the crate are also exposed as the
`textclap::BUNDLED_TOKENIZER` `&'static [u8]` constant — useful when constructing via `from_memory`.

## Model files

textclap loads two ONNX files at runtime; the third file (`tokenizer.json`) is **bundled with the
crate**. Download the ONNX files from
[Xenova/clap-htsat-unfused](https://huggingface.co/Xenova/clap-htsat-unfused) and verify SHA256:

| File                         | Size   | SHA256                                              |
|------------------------------|--------|-----------------------------------------------------|
| `audio_model_quantized.onnx` | 33 MB  | *(see `models/MODELS.md`)*                          |
| `text_model_quantized.onnx`  | 121 MB | *(see `models/MODELS.md`)*                          |
| `tokenizer.json`             | 2.0 MB | **bundled with the crate** — no download needed     |

**The bundled `tokenizer.json` comes from the Xenova export** — *not* from
`laion/clap-htsat-unfused` directly. They differ subtly and produce token-id mismatches that pass tests
on common English but break on edge cases. If you need to override it, see the advanced section above.

## Deployment

Thread-per-core. Each worker thread loads its own encoder once at startup; expect **150–300 MB
resident per worker** (33 MB int8 audio + 121 MB int8 text + ORT working buffers). Build workers
**sequentially** at startup to avoid transient 2× peak memory during ORT weight reformatting.

For thread tuning or execution-provider selection (CUDA, CoreML), build your own `ort::Session`
directly and pass it via `from_ort_session`. textclap deliberately does not expose `with_intra_threads`
to keep deployment-specific runtime policy outside the API.

## Sibling-crate coupling

`ort = "2.0.0-rc.12"` (caret) matches sibling crates `silero` (VAD) and `soundevents` (sound classification).
Bumping `ort` requires a coordinated change across the trio.

## Model attribution

Downstream users redistributing model files take on the upstream attribution responsibilities:
- LAION CLAP weights: **CC-BY 4.0** — attribution required when redistributing.
- Xenova ONNX export: **Apache-2.0**.
- HTSAT and CLAP papers: citation required.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
