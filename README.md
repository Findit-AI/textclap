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

// Query-only worker:
async fn run_query(user_query: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = TextEncoder::from_files(
        "text_model_quantized.onnx", "tokenizer.json", Options::new(),
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

## Model files

textclap loads three files at runtime. They are **not bundled** with the crate. Download from
[Xenova/clap-htsat-unfused](https://huggingface.co/Xenova/clap-htsat-unfused) and verify SHA256:

| File                         | Size   | SHA256                                              |
|------------------------------|--------|-----------------------------------------------------|
| `audio_model_quantized.onnx` | 33 MB  | *(see `tests/fixtures/MODELS.md`)*                  |
| `text_model_quantized.onnx`  | 121 MB | *(see `tests/fixtures/MODELS.md`)*                  |
| `tokenizer.json`             | 2.0 MB | *(see `tests/fixtures/MODELS.md`)*                  |

**The `tokenizer.json` must come from the same Xenova export** — *not* from
`laion/clap-htsat-unfused` directly. They differ subtly and produce token-id mismatches that pass tests
on common English but break on edge cases.

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
