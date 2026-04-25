# textclap — CLAP Inference Library Design

**Status:** Draft
**Date:** 2026-04-25
**Target version:** 0.1.0

## 1. Purpose

textclap is a Rust inference library for **CLAP** (Contrastive Language-Audio Pre-training), specifically the
`Xenova/clap-htsat-unfused` ONNX export of LAION's `clap-htsat-unfused` model. It exposes both the audio
(HTSAT) and text (RoBERTa) encoders behind one library, plus a zero-shot classification helper, and is built
to fit alongside the sibling crates `silero` (VAD), `soundevents` (sound classification), and `mediatime`
(rational time primitives) in the Findit-AI ecosystem.

The library is designed for an audio search pipeline:

```
audio frames → silero (VAD) → speech segments
            → STT (e.g. Whisper) → transcripts per segment
            → textclap audio encoder → 512-dim audio embedding per segment
            → textclap text encoder  → 512-dim text embedding per transcript
            → store both in lancedb (or any Arrow-based vector DB)
            → query: text → text encoder → cosine similarity search
```

CLAP is **not** a streaming model — it embeds fixed-length audio clips (typically 10 s at 48 kHz). This crate
does inference per clip or per VAD-derived segment; long inputs are handled by chunking + aggregation.

## 2. Non-goals

- **Audio resampling.** Input must be 48 kHz mono `f32` PCM. Caller's responsibility, matching silero/soundevents.
- **Streaming inference.** CLAP isn't streaming; we don't pretend it is.
- **Vector store integration.** Embeddings are emitted; storage and ANN search live in the caller.
- **Model bundling.** No models in the crate or downloaded at build time. Caller supplies file paths or bytes.
- **Async / runtime ownership.** Synchronous library, like the sibling crates.
- **Custom mel-scale conventions.** Whatever the HF reference uses, we match.

## 3. Crate layout

```
textclap/
├── Cargo.toml
├── build.rs                       # kept from template — tarpaulin feature detection
├── README.md
├── CHANGELOG.md
├── LICENSE-MIT / LICENSE-APACHE / COPYRIGHT
├── src/
│   ├── lib.rs                     # module decls + public re-exports + crate-level docs
│   ├── error.rs                   # Error enum (thiserror)
│   ├── options.rs                 # Options, ChunkingOptions, Aggregation
│   ├── mel.rs                     # MelExtractor: STFT → mel filterbank → log-mel
│   ├── audio.rs                   # AudioEncoder
│   ├── text.rs                    # TextEncoder
│   └── clap.rs                    # Clap (both encoders) + zero-shot helper
├── tests/
│   ├── clap_integration.rs        # gated on TEXTCLAP_MODELS_DIR env var
│   └── fixtures/
│       ├── sample.wav             # ~200 KB public-domain golden audio
│       ├── golden_mel.npy         # HF reference mel features
│       ├── golden_audio_emb.npy   # HF reference audio embedding
│       ├── golden_text_embs.npy   # HF reference text embeddings for fixed labels
│       └── regen_golden.py        # pinned-version Python that produced the goldens
├── benches/
│   ├── bench_mel.rs
│   ├── bench_audio_encode.rs
│   └── bench_text_encode.rs
├── examples/
│   └── index_and_search.rs        # pipeline shape (lancedb stubbed)
└── docs/superpowers/specs/        # this file lives here
```

## 4. Dependencies

### Default

| Crate         | Version       | Purpose                                          |
|---------------|---------------|--------------------------------------------------|
| `ort`         | `2.0.0-rc.12` | ONNX Runtime Rust bindings (matches siblings)    |
| `rustfft`     | `6`           | Real-input STFT for mel extraction               |
| `tokenizers`  | `0.20`        | HF tokenizer.json loader (RoBERTa BPE)           |
| `thiserror`   | `2`           | Error derives                                    |

### Optional features

- **`serde`** — `Serialize` / `Deserialize` derives on `Options`, `ChunkingOptions`, `Aggregation`,
  `LabeledScore`, and `Embedding` (sequence form).

### Excluded (deliberate)

- No `tokio`, no async — synchronous library.
- No `download` feature — no network, no `ureq`/`sha2`/`reqwest`.
- No model bundling — no `bundled` feature.

## 5. Toolchain & metadata

- **Rust edition:** 2024
- **MSRV:** 1.85
- **License:** MIT OR Apache-2.0
- **Crate-level lints:** `#![deny(missing_docs)]`, `#![forbid(unsafe_code)]`
- **Initial version:** `0.1.0` (matches branch name)

## 6. Public API

All public structs use private fields and accessor methods, matching silero/soundevents/mediatime conventions.
Field-less unit enums (`Aggregation`, `Error` variants) are public-as-data. Builder-style `with_*` methods
return `Self` by value; getters return references or `Copy` values.

### 6.1 Top-level types

```rust
pub struct Clap          { /* AudioEncoder + TextEncoder */ }
pub struct AudioEncoder  { /* ort::Session + MelExtractor + scratch */ }
pub struct TextEncoder   { /* ort::Session + Tokenizer + scratch */ }

pub struct Embedding     { /* invariant: L2-normalized [f32; 512] */ }
pub struct LabeledScore<'a> { /* private */ }

pub struct Options       { /* private */ }
pub struct ChunkingOptions { /* private */ }
pub enum   Aggregation   { Mean, Max }

pub type Result<T, E = Error> = std::result::Result<T, E>;
```

### 6.2 `Clap`

```rust
impl Clap {
    pub fn from_files<P: AsRef<Path>>(
        audio_onnx: P, text_onnx: P, tokenizer_json: P, opts: Options,
    ) -> Result<Self>;

    pub fn from_memory(
        audio_bytes: &[u8], text_bytes: &[u8], tokenizer_bytes: &[u8], opts: Options,
    ) -> Result<Self>;

    pub fn audio_mut(&mut self) -> &mut AudioEncoder;
    pub fn text_mut(&mut self)  -> &mut TextEncoder;

    // Single ~10s clip
    pub fn classify<'a>(&mut self, samples: &[f32], labels: &'a [&str], k: usize)
        -> Result<Vec<LabeledScore<'a>>>;
    pub fn classify_all<'a>(&mut self, samples: &[f32], labels: &'a [&str])
        -> Result<Vec<LabeledScore<'a>>>;

    // Long clip (chunked + aggregated)
    pub fn classify_chunked<'a>(
        &mut self, samples: &[f32], labels: &'a [&str], k: usize, opts: &ChunkingOptions,
    ) -> Result<Vec<LabeledScore<'a>>>;
}
```

`classify` is `classify_all` followed by heap-based top-k (no full sort), matching `soundevents::Classifier::classify`.
Score is **cosine similarity** between L2-normalized audio and text embeddings (range ≈ `[-1, 1]`); higher is more
relevant. Order is descending by score; tie-break is input-label order (stable).

### 6.3 `AudioEncoder`

```rust
impl AudioEncoder {
    pub fn from_file<P: AsRef<Path>>(onnx_path: P, opts: Options) -> Result<Self>;
    pub fn from_memory(onnx_bytes: &[u8], opts: Options) -> Result<Self>;
    pub fn from_ort_session(session: ort::session::Session, opts: Options) -> Result<Self>;

    /// Single clip, length ≤ 480_000 samples (10 s @ 48 kHz). Shorter inputs are
    /// repeat-padded inside the mel extractor; longer inputs return Error::AudioTooLong.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Embedding>;

    /// N equal-length clips. Returns Error::BatchLengthMismatch on uneven input.
    pub fn embed_batch(&mut self, clips: &[&[f32]]) -> Result<Vec<Embedding>>;

    /// Arbitrary-length input. Windows + aggregates per ChunkingOptions.
    pub fn embed_chunked(&mut self, samples: &[f32], opts: &ChunkingOptions) -> Result<Embedding>;
}
```

`&mut self` because mel-spec scratch is mutable. `AudioEncoder` is `Send` but **not `Sync`**.

### 6.4 `TextEncoder`

```rust
impl TextEncoder {
    pub fn from_files<P: AsRef<Path>>(onnx_path: P, tokenizer_json_path: P, opts: Options) -> Result<Self>;
    pub fn from_memory(onnx_bytes: &[u8], tokenizer_json_bytes: &[u8], opts: Options) -> Result<Self>;
    pub fn from_ort_session(
        session: ort::session::Session, tokenizer: tokenizers::Tokenizer, opts: Options,
    ) -> Result<Self>;

    pub fn embed(&mut self, text: &str) -> Result<Embedding>;
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>>;
}
```

Same `&mut self` rationale. Padding is dynamic (pad to longest in batch). Truncation is whatever `tokenizer.json`
configures (typically `max_length = 77` for CLAP). No `max_length` knob is exposed; users wanting overrides
build their own `Tokenizer` and use `from_ort_session`.

### 6.5 `Embedding`

```rust
impl Embedding {
    pub const DIM: usize = 512;
    pub fn dim(&self) -> usize;

    // Borrow-only access — zero-copy ingestion into Arrow / lancedb.
    pub fn as_array(&self) -> &[f32; 512];
    pub fn as_slice(&self) -> &[f32];

    // Owned conversions.
    pub fn to_vec(&self) -> Vec<f32>;
    pub fn into_array(self) -> [f32; 512];

    // Reconstruction from storage (DB round-trip).
    pub fn from_unit_array(arr: [f32; 512]) -> Self;          // debug_assert unit-norm
    pub fn try_from_unit_slice(s: &[f32]) -> Result<Self>;    // checks len; debug_assert unit-norm
    pub fn from_unnormalized(arr: [f32; 512]) -> Self;        // always normalizes (escape hatch)

    // Similarity (== for unit vectors).
    pub fn dot(&self, other: &Embedding) -> f32;
    pub fn cosine(&self, other: &Embedding) -> f32;
}

impl AsRef<[f32]> for Embedding;          // delegates to as_slice()
// derive: Clone, Debug, PartialEq.  No Eq/Hash (f32 isn't totally ordered).
#[cfg(feature = "serde")] // serializes as a sequence of 512 f32 values.
```

**Invariant:** every `Embedding` returned by this crate is L2-normalized to unit length. Internal constructors
divide raw ONNX output by its L2 norm; external constructors either trust (`from_unit_array`,
`try_from_unit_slice`) or normalize (`from_unnormalized`). Bug-resistant by construction; ANN-store-friendly.

### 6.6 `LabeledScore`

```rust
impl<'a> LabeledScore<'a> {
    pub fn label(&self) -> &'a str;
    pub fn score(&self) -> f32;     // cosine similarity in [-1, 1]
}
```

Borrows from the input `labels: &'a [&str]` slice — zero allocation for the label side. Ordered by score
descending in returned vectors.

### 6.7 `Options`

```rust
impl Options {
    pub fn new() -> Self;                                                 // sensible defaults
    pub fn with_graph_optimization_level(self, level: GraphOptimizationLevel) -> Self;
    pub fn graph_optimization_level(&self) -> GraphOptimizationLevel;
    pub fn with_intra_threads(self, n: usize) -> Self;
    pub fn intra_threads(&self) -> usize;
}
```

`GraphOptimizationLevel` is re-exported from `ort`. Threading defaults inherit ort defaults; no opinionated
override.

### 6.8 `ChunkingOptions`

```rust
impl ChunkingOptions {
    pub fn new() -> Self;                                  // window=480_000, hop=480_000, agg=Mean, batch_size=8
    pub fn with_window_samples(self, n: usize) -> Self;
    pub fn window_samples(&self) -> usize;
    pub fn with_hop_samples(self, n: usize) -> Self;
    pub fn hop_samples(&self) -> usize;
    pub fn with_aggregation(self, agg: Aggregation) -> Self;
    pub fn aggregation(&self) -> Aggregation;
    pub fn with_batch_size(self, n: usize) -> Self;
    pub fn batch_size(&self) -> usize;
}
```

Validation runs at use, not at build (matches silero `SpeechOptions`): `embed_chunked` returns
`Error::ChunkingConfig` if any of `window_samples`, `hop_samples`, or `batch_size` is `0`.

## 7. Audio inference pipeline

### 7.1 Mel-spectrogram extractor (`src/mel.rs`)

`MelExtractor` is `pub(crate)` — never appears in the public API. CLAP-fixed parameters baked in:

| Parameter        | Value      |
|------------------|------------|
| Sample rate      | 48 000 Hz  |
| Target samples   | 480 000 (10 s) |
| `n_fft`          | 1024       |
| Hop length       | 480        |
| Window           | Hann, length 1024 |
| Mel bins         | 64         |
| Frequency range  | 50 – 14 000 Hz |
| Padding mode     | repeatpad  |
| Truncation mode  | head (deterministic) |

Pipeline per call:

```
samples (f32, 48 kHz mono, length L)
  → pad-or-truncate to 480_000 samples         (repeatpad if L < target; head-truncate if L > target)
  → STFT (n_fft=1024, hop=480, Hann window)    via rustfft RealFftPlanner → [513 freq bins × 1000 frames]
  → |·|² (power spectrogram)                   → [513 × 1000]
  → mel filterbank multiply                    → [64 × 1000]
  → log10 with eps clamp (max(eps, x))         → [64 × 1000]
  → write into caller-provided [64 × 1000] f32 buffer (row-major, time-major contiguous)
```

State (allocated once in `new()`, reused per call):
- Hann window (`Vec<f32>`, len 1024)
- Mel filterbank (`Vec<f32>`, len 64 × 513)
- `RealFftPlanner` instance
- FFT input/output/scratch buffers

`MelExtractor` is `Send` but not `Sync`. Each `AudioEncoder` owns one.

**Mel-scale convention (Slaney vs HTK), eps value, log base, and any other preprocessing edge cases will be
verified against HF Python reference output during implementation.** The integration test (Section 9) catches
drift; tolerance budget is `max_abs_diff < 1e-4` between Rust and Python mel features.

### 7.2 `AudioEncoder` orchestration

**`embed(samples)`:**
1. Validate `samples.len() ≤ 480_000` (else `AudioTooLong`).
2. Mel extractor writes into `[64 × 1000]` scratch buffer.
3. Wrap scratch as ort tensor `[1, 1, 64, 1000]` f32.
4. `session.run()` → output `[1, 512]`.
5. L2-normalize row 0; wrap as `Embedding`.

**`embed_batch(clips)`:**
1. Verify all clips equal length (else `BatchLengthMismatch`).
2. For each clip, mel extractor writes into the appropriate row of an `[N × 64 × 1000]` scratch buffer.
3. One ort tensor `[N, 1, 64, 1000]`, one `session.run()`.
4. Row-by-row L2-normalize → `Vec<Embedding>`.

**`embed_chunked(samples, opts)`:**
1. Validate `opts.window_samples > 0 && opts.hop_samples > 0 && opts.batch_size > 0`.
2. Compute chunk offsets: `0, hop, 2·hop, ...` while `offset < samples.len()`.
3. Each chunk is `samples[offset .. min(offset + window, len)]`. Trailing short chunk goes through
   the mel extractor's repeat-pad. Single-chunk case (input shorter than window) is handled identically.
4. Process chunks in groups of `opts.batch_size` via `embed_batch`.
5. Aggregate the resulting embeddings:
   - `Mean`: component-wise average → L2-normalize → `Embedding`
   - `Max`: component-wise element-wise max → L2-normalize → `Embedding`
6. Single chunk: skip aggregation (the lone embedding is already unit-norm).

**Allocation budget after warm-up:** O(1) for `embed`, O(N) for batch result. Mel scratch, FFT scratch, and
ONNX input backing live on the encoder.

## 8. Text inference pipeline

### 8.1 Tokenizer

Loaded once at construction via `tokenizers::Tokenizer::from_bytes` / `Tokenizer::from_file`. The tokenizer
itself owns BPE merges, vocab, pre/post processors, special tokens, truncation strategy, and pad token id —
all encoded in `tokenizer.json`. textclap inspects the tokenizer at construction to cache:

- `pad_id: i64` (from `tokenizer.get_padding()` or the default config)
- `max_length: usize` (from the tokenizer's truncation params; typically 77 for CLAP)

If padding isn't already configured in `tokenizer.json`, textclap calls `Tokenizer::with_padding(...)` to
enable batch-longest padding.

### 8.2 `TextEncoder` orchestration

**`embed(text)`:**
1. Reject empty `text` with `Error::EmptyInput`.
2. `tokenizer.encode(text, add_special_tokens=true)` → `Encoding` (input_ids, attention_mask as `Vec<u32>`).
3. Cast to `Vec<i64>` in encoder scratch buffers.
4. ort tensors `input_ids: [1, T] i64`, `attention_mask: [1, T] i64`.
5. `session.run()` → output `[1, 512]`.
6. L2-normalize → `Embedding`.

**`embed_batch(texts)`:**
1. Reject empty slice or any empty string with `Error::EmptyInput`.
2. `tokenizer.encode_batch(texts)` → `Vec<Encoding>` (varying lengths).
3. Determine `T_max = max len` in batch; fill `[N × T_max] i64` ids and mask scratch buffers, padding with `pad_id`.
4. ort tensors `[N, T_max] i64` × 2.
5. `session.run()` → output `[N, 512]`.
6. Row-by-row L2-normalize → `Vec<Embedding>`.

### 8.3 ONNX input/output names

The exact tensor names (`input_ids`, `attention_mask`, output embedding name) match the Xenova export's
ONNX graph. They will be hardcoded in `text.rs` after inspection during implementation; mismatched names
surface as `Error::UnexpectedTensorShape` at the first inference call, not at load time.

## 9. Error type

Single `thiserror` enum, exposed at crate root, `#[non_exhaustive]` for additive evolution.

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("failed to load ONNX model: {0}")]
    OnnxLoad(#[source] ort::Error),

    #[error("failed to load tokenizer: {0}")]
    TokenizerLoad(String),

    #[error("failed to read file {path}: {source}")]
    Io { path: PathBuf, #[source] source: std::io::Error },

    #[error("audio input length {got} exceeds maximum {max} samples (10 s @ 48 kHz)")]
    AudioTooLong { got: usize, max: usize },

    #[error("batch clips must all be the same length; got lengths {first} and {other}")]
    BatchLengthMismatch { first: usize, other: usize },

    #[error("invalid chunking options: {reason}")]
    ChunkingConfig { reason: &'static str },

    #[error("tokenization failed: {0}")]
    Tokenize(String),

    #[error("input text or batch is empty")]
    EmptyInput,

    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch { expected: usize, got: usize },

    #[error("unexpected tensor shape: expected {expected}, got {got}")]
    UnexpectedTensorShape { expected: String, got: String },

    #[error("ONNX runtime error: {0}")]
    Onnx(#[from] ort::Error),
}
```

`tokenizers` errors are stringified — the upstream error type isn't structurally useful to match on.

## 10. Testing strategy

### 10.1 Unit tests (per module)

- **`mel.rs`:** Hann window numerical correctness, mel filterbank center frequencies, repeat-pad behavior,
  output buffer shape, eps clamp behavior on silence input.
- **`audio.rs`:** `AudioTooLong` boundary at exactly 480_000 samples, `BatchLengthMismatch` detection,
  chunked windowing offsets and chunk count for representative input lengths and hop sizes.
- **`text.rs`:** `EmptyInput` for empty `&str` and empty batch, batch padding fills exactly
  `T_max - len(i)` slots with `pad_id`, special tokens prepended/appended.
- **`clap.rs`:** `classify` returns top-k in score-descending order, `classify_all` returns one entry per
  input label, ranking stability with tied scores.
- **`options.rs`:** Builder methods round-trip through accessors, defaults match documented values.
- **`Embedding`:** `from_unnormalized` always produces unit-norm output, `try_from_unit_slice` rejects
  wrong lengths, `dot` equals `cosine` for unit inputs, `to_vec`/`as_slice`/`as_array` are consistent.

### 10.2 Integration test (`tests/clap_integration.rs`)

Gated on `TEXTCLAP_MODELS_DIR` env var (skip with `eprintln!` if unset, do not fail). Models are not committed.

Fixtures (committed):
- `tests/fixtures/sample.wav` — public-domain ~3 s 48 kHz mono WAV (~200 KB)
- `tests/fixtures/golden_mel.npy` — `[64, 1000]` HF reference mel features for that WAV
- `tests/fixtures/golden_audio_emb.npy` — `[512]` HF reference audio embedding (L2-normalized)
- `tests/fixtures/golden_text_embs.npy` — `[5, 512]` HF reference text embeddings for fixed labels:
  `["a dog barking", "speech", "music", "silence", "door creaking"]`
- `tests/fixtures/regen_golden.py` — pinned-version Python that produced the goldens

Assertions:

| Check                                        | Tolerance (`max_abs_diff`) |
|----------------------------------------------|----------------------------|
| Rust mel features vs golden mel              | < 1e-4                     |
| Rust audio embedding vs golden audio emb     | < 1e-3                     |
| Rust text embeddings vs golden text embs     | < 1e-3                     |
| `classify_all`: `"a dog barking"` ranks #1   | exact ordering             |

Audio/text embedding tolerances are looser than mel because INT8-quantized ONNX outputs differ from
full-precision Python reference. If quantization-induced drift exceeds 1e-3 in practice, the tolerance
will be widened with documentation; the ranking assertion is the harder correctness gate.

### 10.3 Doctests

Every public function on `Clap`, `AudioEncoder`, `TextEncoder`, `Embedding` ships a runnable rustdoc
example. `Embedding` examples are `no_run`-free (no model dependency); encoder examples use `# no_run`
to skip execution unless the user has models locally.

### 10.4 Benches (`benches/`)

Three Criterion benchmarks, no correctness assertions:
- `bench_mel.rs` — `MelExtractor::extract_into` on a fixed 10 s buffer.
- `bench_audio_encode.rs` — full encode (mel + ONNX) for batch sizes 1, 4, 8.
- `bench_text_encode.rs` — text encode for batch sizes 1, 8, 32.

### 10.5 CI

Same shape as silero/soundevents:

- **rustfmt** (Linux)
- **clippy** (Linux/macOS/Windows × default features × all features)
- **build + test** matrix (Linux/macOS/Windows × stable Rust)
- **doctest** (Linux, all features, `# no_run` keeps model-dependent examples non-executing)
- **coverage** (tarpaulin → codecov, Linux)
- **integration job** (Linux only) — fetches model files into a runner cache before tests, sets
  `TEXTCLAP_MODELS_DIR`, runs `cargo test --test clap_integration`

Removed from the original template's CI (not applicable to an `ort`-dependent crate):
- WASM, RISC-V, PowerPC64, and other cross-compile targets `ort` doesn't support.
- Miri, ASAN/LSAN/MSAN/TSAN, Loom — we're `#![forbid(unsafe_code)]` and write no custom sync primitives.

## 11. Migration from current template

textclap is currently the bare `al8n/template-rs` scaffold (single "Initial commit", `src/lib.rs` is 11
lines of lint config, version 0.0.0, no deps, comprehensive but largely-irrelevant CI matrix).

### Replace
- `Cargo.toml` (identity, deps, features, MSRV, version 0.1.0).
- `README.md` (purpose, install, quick-start including `Clap::from_files` + audio embed + text embed +
  zero-shot classify, model-acquisition note pointing to HuggingFace, license).
- `src/lib.rs` (keep crate-level lints; replace body with module decls and re-exports).
- `tests/foo.rs` → delete; replaced by per-module unit tests + `tests/clap_integration.rs`.
- `benches/foo.rs` → delete; replaced by the three Criterion benches.
- `examples/foo.rs` → delete; add `examples/index_and_search.rs` showing the pipeline shape (lancedb stubbed).
- `CHANGELOG.md` → reset to Keep-a-Changelog stub starting at `[0.1.0]`.

### Keep
- `build.rs` (tarpaulin feature detection — used by sibling crates' coverage runs).
- License files (`LICENSE-MIT`, `LICENSE-APACHE`, `COPYRIGHT`); update copyright holder/year.
- `.github/workflows/` skeleton, with deletions per Section 10.5.

### Add
- `src/error.rs`, `src/options.rs`, `src/mel.rs`, `src/audio.rs`, `src/text.rs`, `src/clap.rs`.
- `tests/fixtures/` contents per Section 10.2.
- This spec under `docs/superpowers/specs/`.

## 12. Known follow-ups (out of scope for 0.1.0)

- `serde` round-trip tests for `Embedding`/`Options`/`ChunkingOptions` once `serde` feature lands.
- A `silero` ↔ `textclap` integration example showing the full VAD-to-embedding flow.
- Optional `from_ort_session` overload that accepts pre-tuned execution providers
  (CUDA, CoreML) — the current `from_ort_session` already enables this since the caller builds the session.
- Streaming-friendly batch builder for service layers that accumulate variable-length text inputs across
  request boundaries.
- Performance comparison against the Python `transformers` reference on representative corpora.
