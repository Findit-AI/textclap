# textclap — CLAP Inference Library Design

**Status:** Draft (revision 2, post-review)
**Date:** 2026-04-25
**Target version:** 0.1.0

## 1. Purpose

textclap is a Rust inference library for **CLAP** (Contrastive Language-Audio Pre-training). It loads the audio
(HTSAT) and text (RoBERTa) ONNX encoders of LAION's `clap-htsat-unfused` model — typically the
`Xenova/clap-htsat-unfused` export — and exposes them alongside a zero-shot classification helper. It is
designed to fit alongside the sibling crates `silero` (VAD), `soundevents` (sound classification), and
`mediatime` (rational time primitives) in the Findit-AI ecosystem.

The crate is **precision-agnostic at the API level** — `from_files` / `from_memory` take whatever ONNX bytes
the caller supplies. **0.1.0 is verified against the INT8-quantized export specifically**; the fp16 and fp32
exports are expected to work (same I/O contract) but have not been measured. See §11.3 for the
quantization-tolerance posture.

Pipeline:

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
- **Model bundling or download helpers.** No models in the crate, no network at build or runtime. Caller supplies file paths or bytes.
- **Async / runtime ownership.** Synchronous library, like the sibling crates.
- **Multi-variant CLAP support in 0.1.0.** Only the 512-dim `Xenova/clap-htsat-unfused` export is verified (against the INT8 quantization specifically). The public API does not lock to this dimension (see §7.5), so 1024-dim variants like `larger_clap_general` become an additive change later. Other precisions (fp16, fp32) of the same architecture are not blocked by the API but are unverified — see §11.3.

## 3. Pre-implementation prerequisites

The original review of revision 1 identified that several parameters in the audio preprocessing pipeline were
left as "verify during implementation." Those parameters (mel scale, filterbank normalization, log transform,
window endpoint convention, ONNX tensor names) are exactly the ones that, if wrong, cause silent embedding drift
and unending iteration on golden tests. This section lists the work that **must complete before any Rust is
written**, so the spec can be updated with concrete values rather than caveats.

### 3.1 Reference-parameter dump and golden generation

A pinned-version Python script `tests/fixtures/regen_golden.py` that:

1. Loads the test audio fixture (`tests/fixtures/sample.wav`, ~3 s, 48 kHz mono, ≤10 s by design so truncation
   is irrelevant).
2. Constructs `ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")` and writes a sidecar JSON
   `tests/fixtures/golden_params.json` capturing every numerical-fidelity parameter:
   - `sampling_rate`, `feature_size` (mel bins), `fft_window_size`, `hop_length`, `max_length_s`
   - `mel_scale` (expected: `"slaney"`)
   - filterbank normalization (`norm` argument; expected: `"slaney"`)
   - `power_to_db` parameters: `amin` (expected: `1e-10`), `ref` (expected: `1.0`), `top_db` (expected: `None`)
   - Window function: verify whether it's *periodic* (length+1 generated, last sample dropped — librosa/torch
     convention) or *symmetric* (numpy.hanning convention). Expected: periodic.
   - `padding` mode (expected: `"repeatpad"`)
   - `truncation` mode (expected: `"rand_trunc"` — but our fixture is ≤10 s, so this never triggers).
     The Rust crate uses **deterministic head truncation** for inputs longer than 10 s (see §8.1). Because
     the test fixture is ≤10 s, neither codepath truncates, so head-vs-rand_trunc divergence is **not
     exercised in tests**. Deployers feeding >10 s clips through `embed` rather than `embed_chunked` should
     understand they get head truncation, not the HF default.
   - `frequency_min` / `frequency_max` (expected: 50 / 14000 Hz)
3. Runs the extractor on the fixture and saves the resulting `[64, 1000]` float array to
   `tests/fixtures/golden_mel.npy`.
4. Loads `audio_model_quantized.onnx` via `onnxruntime.InferenceSession`, runs it on the golden mel features,
   L2-normalizes the output, saves `[512]` to `tests/fixtures/golden_audio_emb.npy`.
5. Loads `text_model_quantized.onnx` and `tokenizer.json` via `onnxruntime` + the `tokenizers` Python binding,
   runs five fixed labels (`["a dog barking", "speech", "music", "silence", "door creaking"]`),
   L2-normalizes outputs, saves `[5, 512]` to `tests/fixtures/golden_text_embs.npy`.

**Why generate goldens against the int8 ONNX rather than the fp32 PyTorch path:** the Rust crate runs the int8
ONNX. Golden audio and text embeddings must come from the same int8 ONNX run in Python — otherwise the test
tolerance has to absorb both quantization drift *and* implementation differences, and we can't tell them apart.
Mel goldens still come from `ClapFeatureExtractor` (which is fp32 NumPy and is what the int8 ONNX expects as input).

**To verify a non-int8 setup** (fp16 or fp32), users regenerate the goldens by pointing `regen_golden.py` at
the alternate ONNX file. The mel golden does not change (preprocessing is precision-independent); only the
audio and text embedding goldens are recomputed. Tolerances may need to widen — see §11.3.

### 3.2 ONNX graph IO inspection

A second short script `tests/fixtures/inspect_onnx.py` that loads each ONNX file with `onnx.load`, prints
`graph.input` and `graph.output` (name, dtype, shape with dynamic dims marked), and writes the results into
this spec's audio and text orchestration sections (§8.2, §9.2). The exact tensor names and shape conventions
must replace placeholders such as `[N, 1, 64, 1000]`. The probable shapes are:

- Audio model input: `[batch, 1, 64, 1000]` *or* `[batch, 64, 1000]` (no channel dim) — to be confirmed.
- Audio model output: `[batch, 512]`.
- Text model inputs: `input_ids: [batch, T] i64`, `attention_mask: [batch, T] i64`. Verify whether
  `position_ids` or `token_type_ids` are also required (RoBERTa typically inlines position calculation, but
  the export may externalize it).
- Text model output: `[batch, 512]`.

### 3.3 Spec update commit

Once §3.1 and §3.2 produce results, a single commit updates this spec to:

- Replace expected mel parameters in §8.1 with the real values from `golden_params.json`.
- Replace placeholder tensor shapes in §8.2 / §9.2 with real names and shapes.
- Pin the test tolerance table in §12.2 to values calibrated against the actual int8-vs-int8 drift observed.

Only then does Rust implementation begin.

## 4. Crate layout

```
textclap/
├── Cargo.toml
├── build.rs                        # kept from template — tarpaulin feature detection
├── README.md
├── CHANGELOG.md
├── LICENSE-MIT / LICENSE-APACHE / COPYRIGHT
├── src/
│   ├── lib.rs                      # module decls + public re-exports + crate-level docs
│   ├── error.rs                    # Error enum (thiserror)
│   ├── options.rs                  # Options, ChunkingOptions
│   ├── mel.rs                      # MelExtractor: STFT → mel filterbank → log-mel
│   ├── audio.rs                    # AudioEncoder
│   ├── text.rs                     # TextEncoder
│   └── clap.rs                     # Clap (both encoders) + zero-shot helper
├── tests/
│   ├── clap_integration.rs         # gated on TEXTCLAP_MODELS_DIR env var
│   └── fixtures/
│       ├── sample.wav              # ~200 KB public-domain golden audio (≤10 s)
│       ├── golden_params.json      # parameters dumped from ClapFeatureExtractor
│       ├── golden_mel.npy          # HF reference mel features [64, 1000]
│       ├── golden_audio_emb.npy    # int8 ONNX audio embedding [512] (L2-normalized)
│       ├── golden_text_embs.npy    # int8 ONNX text embeddings [5, 512] (L2-normalized)
│       ├── regen_golden.py         # pinned-version Python that produced the above
│       └── inspect_onnx.py         # ONNX graph IO dumper
├── benches/
│   ├── bench_mel.rs
│   ├── bench_audio_encode.rs
│   └── bench_text_encode.rs
├── examples/
│   ├── index_and_search.rs         # end-to-end pipeline shape (lancedb stubbed)
│   └── vad_to_clap.rs              # silero → textclap audio encoder
└── docs/superpowers/specs/         # this file lives here
```

## 5. Dependencies

### Default

| Crate         | Version          | Purpose                                          |
|---------------|------------------|--------------------------------------------------|
| `ort`         | `=2.0.0-rc.12`   | ONNX Runtime Rust bindings (matches siblings, exact pin) |
| `rustfft`     | `^6`             | Real-input STFT for mel extraction               |
| `tokenizers`  | `^0.20`          | HF tokenizer.json loader (RoBERTa BPE)           |
| `thiserror`   | `^2`             | Error derives                                    |

`ort` is pinned to **exactly** `2.0.0-rc.12` (the same release-candidate silero and soundevents pin to). Cross-RC
breakage in `ort` is silent; pinning prevents Cargo from picking up a newer RC and producing a divergent ONNX
session API at compile time. Bumping `ort` is a coordinated change across the four sibling crates.

### Optional features

- **`serde`** — `Serialize` / `Deserialize` derives on `Options`, `ChunkingOptions`,
  `LabeledScore`, `LabeledScoreOwned`, and `Embedding` (sequence form, length 512).

### Excluded (deliberate)

- No `tokio`, no async — synchronous library.
- No `download` feature — no network, no `ureq`/`sha2`/`reqwest`.
- No model bundling — no `bundled` feature.
- No BLAS / `ndarray` — the mel filterbank multiply is small enough to write by hand.

## 6. Toolchain & metadata

- **Rust edition:** 2024
- **MSRV:** 1.85
- **License:** MIT OR Apache-2.0
- **Crate-level lints:** `#![deny(missing_docs)]`, `#![forbid(unsafe_code)]`
- **Initial version:** `0.1.0` (matches the current branch name)

## 7. Public API

All public structs use private fields and accessor methods, matching silero/soundevents/mediatime conventions.
Builder-style `with_*` methods return `Self` by value; getters return references or `Copy` values. Field-less
unit enums (`Error` variants) are public-as-data. **No public signature exposes `[f32; 512]`** — this keeps
the door open for swapping in larger CLAP variants without breaking dependents.

### 7.1 Top-level types

```rust
pub struct Clap          { /* AudioEncoder + TextEncoder */ }
pub struct AudioEncoder  { /* Arc<ort::Session> + Arc<MelExtractor> + encoder-owned scratch */ }
pub struct TextEncoder   { /* Arc<ort::Session> + Tokenizer + cached pad_id + encoder-owned scratch */ }

pub struct Embedding     { /* invariant: L2-normalized, internal storage [f32; 512] */ }

pub struct LabeledScore<'a>      { /* private; borrows label */ }
pub struct LabeledScoreOwned     { /* private; owns label */ }

pub struct Options       { /* private */ }
pub struct ChunkingOptions { /* private */ }

pub type Result<T, E = Error> = std::result::Result<T, E>;
```

`AudioEncoder` and `TextEncoder` hold their `ort::Session` and (for audio) the immutable mel-extractor state
(filterbank, Hann window, FFT planner) behind `Arc`s. This makes `try_clone_with_shared_session` (§7.3, §7.4)
a cheap operation that spawns a sibling encoder for another worker thread without re-loading the 33 MB / 121 MB
model weights. Mutable scratch is never shared.

### 7.2 `Clap`

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

    /// Spawn a sibling Clap that shares both ONNX sessions (Arc) and immutable
    /// mel-extractor state, but allocates fresh scratch. For thread-per-core
    /// service deployment: load once, clone per worker. Memory cost is
    /// O(workers · scratch), not O(workers · model_weights).
    pub fn try_clone_with_shared_session(&self) -> Result<Self>;

    /// Run a dummy forward through both encoders to amortize ORT operator
    /// specialization (typically 5–20× cold-start cost on first run) and
    /// to size the encoder-owned scratch buffers.
    ///
    /// `warmup()` only sizes scratch for batch size 1. Workloads that batch
    /// (e.g. embedding many Whisper transcripts at once) will see one extra
    /// scratch growth on the first batched call.
    pub fn warmup(&mut self) -> Result<()>;

    // Single ~10 s clip
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

`classify` is `classify_all` followed by heap-based top-k (no full sort), matching
`soundevents::Classifier::classify`. Score is **cosine similarity** between L2-normalized audio and text
embeddings (range ≈ `[-1, 1]`); higher is more relevant. Order is descending by score; tie-break is input-label
order (stable).

### 7.3 `AudioEncoder`

```rust
impl AudioEncoder {
    pub fn from_file<P: AsRef<Path>>(onnx_path: P, opts: Options) -> Result<Self>;
    pub fn from_memory(onnx_bytes: &[u8], opts: Options) -> Result<Self>;
    pub fn from_ort_session(session: ort::session::Session, opts: Options) -> Result<Self>;

    /// Single clip, length ≤ 480_000 samples (10 s @ 48 kHz). Shorter inputs are
    /// repeat-padded internally. Empty input returns Error::EmptyAudio. Inputs
    /// longer than the window return Error::AudioTooLong.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Embedding>;

    /// N equal-length clips. Empty *slice* returns Ok(Vec::new()). Any clip of
    /// length 0 returns Error::EmptyAudio. Uneven lengths return
    /// Error::BatchLengthMismatch.
    pub fn embed_batch(&mut self, clips: &[&[f32]]) -> Result<Vec<Embedding>>;

    /// Arbitrary-length input. Windows + aggregates by mean (the only strategy
    /// in 0.1.0). Single chunk shorter than the window is handled identically
    /// (repeat-padded). Empty input returns Error::EmptyAudio.
    pub fn embed_chunked(&mut self, samples: &[f32], opts: &ChunkingOptions) -> Result<Embedding>;

    /// Spawn a sibling AudioEncoder sharing the underlying Arc<Session> and
    /// Arc<MelExtractor>; fresh scratch is allocated. See Clap::try_clone_with_shared_session.
    pub fn try_clone_with_shared_session(&self) -> Result<Self>;

    /// See Clap::warmup. Same caveat: sizes scratch for batch size 1 only.
    pub fn warmup(&mut self) -> Result<()>;
}
```

**Concurrency model:** `AudioEncoder` is `Send` but **not `Sync`**. Each worker thread owns its own
`AudioEncoder`; this matches the thread-per-core architecture this crate is designed for. The encoder owns
its mel feature buffer, FFT scratch, and ONNX input tensor backing as growable `Vec<f32>`s. They are
sized on the first call (or amortized via `warmup()`) and reused for the encoder's lifetime via
`Vec::resize_with` (which preserves capacity) — the hot path performs no heap allocation after warmup.
Service layers needing concurrency construct one encoder per worker via `try_clone_with_shared_session`:
the underlying ONNX session and immutable mel-extractor state are `Arc`-shared, so memory cost scales with
`O(workers · scratch)` rather than `O(workers · model_weights)`.

### 7.3.1 Expected segment length (your VAD pipeline)

silero typically emits speech segments between 0.5 s and 8 s. Each one is shorter than CLAP's 10 s window.
Practical guidance for callers:

| Segment length              | Method to use                                                  |
|-----------------------------|----------------------------------------------------------------|
| ≤ 10 s (480 000 samples)    | `embed(&samples)` — the mel extractor repeat-pads to 10 s      |
| > 10 s, single embedding    | `embed_chunked(&samples, &ChunkingOptions::new())` — windows + Mean |
| Many short segments at once | `embed_batch(&clips)` after **truncating or repeat-padding all clips to a uniform length** outside the crate (CLAP's window ≤ 10 s makes equal-length packing trivial)  |

For the silero+STT+CLAP pipeline this means: take each VAD segment as-is, pass it to `embed`. The repeat-pad is
a CLAP architectural artifact (the model expects exactly 10 s of audio); textclap surfaces the cost honestly
rather than hiding it.

### 7.4 `TextEncoder`

```rust
impl TextEncoder {
    pub fn from_files<P: AsRef<Path>>(onnx_path: P, tokenizer_json_path: P, opts: Options) -> Result<Self>;
    pub fn from_memory(onnx_bytes: &[u8], tokenizer_json_bytes: &[u8], opts: Options) -> Result<Self>;
    pub fn from_ort_session(
        session: ort::session::Session, tokenizer: tokenizers::Tokenizer, opts: Options,
    ) -> Result<Self>;

    /// Empty &str returns Error::EmptyInput. Whitespace-only strings are
    /// accepted as-is (they tokenize to <s> + minimal content + </s>); if the
    /// caller wants to treat them as empty, it's the caller's job to filter.
    pub fn embed(&mut self, text: &str) -> Result<Embedding>;

    /// Empty *slice* returns Ok(Vec::new()). Any empty string in the batch
    /// returns Error::EmptyInput.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>>;

    /// See AudioEncoder::try_clone_with_shared_session.
    pub fn try_clone_with_shared_session(&self) -> Result<Self>;

    pub fn warmup(&mut self) -> Result<()>;
}
```

**Concurrency model:** `TextEncoder` is `Send` but **not `Sync`** (matches `AudioEncoder`; same thread-per-core
rationale). Encoder-owned scratch (`Vec<i64>` for token ids and attention mask) is grown on demand via
`Vec::resize_with` and reused for the encoder's lifetime; the hot path is allocation-free after warmup.
`try_clone_with_shared_session` is the standard way to spawn a sibling encoder for another worker — the ONNX
session is `Arc`-shared, the `Tokenizer` is cheap to `Clone` (its own internal Arc), and only scratch is fresh.

The attention mask is critical — RoBERTa's position calculation is `pad_id + 1 + cumsum(non_pad_mask)`,
which is inlined into the Xenova ONNX export but only produces correct results if the mask is right.

Padding strategy is dynamic (pad each batch to its own longest item) rather than always padding to model max,
because typical Whisper transcripts are far shorter than 77 tokens.

Truncation is whatever `tokenizer.json` configures (typically `max_length = 77` for CLAP). No `max_length` knob
is exposed; users wanting overrides build their own `Tokenizer` and use `from_ort_session`.

### 7.5 `Embedding`

```rust
impl Embedding {
    pub fn dim(&self) -> usize;            // 512 for 0.1.0; runtime-queryable, future-proof

    // Borrow-only access — zero-copy ingestion into Arrow / lancedb.
    pub fn as_slice(&self) -> &[f32];

    // Owned conversion.
    pub fn to_vec(&self) -> Vec<f32>;

    /// Reconstruct from a stored unit vector. Validates length AND norm
    /// (release-mode check: `(norm² − 1).abs() < 1e-3`). The norm check
    /// is ~512 fmadds and prevents non-unit vectors from polluting an
    /// ANN index — a classic, expensive bug.
    pub fn try_from_unit_slice(s: &[f32]) -> Result<Self>;

    /// Construct from any non-zero slice; always re-normalizes to unit length
    /// (idempotent for input that's already unit-norm). Validates length and
    /// rejects all-zero input via Error::EmbeddingNotUnitNorm. Use this when
    /// you have a vector and don't know whether it's normalized.
    pub fn from_slice_normalizing(s: &[f32]) -> Result<Self>;

    // Similarity (== for unit vectors).
    pub fn dot(&self, other: &Embedding) -> f32;
    pub fn cosine(&self, other: &Embedding) -> f32;
}

impl AsRef<[f32]> for Embedding;          // delegates to as_slice()
// derives: Clone, Debug, PartialEq.
// No Eq / Hash. Bit-pattern equality is incompatible with f32's PartialEq
// semantics around +0/-0 and NaN; storing embeddings in a HashMap is not a
// supported pattern (use the ANN index for similarity lookups).
#[cfg(feature = "serde")] // serializes as a sequence of 512 f32 values.
```

**No public method exposes a fixed-size array.** Internal storage is `[f32; 512]` for 0.1.0 (cheap,
stack-friendly); that detail can change to `Box<[f32]>` later to support 1024-dim CLAP variants without
breaking the public API. **There is no `pub const DIM`** — code that needs the dimension calls `dim()` or
`embedding.as_slice().len()`. The lancedb snippet in §13 demonstrates this pattern.

**PartialEq caveat:** `Embedding: PartialEq` compares the full 512-component f32 arrays bitwise after L2
normalization. Two embeddings produced from the same input on different threads (or even the same thread
across separate processes) **may not compare equal** if `intra_threads > 1`, due to ORT reduce-order
non-determinism. Tests that compare embeddings should run with `Options::new().with_intra_threads(1)` (§11.5).

**Invariant:** every `Embedding` returned by this crate is L2-normalized to unit length. Internal constructors
divide raw ONNX output by its L2 norm; `try_from_unit_slice` validates the invariant; `from_slice_normalizing`
re-establishes it.

### 7.6 `LabeledScore` and `LabeledScoreOwned`

```rust
impl<'a> LabeledScore<'a> {
    pub fn label(&self) -> &'a str;
    pub fn score(&self) -> f32;
    pub fn to_owned(&self) -> LabeledScoreOwned;
}

impl LabeledScoreOwned {
    pub fn label(&self) -> &str;
    pub fn score(&self) -> f32;
    pub fn into_label(self) -> String;
}
```

`LabeledScore<'a>` borrows from the input `labels: &'a [&str]` slice — zero allocation for the label side,
ideal for in-thread top-k. `LabeledScoreOwned` is for serialization, cross-thread send, or DB row construction;
`to_owned()` converts.

### 7.7 `Options`

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

### 7.8 `ChunkingOptions`

```rust
impl ChunkingOptions {
    pub fn new() -> Self;                                  // window=480_000, hop=480_000, batch_size=8
    pub fn with_window_samples(self, n: usize) -> Self;
    pub fn window_samples(&self) -> usize;
    pub fn with_hop_samples(self, n: usize) -> Self;
    pub fn hop_samples(&self) -> usize;
    pub fn with_batch_size(self, n: usize) -> Self;
    pub fn batch_size(&self) -> usize;
}
```

Aggregation strategy is **fixed to mean** in 0.1.0 — there is no `Aggregation` enum and no
`with_aggregation` setter. A second strategy (max, attention pooling, mean-of-logits, etc.) would be added
together with the enum and setter in a minor-version bump; until then a single-variant enum carries no
information and clutters the API. See §14.

Validation runs at use, not at build (matches silero `SpeechOptions`): `embed_chunked` returns
`Error::ChunkingConfig` if any of `window_samples`, `hop_samples`, or `batch_size` is `0`.

## 8. Audio inference pipeline

### 8.1 Mel-spectrogram extractor (`src/mel.rs`)

`MelExtractor` is `pub(crate)` — never appears in the public API. CLAP-fixed parameters baked in:

| Parameter            | Value                                             |
|----------------------|---------------------------------------------------|
| Sample rate          | 48 000 Hz                                         |
| Target samples       | 480 000 (10 s)                                    |
| `n_fft`              | 1024                                              |
| Hop length           | 480                                               |
| Window               | **Hann, periodic, length 1024**                   |
| Mel bins             | 64                                                |
| Mel scale            | **Slaney**                                        |
| Filterbank norm      | **Slaney** (per-filter bandwidth normalization)   |
| Frequency range      | 50 – 14 000 Hz                                    |
| Power spectrum       | `|X|²` (squared magnitude, not magnitude)         |
| Mel→dB transform     | **`10 · log10(max(amin, x))` with `amin = 1e-10`, `ref = 1.0`, no `top_db` clip** |
| Padding mode         | repeatpad (tile input to 480 000 samples)         |
| Truncation mode      | head (deterministic, first 480 000 samples)       |

These exact values are **expected**; §3.1 confirms or corrects them by inspecting `ClapFeatureExtractor`
directly. Any divergence between this table and `golden_params.json` triggers a spec-update commit before
implementation.

Pipeline per call:

```
samples (f32, 48 kHz mono, length L)
  → pad-or-truncate to 480_000 samples           (repeatpad if L < target; head-truncate if L > target)
  → STFT (n_fft=1024, hop=480, periodic Hann)    via rustfft RealFftPlanner → [513 freq bins × 1000 frames]
  → |·|² (power spectrogram)                     → [513 × 1000]
  → mel filterbank multiply (Slaney scale, Slaney norm)  → [64 × 1000]
  → 10 · log10(max(amin, x)) with amin=1e-10     → [64 × 1000]
  → write into caller-provided [64 × 1000] f32 buffer (row-major, time-major contiguous)
```

State allocated once in `new()` and held inside `Arc<MelExtractor>` so multiple sibling encoders (created via
`AudioEncoder::try_clone_with_shared_session`) can share these read-only tables without re-computing them:
- Hann window (`Vec<f32>`, len 1024). Generated as the first 1024 samples of a length-1025 symmetric Hann
  (the periodic convention).
- Mel filterbank (`Vec<f32>`, len 64 × 513). Generated using Slaney's MEL-scale formula and Slaney
  normalization (each filter divided by its bandwidth in Hz).
- The real-FFT plan handle from `RealFftPlanner<f32>`.

Per-call (allocated on the call stack or via `Vec::with_capacity`):
- FFT input buffer (1024 f32), output buffer (513 Complex<f32>), scratch (planner-defined).
- Power spectrum frame buffer.
- Mel feature output (caller-provided).

### 8.1.1 Filterbank-correctness unit test

Beyond the integration test in §12.2, `mel.rs` ships a unit test that compares filter row 0 (the lowest-frequency
filter) and filter row 32 (a mid-band filter) against pre-computed reference rows extracted from
`librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64, fmin=50, fmax=14000, htk=False, norm='slaney')` at design
time and committed as small `.npy` arrays. Tolerance: `max_abs_diff < 1e-6`. This catches filterbank-construction
bugs without needing the full integration fixture.

### 8.2 `AudioEncoder` orchestration

> **§3.2 backfill required:** the exact ONNX input tensor name and shape (`[batch, 1, 64, 1000]` *or*
> `[batch, 64, 1000]`) are confirmed by `tests/fixtures/inspect_onnx.py` before implementation. Below assumes
> a 4-D input; the implementation drops the `1` channel dim if the export uses 3-D.

**`embed(samples)`:**
1. Reject empty input: `samples.is_empty()` → `Error::EmptyAudio`. (The mel extractor's repeat-pad logic
   requires `len ≥ 1` to tile; an empty clip has no defined mel representation.)
2. Validate `samples.len() ≤ 480_000` (else `AudioTooLong`).
3. Resize the encoder's mel scratch to `[64 × 1000]` (`Vec::resize_with`; no-op after first call).
4. `MelExtractor::extract_into(samples, &mut self.mel_scratch)`.
5. Build ort tensor `[1, 1, 64, 1000]` (or `[1, 64, 1000]`, per §3.2) f32 backed by scratch.
6. `session.run()` → output `[1, 512]`.
7. L2-normalize row 0 → `Embedding`.

**`embed_batch(clips)`:**
1. Empty slice → `Ok(Vec::new())`.
2. Reject any clip with `len == 0` → `Error::EmptyAudio` (with index of first offender for diagnostics).
3. Verify all clips equal length (else `BatchLengthMismatch { first, other }`).
4. Resize the encoder's mel scratch to `[N × 64 × 1000]` via `Vec::resize_with`. After the first batch of
   size `N`, subsequent batches of size ≤ `N` are allocation-free.
5. For each clip: mel extractor writes into the appropriate row.
6. One ort tensor `[N, 1, 64, 1000]`, one `session.run()`.
7. Row-by-row L2-normalize → `Vec<Embedding>`.

**`embed_chunked(samples, opts)`:**
1. Reject empty input: `samples.is_empty()` → `Error::EmptyAudio`.
2. Validate `opts.window_samples > 0 && opts.hop_samples > 0 && opts.batch_size > 0`.
3. Compute chunk offsets: `0, hop, 2·hop, ...` while `offset < samples.len()`. (Step 1's empty-input check
   prevents the case where `hop > samples.len()` produces zero chunks.)
4. Each chunk is `samples[offset .. min(offset + window, len)]`. Trailing short chunk goes through repeat-pad.
   Single-chunk case (input shorter than window) handled identically.
5. Process chunks in groups of `opts.batch_size` via `embed_batch`.
6. Aggregate the resulting embeddings by **mean** (the only strategy in 0.1.0): component-wise average →
   L2-normalize → `Embedding`.
7. Single chunk: skip aggregation (the lone embedding is already unit-norm).

**Numerical edge case in mean aggregation:** if the chunk embeddings are nearly orthogonal (e.g. radically
different sound content across windows), their component-wise mean has small norm and the L2-normalize step
amplifies floating-point noise. In practice this never occurs for natural audio sharing a single
provenance, but it is theoretically possible. The implementation does not special-case this; the resulting
direction will be numerically unstable but the embedding is still unit-norm and won't break downstream
similarity math.

**Allocation budget per call (after warmup):**
- `embed`: only the output `Embedding`. Mel scratch, FFT scratch, and ONNX input backing live on the encoder.
- `embed_batch(N)`: `Vec<Embedding>` of N entries; mel scratch grows once on the first batch of a new max size,
  reused thereafter.
- `embed_chunked(L, batch=B)`: same — scratch sized to `B` once, reused; the per-call cost is the
  `Vec<Embedding>` of `ceil(L/hop)` chunk embeddings during aggregation.

`warmup()` runs a single `embed` (silent input) which sizes the steady-state scratch and triggers ORT operator
specialization — production paths see allocation-free, fully-specialized inference from request 1.

## 9. Text inference pipeline

### 9.1 Tokenizer

Loaded once at construction via `tokenizers::Tokenizer::from_bytes` / `from_file`. textclap inspects the
tokenizer at construction to cache:
- `pad_id: i64` (from `tokenizer.get_padding()` or the configured pad-token id; RoBERTa pad is typically `1`)
- `max_length: usize` (from the tokenizer's truncation params; typically 77 for CLAP)

If padding isn't already configured in `tokenizer.json`, textclap calls `Tokenizer::with_padding(...)` to
enable batch-longest padding.

### 9.2 `TextEncoder` orchestration

> **§3.2 backfill required:** the exact text-model tensor names and dtypes (`input_ids`, `attention_mask`,
> possibly `position_ids` or `token_type_ids`) are confirmed by `tests/fixtures/inspect_onnx.py` before
> implementation. Below assumes the standard two-input RoBERTa export.

**`embed(text)`:**
1. Reject empty `text` with `Error::EmptyInput`.
2. `tokenizer.encode(text, add_special_tokens=true)` → `Encoding` (input_ids, attention_mask as `Vec<u32>`).
3. Resize encoder-owned `ids: Vec<i64>` and `mask: Vec<i64>` to `T`; cast and copy.
4. ort tensors `input_ids: [1, T] i64`, `attention_mask: [1, T] i64`.
5. `session.run()` → output `[1, 512]`.
6. L2-normalize → `Embedding`.

**`embed_batch(texts)`:**
1. Empty slice → `Ok(Vec::new())`.
2. Reject batch containing any empty string with `Error::EmptyInput`.
3. `tokenizer.encode_batch(texts)` → `Vec<Encoding>` (varying lengths).
4. Determine `T_max = max len`; resize encoder-owned ids and mask scratch to `[N × T_max]` (grows on demand,
   reused when subsequent batches fit).
5. Fill row-by-row, padding shorter rows with `pad_id`.
6. ort tensors `[N, T_max] i64` × 2.
7. `session.run()` → output `[N, 512]`.
8. Row-by-row L2-normalize → `Vec<Embedding>`.

## 10. Error type

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

    /// Audio input has length 0. VAD pipelines occasionally emit zero-length
    /// segments due to frame-quantization edge cases; callers must filter or
    /// handle this. For batch input, `clip_index` identifies the first
    /// offending clip; for single-clip input it is `None`.
    #[error("audio input is empty (clip index: {clip_index:?})")]
    EmptyAudio { clip_index: Option<usize> },

    #[error("batch clips must all be the same length; got lengths {first} and {other}")]
    BatchLengthMismatch { first: usize, other: usize },

    #[error("invalid chunking options: {reason}")]
    ChunkingConfig { reason: &'static str },

    #[error("tokenization failed: {0}")]
    Tokenize(String),

    /// Empty `&str` passed to TextEncoder::embed, or empty string inside a batch
    /// passed to embed_batch. Empty *batch slices* are not an error — they
    /// return an empty Vec.
    #[error("input text is empty")]
    EmptyInput,

    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch { expected: usize, got: usize },

    #[error("embedding norm out of tolerance: |norm² − 1| = {deviation:.3e}")]
    EmbeddingNotUnitNorm { deviation: f32 },

    #[error("unexpected tensor shape: expected {expected}, got {got}")]
    UnexpectedTensorShape { expected: String, got: String },

    #[error("ONNX runtime error: {0}")]
    Onnx(#[from] ort::Error),
}
```

`tokenizers` errors are stringified — the upstream error type isn't structurally useful to match on.

## 11. Engineering robustness

### 11.1 ort version coupling

`ort = "=2.0.0-rc.12"` is exact. silero, soundevents, and textclap must all pin to the same RC. Bumping
requires a coordinated change across the trio; otherwise a downstream user combining them will fail to
compile. The README documents this coupling explicitly.

### 11.2 Model file integrity

The README publishes the SHA256 of each known-good model artifact:
- `audio_model_quantized.onnx` (33 MB)
- `text_model_quantized.onnx` (121 MB)
- `tokenizer.json` (2.0 MB)

If a user supplies a different file, they get a runtime tensor-shape error eventually, but the README warns
ahead of time. This matters because Xenova ships fp32, fp16, and int8 variants under similar URLs — picking
the wrong one is easy.

### 11.3 Quantization variant compatibility

textclap 0.1.0 is verified against the **INT8-quantized** export specifically. The fp32 and fp16 exports are
expected to work (same I/O contract) but their golden-test tolerances would differ:

| Variant | Audio embedding tolerance vs Python int8 reference | Notes              |
|---------|----------------------------------------------------|--------------------|
| int8    | < 1e-3 (verified target)                           | This release       |
| fp16    | likely < 5e-3                                      | Not verified       |
| fp32    | likely < 1e-2                                      | Not verified       |

The README states: "0.1.0 is verified against the int8 quantized export. Other precisions should work but
have not been tested." A quantization-tolerance matrix is a §13 follow-up.

### 11.4 Cold-start latency

First `session.run()` after construction includes ORT operator specialization, which can be 5–20× slower than
steady-state inference. `AudioEncoder::warmup(&mut self)`, `TextEncoder::warmup(&mut self)`, and
`Clap::warmup(&mut self)` run a single dummy forward each — 480 000 samples of silence for audio,
`"hello world"` for text — so production paths see steady-state latency from the first real request. The
dummy forward also sizes the encoder-owned scratch buffers to the steady-state for batch size 1.

**Caveat:** workloads that batch (e.g. embedding many Whisper transcripts at once) will see one extra
scratch growth on the first batched call, since `warmup()` only sizes scratch for `N=1`. A
`warmup_for_batch(audio_n, text_n)` variant is a §14 follow-up if the extra growth turns out to matter in
practice. README quick-start example calls `clap.warmup()?` after construction.

### 11.5 Test determinism

Integration tests construct sessions with `Options::new().with_intra_threads(1)` so that reduce-order
variability across thread schedules doesn't introduce flake. Real users should not set this — the production
default is whatever ort decides.

### 11.6 Model attribution

LAION CLAP weights are CC-BY 4.0; the Xenova ONNX export is Apache-2.0; the HTSAT paper has citation
requirements. A "Model attribution" section in the README points to:
- Original LAION model card and license.
- Xenova export license.
- HTSAT and CLAP paper citations (BibTeX).

## 12. Testing strategy

### 12.1 Unit tests (per module)

- **`mel.rs`:** Hann window numerical correctness (periodic convention), mel filterbank center frequencies,
  filter rows 0 and 32 vs librosa-precomputed reference (tolerance 1e-6), repeat-pad behavior, output buffer
  shape, eps clamp behavior on silence input.
- **`audio.rs`:** `AudioTooLong` boundary at exactly 480_000 samples, `EmptyAudio` rejection on
  `embed(&[])` / `embed_chunked(&[], ..)` / batch-with-empty-clip (with correct `clip_index`),
  `BatchLengthMismatch` detection, empty batch slice returns empty Vec, chunked windowing offsets and chunk
  count for representative input lengths and hop sizes, `try_clone_with_shared_session` produces an encoder
  whose first `embed` allocates new scratch but does not re-load the ONNX session (assert via memory probe
  or by sharing assertion on the inner `Arc`).
- **`text.rs`:** `EmptyInput` for empty `&str` and for a batch containing an empty string, empty batch slice
  returns empty Vec, batch padding fills exactly `T_max - len(i)` slots with `pad_id`, special tokens
  prepended/appended.
- **`clap.rs`:** `classify` returns top-k in score-descending order, `classify_all` returns one entry per
  input label, ranking stability with tied scores, `LabeledScore::to_owned()` round-trip.
- **`options.rs`:** Builder methods round-trip through accessors, defaults match documented values.
- **`Embedding`:** `from_slice_normalizing` always produces unit-norm output and rejects all-zero input,
  `try_from_unit_slice` rejects wrong lengths *and* non-unit-norm input (release-mode check), `dot` equals
  `cosine` for unit inputs, `to_vec` and `as_slice` are byte-equal, no `pub const DIM` exists.

### 12.2 Integration test (`tests/clap_integration.rs`)

Gated on `TEXTCLAP_MODELS_DIR` env var (skip with `eprintln!` if unset, do not fail). Models are not committed.

Sessions constructed with `intra_threads(1)` for determinism (§11.5).

Fixtures (committed):
- `tests/fixtures/sample.wav` — public-domain ~3 s 48 kHz mono WAV (~200 KB), shorter than 10 s so truncation
  doesn't trigger.
- `tests/fixtures/golden_params.json` — parameters dumped from `ClapFeatureExtractor`.
- `tests/fixtures/golden_mel.npy` — `[64, 1000]` HF reference mel features.
- `tests/fixtures/golden_audio_emb.npy` — `[512]` int8 ONNX audio embedding (L2-normalized).
- `tests/fixtures/golden_text_embs.npy` — `[5, 512]` int8 ONNX text embeddings (L2-normalized) for fixed labels:
  `["a dog barking", "speech", "music", "silence", "door creaking"]`.
- `tests/fixtures/regen_golden.py` — pinned-version Python that produced the goldens.
- `tests/fixtures/inspect_onnx.py` — ONNX graph IO dumper.

Assertions:

| Check                                                          | Tolerance (`max_abs_diff`)         |
|----------------------------------------------------------------|-------------------------------------|
| Rust mel features vs `golden_mel.npy`                          | < 1e-4                              |
| Rust audio embedding vs `golden_audio_emb.npy` (int8 vs int8)  | < 1e-5                              |
| Rust text embeddings vs `golden_text_embs.npy` (int8 vs int8)  | < 1e-5                              |
| `classify_all` ranks `"a dog barking"` first for the dog WAV   | exact ordering                      |

Both audio and text goldens are produced by running the **same int8 ONNX** in Python (§3.1) against the
**same mel features** Rust will compute (the mel test pins those down to 1e-4). The remaining sources of
divergence are: (a) ORT reduce-order non-determinism — neutralized by `intra_threads(1)` in tests (§11.5);
(b) float-summation drift across language bindings — sub-ULP for 512-component dot products. The 1e-5
budget reflects what's actually plausible. A wider budget would mask filterbank, log-floor, or normalization
bugs that pass the mel test (which uses a separate tolerance) but corrupt embeddings. If real-run measurement
exceeds 1e-5 systematically, the tolerance is widened with a code comment explaining the source — the change
must not be silent.

### 12.3 Doctests

Every public function on `Clap`, `AudioEncoder`, `TextEncoder`, `Embedding` ships a runnable rustdoc example.
`Embedding` examples are runnable (no model dependency); encoder examples use `# no_run` to skip execution
unless the user has models locally.

### 12.4 Benches (`benches/`)

Three Criterion benchmarks, no correctness assertions:
- `bench_mel.rs` — `MelExtractor::extract_into` on a fixed 10 s buffer.
- `bench_audio_encode.rs` — full encode (mel + ONNX) for batch sizes 1, 4, 8.
- `bench_text_encode.rs` — text encode for batch sizes 1, 8, 32.

### 12.5 CI

Same shape as silero/soundevents:
- **rustfmt** (Linux)
- **clippy** (Linux/macOS/Windows × default features × all features)
- **build + test** matrix (Linux/macOS/Windows × stable Rust)
- **doctest** (Linux, all features, `# no_run` keeps model-dependent examples non-executing)
- **coverage** (tarpaulin → codecov, Linux)
- **integration job** (Linux only) — fetches model files into a runner cache before tests, sets
  `TEXTCLAP_MODELS_DIR`, runs `cargo test --test clap_integration`.

Removed from the original template's CI (not applicable to an `ort`-dependent crate):
- WASM, RISC-V, PowerPC64, and other cross-compile targets `ort` doesn't support.
- Miri, ASAN/LSAN/MSAN/TSAN, Loom — we're `#![forbid(unsafe_code)]` and write no custom sync primitives.

## 13. Migration from current template

textclap is currently the bare `al8n/template-rs` scaffold (single "Initial commit", `src/lib.rs` is 11
lines of lint config, version 0.0.0, no deps, comprehensive but largely-irrelevant CI matrix).

### Replace
- `Cargo.toml` (identity, deps, features, MSRV, version 0.1.0, exact `ort` pin).
- `README.md` (purpose, install, quick-start including `Clap::from_files` + `warmup()` + audio embed + text
  embed + zero-shot classify, model-acquisition note pointing to HuggingFace **with SHA256s**, an explicit
  warning that `tokenizer.json` must come from the same Xenova export — *not* from `laion/clap-htsat-unfused`
  on Hugging Face, which differs subtly and produces token-id mismatches that pass tests on common English
  but break on edge cases — model attribution section, ort-coupling note, license, lancedb integration
  snippet, thread-per-core deployment example using `try_clone_with_shared_session`).
- `src/lib.rs` (keep crate-level lints; replace body with module decls and re-exports).
- `tests/foo.rs` → delete; replaced by per-module unit tests + `tests/clap_integration.rs`.
- `benches/foo.rs` → delete; replaced by the three Criterion benches.
- `examples/foo.rs` → delete; add `examples/index_and_search.rs` and `examples/vad_to_clap.rs`.
- `CHANGELOG.md` → reset to Keep-a-Changelog stub starting at `[0.1.0]`.

### Keep
- `build.rs` (tarpaulin feature detection — used by sibling crates' coverage runs).
- License files (`LICENSE-MIT`, `LICENSE-APACHE`, `COPYRIGHT`); update copyright holder/year.
- `.github/workflows/` skeleton, with deletions per §12.5.

### Add
- `src/error.rs`, `src/options.rs`, `src/mel.rs`, `src/audio.rs`, `src/text.rs`, `src/clap.rs`.
- `tests/fixtures/` contents per §3.1, §3.2, §12.2.
- `examples/index_and_search.rs` — pipeline shape, lancedb stubbed.
- `examples/vad_to_clap.rs` — `silero::VoiceActivityDetector` → `Vec<SpeechSegment>` →
  `textclap::AudioEncoder::embed` (or `embed_chunked` for long segments) → `Embedding`.
- This spec under `docs/superpowers/specs/`.

### lancedb integration snippet (for README)

```rust
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use textclap::Embedding;

// Ingest (clap is owned mutably by this worker thread):
let embedding: Embedding = clap.audio_mut().embed(&pcm_48khz_mono)?;
let dim = embedding.as_slice().len() as i32;          // dimension-agnostic
let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
builder.values().append_slice(embedding.as_slice()); // zero-copy from &[f32]
builder.append(true);

// Query: lancedb takes Vec<f32>
let query: Embedding = clap.text_mut().embed("dog barking near a door")?;
let _ = table.search(query.to_vec()).limit(10).execute().await?;

// Read-back (rare; lancedb usually computes similarity for you):
let raw: Vec<f32> = row.get("embedding")?;
let stored = Embedding::try_from_unit_slice(&raw)?;     // validates len AND norm
let sim = query.cosine(&stored);
```

## 14. Known follow-ups (out of scope for 0.1.0)

- `serde` round-trip tests for `Embedding`, `Options`, `ChunkingOptions` once the feature lands.
- 1024-dim CLAP variants (`larger_clap_general`, `larger_clap_music`, `clap-htsat-fused`). The public API is
  already dimension-agnostic at signature level (§7.5); the work is internal storage, ONNX I/O, and goldens.
- Quantization-tolerance matrix populated for fp16 and fp32 exports (§11.3).
- Optional execution-provider configuration example (CUDA, CoreML) layered on top of `from_ort_session`.
- Streaming-friendly batch builder for service layers that accumulate variable-length text inputs across
  request boundaries.
- Performance comparison against the Python `transformers` reference on representative corpora.
- `warmup_for_batch(audio_n: usize, text_n: usize)` if profiling shows the first batched call's scratch
  growth costs measurable latency (§11.4).
- A second chunking-aggregation strategy (max, attention pooling, mean-of-logits, etc.) if a real CLAP use
  case demonstrates value. Adding it brings back the `Aggregation` enum + `ChunkingOptions::with_aggregation`
  setter as `#[non_exhaustive]` additions in a minor-version bump.
- A doctest on `Embedding::cosine` showing the lancedb round-trip (`try_from_unit_slice` of a stored vector
  → `cosine` against the query embedding).
