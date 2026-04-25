# textclap — CLAP Inference Library Design

**Status:** Draft (revision 5, post-rev-3 review)
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

### 1.1 Pipeline and the role of CLAP within it

```
audio frames → silero (VAD)         [16 kHz internally]   → speech segment boundaries
            → STT (e.g. Whisper)    [whatever rate it wants] → transcripts per segment
            → textclap audio encoder [48 kHz, ≤10 s clip]  → 512-dim audio embedding per segment
            → textclap text encoder  [text in]              → 512-dim text embedding per transcript
            → store both in lancedb (or any Arrow-based vector DB)
            → query: text → text encoder → cosine similarity search
```

**Sample-rate mismatch is the caller's problem.** silero requires 16 kHz mono; CLAP requires 48 kHz mono.
textclap does not resample (see §2). The realistic deployment is: the source audio is in some native rate
(commonly 48 kHz from microphones or 44.1 kHz from media files), the caller resamples to 16 kHz for silero,
silero returns segment boundaries in *time* (e.g. via `mediatime::TimeRange`), the caller slices the *original
48 kHz* audio at those boundaries, and that slice goes into textclap. The `examples/vad_to_clap.rs` example
demonstrates one such flow using `rubato` for resampling.

### 1.2 What CLAP actually does on speech segments

Worth setting expectations explicitly: **CLAP-HTSAT-unfused was trained on AudioSet plus general-audio
captions, not on conversational speech**. Speech segments embedded through the audio encoder cluster tightly
in CLAP-audio space and don't discriminate well *between speech contents* — that's not the model's job. In
the pipeline above, the audio embedding captures **non-speech acoustic context accompanying the speech**
(background sounds, ambience, music, dog barks behind a conversation, traffic noise); discrimination between
*what was said* lives on the **text** branch (Whisper transcript → CLAP text encoder). The two branches are
complementary, not redundant.

## 2. Non-goals

- **Audio resampling.** Input must be 48 kHz mono `f32` PCM. Caller's responsibility, matching silero/soundevents.
  See §1.1 for why this matters in practice.
- **Streaming inference.** CLAP isn't streaming; we don't pretend it is.
- **Vector store integration.** Embeddings are emitted; storage and ANN search live in the caller.
- **Model bundling or download helpers.** No models in the crate, no network at build or runtime.
- **Async / runtime ownership.** Synchronous library, like the sibling crates.
- **Multi-variant CLAP support in 0.1.0.** Only the 512-dim `Xenova/clap-htsat-unfused` export is verified
  (against the INT8 quantization specifically). The public API does not lock to this dimension (see §7.5),
  so 1024-dim variants like `larger_clap_general` become an additive change later.
- **NaN/Inf-tolerance.** Callers must ensure samples are finite f32; non-finite input produces undefined
  embeddings (typically all-NaN, which `try_from_unit_slice` rejects on round-trip).
- **Cross-tool embedding interop for chunked audio.** textclap's `embed_chunked` is a textclap-specific
  convention, not LAION-reference compatible (see §7.3, §8.2). Single-window `embed` (≤10 s) does match the
  LAION reference within the verified tolerance.

## 3. Pre-implementation prerequisites

The original review of revision 1 identified that several parameters in the audio preprocessing pipeline were
left as "verify during implementation." Those parameters are exactly the ones that, if wrong, cause silent
embedding drift and unending iteration on golden tests. Rev-3 review surfaced several more (frame count, HTSAT
input normalization, position_ids externalization, tokenizer truncation max_length). This section lists the
work that **must complete before any Rust is written**, so the spec can be updated with concrete values.

### 3.1 Reference-parameter dump and golden generation

A pinned-version Python script `tests/fixtures/regen_golden.py` that:

1. Loads the test audio fixture (`tests/fixtures/sample.wav`, ≤10 s, 48 kHz mono — see §4 and §12.2 for
   provenance and attribution).
2. Constructs `ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")` and writes a sidecar JSON
   `tests/fixtures/golden_params.json` capturing every numerical-fidelity parameter:
   - `sampling_rate`, `feature_size` (mel bins), `fft_window_size`, `hop_length`, `max_length_s`
   - `mel_scale` (expected: `"slaney"`)
   - filterbank normalization (`norm` argument; expected: `"slaney"`)
   - `power_to_db` parameters: `amin` (expected: `1e-10`), `ref` (expected: `1.0`), `top_db` (expected: `None`)
   - Window function: verify whether it's *periodic* (length+1 generated, last sample dropped — librosa/torch
     convention) or *symmetric* (numpy.hanning convention). Expected: periodic.
   - **Frame count `T`** — the time dimension of the resulting mel features. With `n_fft=1024`, `hop=480`,
     and `target_samples=480_000`: STFT with `center=False` gives `floor((480_000 − 1024)/480)+1 = 998`
     frames; `center=True` gives `1001`. The HF preprocessor_config claims `max_frames=1000`, which neither
     formula yields; HF reaches 1000 via specific framing rules (centering + padding side). The actual `T`
     value, the centering flag, and the pad mode go into `golden_params.json`. The Rust mel extractor matches
     whatever value is recorded — the spec stops asserting "1000" and refers to `T` instead.
   - `padding` mode (expected: `"repeatpad"`)
   - `truncation` mode (expected: `"rand_trunc"`); however our fixture is ≤10 s so this never triggers in
     tests. The Rust crate uses **deterministic head truncation** for inputs longer than 10 s (see §8.1).
     Head-vs-rand_trunc divergence is **not exercised by tests**; deployers feeding >10 s clips through
     `embed` should know this.
   - `frequency_min` / `frequency_max` (expected: 50 / 14000 Hz)
3. Runs the extractor on the fixture and saves the resulting `[64, T]` float array to
   `tests/fixtures/golden_mel.npy`.
4. Loads `audio_model_quantized.onnx` via `onnxruntime.InferenceSession`, runs it on the golden mel features,
   saves the **un-normalized** projection `[512]` to `tests/fixtures/golden_audio_proj.npy`. Then computes
   the L2-normalized embedding **using the exact formula** `x / np.linalg.norm(x).astype(np.float32)` —
   *not* `torch.nn.functional.normalize`, which can differ in summation order — and saves to
   `tests/fixtures/golden_audio_emb.npy`. Saving both lets the Rust integration test verify the raw
   projection (chunking math) and the normalized embedding (single-window path) independently.
5. Loads `text_model_quantized.onnx` and `tokenizer.json`, runs five fixed labels
   (`["a dog barking", "speech", "music", "silence", "door creaking"]`), saves un-normalized projections
   `[5, 512]` to `golden_text_projs.npy` and L2-normalized embeddings to `golden_text_embs.npy` (same
   `np.linalg.norm` formula).

**Why goldens come from the int8 ONNX, not the fp32 PyTorch path:** the Rust crate runs the int8 ONNX.
Goldens must run the same int8 ONNX in Python — otherwise the test tolerance has to absorb both quantization
drift *and* implementation differences, indistinguishably. Mel goldens still come from `ClapFeatureExtractor`
(which is fp32 NumPy and is what the int8 ONNX expects as input).

**To verify a non-int8 setup,** users regenerate the goldens by pointing `regen_golden.py` at the alternate
ONNX file. Mel goldens don't change (preprocessing is precision-independent); audio and text projection /
embedding goldens are recomputed. Tolerances may need to widen — see §11.3.

### 3.2 ONNX graph IO inspection

A short script `tests/fixtures/inspect_onnx.py` that loads each ONNX file with `onnx.load`, prints
`graph.input` / `graph.output` (name, dtype, shape with dynamic dims marked) and the first/last 20 graph
nodes, and writes the results into a `tests/fixtures/golden_onnx_io.json` sidecar. Specifically the script
must answer:

- **Audio input shape:** `[batch, 1, 64, T]` *vs* `[batch, 64, T]` (channel dim present?)
- **Audio input normalization:** does the graph contain `Sub` / `Div` / `BatchNorm` initializers near the
  input that perform AudioSet mean/std normalization? If so, the Rust mel extractor must pass raw log-mels;
  if not, Rust may need to bake the normalization into the mel post-processing. **This is the critical
  question to answer.** Silent drift here is the worst kind.
- **Audio output:** `[batch, 512]` (verify name).
- **Text input names and dtypes:** `input_ids: [batch, T] i64`, `attention_mask: [batch, T] i64`, **plus**
  whether `position_ids` is present as a *third* input (some Xenova RoBERTa exports externalize it instead
  of inlining the calculation). Whatever the graph says, the Rust `TextEncoder` matches.
- **Text truncation max_length** — read from `tokenizer.json` separately (via the tokenizers Python binding's
  `tokenizer.truncation`), not assumed to be 77. Some Xenova exports keep RoBERTa's default 512.
- **Text output:** `[batch, 512]` (verify name).

Script output goes into `golden_onnx_io.json`; spec §8.2 / §9.2 / §7.4 reference that file rather than
hardcoding shapes/names.

### 3.3 Model SHA256 acquisition

Before §3.1 / §3.2, the maintainer downloads `audio_model_quantized.onnx`, `text_model_quantized.onnx`, and
`tokenizer.json` from a pinned Hugging Face revision (recorded in `tests/fixtures/MODELS.md`) and computes:

```
shasum -a 256 audio_model_quantized.onnx text_model_quantized.onnx tokenizer.json
```

The SHA256s, the HF revision hash, and the URL are recorded in `tests/fixtures/MODELS.md` and reproduced in
the README. Re-running `regen_golden.py` against a different revision recomputes the goldens; mismatched
SHA256s in user setups produce undefined results (loud warning in README).

### 3.4 Spec update commit

Once §3.1 / §3.2 / §3.3 produce results, a single commit updates this spec to:

- Replace expected mel parameters in §8.1 with the values from `golden_params.json` (except the truncation
  mode, which intentionally diverges — Rust is head-deterministic, HF is rand_trunc).
- Replace placeholder tensor shapes / names in §8.2 / §9.2 with values from `golden_onnx_io.json`.
- Confirm or correct §7.4's attention-mask / position-ids description.
- Pin the test-tolerance table in §12.2 to values calibrated against the actual int8-vs-int8 drift observed.

Only then does Rust implementation begin.

## 4. Crate layout

```
textclap/
├── Cargo.toml
├── build.rs                        # tarpaulin feature detection (defines `cfg(tarpaulin_include)`
│                                   # so coverage runs can selectively skip blocks)
├── README.md
├── CHANGELOG.md
├── LICENSE-MIT / LICENSE-APACHE / COPYRIGHT
├── src/
│   ├── lib.rs                      # module decls + public re-exports + crate-level docs
│   ├── error.rs                    # Error enum (thiserror)
│   ├── options.rs                  # Options, ChunkingOptions, ChunkingField
│   ├── mel.rs                      # MelExtractor: STFT → mel filterbank → log-mel
│   ├── audio.rs                    # AudioEncoder
│   ├── text.rs                     # TextEncoder
│   └── clap.rs                     # Clap (both encoders) + zero-shot helper
├── tests/
│   ├── clap_integration.rs         # gated on TEXTCLAP_MODELS_DIR env var
│   └── fixtures/
│       ├── README.md               # provenance + license attribution for sample.wav
│       ├── MODELS.md               # SHA256s + HF revision + download URLs
│       ├── sample.wav              # public-domain dog-bark WAV, ≤10 s, 48 kHz mono
│       │                           # (size depends on duration: ~3 s i16 ≈ 290 KB)
│       ├── golden_params.json      # parameters dumped from ClapFeatureExtractor (§3.1)
│       ├── golden_onnx_io.json     # ONNX graph IO inspection (§3.2)
│       ├── golden_mel.npy          # HF reference mel features [64, T]
│       ├── golden_audio_proj.npy   # int8 ONNX audio projection [512] (un-normalized)
│       ├── golden_audio_emb.npy    # int8 ONNX audio embedding [512] (L2-normalized)
│       ├── golden_text_projs.npy   # int8 ONNX text projections [5, 512] (un-normalized)
│       ├── golden_text_embs.npy    # int8 ONNX text embeddings [5, 512] (L2-normalized)
│       ├── filterbank_row_0.npy    # librosa-precomputed mel filterbank row 0
│       ├── filterbank_row_32.npy   # librosa-precomputed mel filterbank row 32
│       ├── regen_golden.py         # pinned-version Python that produced the .npy goldens
│       └── inspect_onnx.py         # ONNX graph IO dumper
├── benches/
│   ├── bench_mel.rs
│   ├── bench_audio_encode.rs
│   └── bench_text_encode.rs
├── examples/
│   ├── index_and_search.rs         # end-to-end pipeline shape (lancedb stubbed)
│   └── vad_to_clap.rs              # silero (16 kHz) + rubato resample → textclap (48 kHz)
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

### Optional features

- **`serde`** — `Serialize` / `Deserialize` derives on `Options`, `ChunkingOptions`,
  `LabeledScore`, `LabeledScoreOwned`, and `Embedding` (sequence form, length 512).

### Dev-dependencies (for tests, benches, examples)

| Crate        | Version | Used by                              |
|--------------|---------|--------------------------------------|
| `criterion`  | `^0.5`  | `benches/`                            |
| `silero`     | path    | `examples/vad_to_clap.rs`            |
| `rubato`     | `^0.16` | `examples/vad_to_clap.rs` (resampling) |
| `npyz`       | `^0.8`  | `tests/clap_integration.rs` (.npy reader) |
| `hound`      | `^3`    | `tests/clap_integration.rs` (WAV reader) |

`silero` is referenced via path dep against the sibling crate — this couples the example to the workspace
layout but keeps it real rather than mocked.

### Excluded (deliberate)

- No `tokio`, no async — synchronous library.
- No `download` feature — no network, no `ureq`/`sha2`/`reqwest`.
- No model bundling — no `bundled` feature.
- No BLAS / `ndarray` — the mel filterbank multiply is small enough to write by hand.
- No `tracing` for 0.1.0 — observability is a §14 follow-up.

## 6. Toolchain & metadata

- **Rust edition:** 2024
- **MSRV:** 1.85
- **License:** MIT OR Apache-2.0
- **Crate-level lints:** `#![deny(missing_docs)]`, `#![forbid(unsafe_code)]`
- **Initial version:** `0.1.0` (matches the current branch name)

## 7. Public API

All public structs use private fields and accessor methods, matching silero/soundevents/mediatime conventions.
Builder-style `with_*` methods return `Self` by value; getters return references or `Copy` values. Field-less
unit enums (`Error` variants, `ChunkingField`) are public-as-data. **No public signature exposes `[f32; 512]`** —
this keeps the door open for swapping in larger CLAP variants without breaking dependents.

### 7.1 Top-level types

```rust
pub struct Clap          { /* AudioEncoder + TextEncoder */ }
pub struct AudioEncoder  { /* ort::Session + MelExtractor + encoder-owned scratch */ }
pub struct TextEncoder   { /* ort::Session + Tokenizer + cached pad_id + encoder-owned scratch */ }

pub struct Embedding     { /* invariant: L2-normalized, internal storage [f32; 512] */ }

pub struct LabeledScore<'a>      { /* private; borrows label */ }
pub struct LabeledScoreOwned     { /* private; owns label */ }

pub struct Options       { /* private */ }
pub struct ChunkingOptions { /* private */ }

#[non_exhaustive]
pub enum ChunkingField { Window, Hop, BatchSize }   // identifies which field violated ChunkingConfig

pub type Result<T, E = Error> = std::result::Result<T, E>;
```

`AudioEncoder` and `TextEncoder` own their `ort::Session` and (for audio) the mel-extractor state by value.
There is no internal `Arc`, no clone-with-shared-session, and no cross-thread session sharing. **The
deployment model is thread-per-core**: each worker thread loads its own encoder once at startup. Memory cost
is **150–300 MB resident per worker** for both encoders together (33 MB int8 audio model + 121 MB int8 text
model on disk, plus ORT working buffers and weight layout — measure on your hardware). This is the deliberate
trade for predictable latency and zero synchronization overhead in the inference path.

### 7.2 `Clap`

```rust
impl Clap {
    pub fn from_files<P: AsRef<Path>>(
        audio_onnx: P, text_onnx: P, tokenizer_json: P, opts: Options,
    ) -> Result<Self>;

    /// Bytes are copied into the ONNX sessions and tokenizer; the slices may
    /// be dropped after this call returns.
    pub fn from_memory(
        audio_bytes: &[u8], text_bytes: &[u8], tokenizer_bytes: &[u8], opts: Options,
    ) -> Result<Self>;

    pub fn audio_mut(&mut self) -> &mut AudioEncoder;
    pub fn text_mut(&mut self)  -> &mut TextEncoder;

    /// Run a dummy forward through both encoders to amortize ORT operator
    /// specialization (typically 5–20× cold-start cost on first run) and
    /// to size the encoder-owned scratch buffers for batch size 1.
    /// Workloads that batch will see one extra scratch growth on the first
    /// batched call.
    pub fn warmup(&mut self) -> Result<()>;

    // Single ~10 s clip (LAION-reference compatible)
    pub fn classify<'a>(&mut self, samples: &[f32], labels: &'a [&str], k: usize)
        -> Result<Vec<LabeledScore<'a>>>;
    pub fn classify_all<'a>(&mut self, samples: &[f32], labels: &'a [&str])
        -> Result<Vec<LabeledScore<'a>>>;

    /// Long clip via textclap-specific chunking (NOT LAION-reference
    /// compatible — see §7.3 embed_chunked docs and §8.2).
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
    pub fn from_memory(onnx_bytes: &[u8], opts: Options) -> Result<Self>;  // bytes copied
    pub fn from_ort_session(session: ort::session::Session, opts: Options) -> Result<Self>;

    /// Single clip, length 1..=480_000 samples (10 s @ 48 kHz).
    /// - Empty input (len==0) returns Error::EmptyAudio.
    /// - len > 480_000 returns Error::AudioTooLong (use embed_chunked instead).
    /// - 0 < len < 480_000 is repeat-padded to 10 s by the mel extractor.
    /// - Caller must ensure samples are finite f32; non-finite values produce
    ///   undefined embeddings.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Embedding>;

    /// Batch of clips of *any* lengths in 1..=480_000. Each clip is
    /// repeat-padded to 10 s independently. Empty *slice* returns Ok(Vec::new()).
    /// Any clip with len==0 returns Error::EmptyAudio with its index. There is
    /// no equal-length requirement — the auto-padding makes uneven lengths
    /// harmless. Long-clip handling is still embed_chunked.
    pub fn embed_batch(&mut self, clips: &[&[f32]]) -> Result<Vec<Embedding>>;

    /// Arbitrary-length input via textclap's chunking convention.
    ///
    /// **WARNING — not LAION-reference compatible.** LAION's reference
    /// implementation for the unfused model uses single-window rand_trunc,
    /// not multi-window aggregation. Aggregation belongs to the *fused*
    /// CLAP variant, which does fusion inside the network. textclap's
    /// embed_chunked computes the *centroid of un-normalized chunk
    /// projections* and L2-normalizes it. Embeddings produced this way
    /// occupy a different region of the 512-dim space than LAION-reference
    /// embeddings. Cross-tool retrieval requires both sides to use textclap.
    ///
    /// Empty input returns Error::EmptyAudio.
    pub fn embed_chunked(&mut self, samples: &[f32], opts: &ChunkingOptions)
        -> Result<Embedding>;

    pub fn warmup(&mut self) -> Result<()>;
}
```

**Concurrency model:** `AudioEncoder` is `Send` but **not `Sync`**. Each worker thread owns its own
`AudioEncoder` — thread-per-core, no sharing. The encoder owns its mel feature buffer, FFT scratch, and ONNX
input tensor backing as growable `Vec<f32>`s. They are sized on the first call (or amortized via `warmup()`)
and reused via `Vec::resize_with` (which preserves capacity) — the hot path performs no heap allocation after
warmup. Service layers spawn N workers, each calling `from_files` once at startup; the resulting N independent
ONNX sessions duplicate the model weights, which is the intended trade for synchronization-free inference.

### 7.4 `TextEncoder`

```rust
impl TextEncoder {
    pub fn from_files<P: AsRef<Path>>(onnx_path: P, tokenizer_json_path: P, opts: Options) -> Result<Self>;
    pub fn from_memory(onnx_bytes: &[u8], tokenizer_json_bytes: &[u8], opts: Options) -> Result<Self>;
    pub fn from_ort_session(
        session: ort::session::Session, tokenizer: tokenizers::Tokenizer, opts: Options,
    ) -> Result<Self>;

    /// Empty &str returns Error::EmptyInput. Whitespace-only strings are
    /// accepted as-is — they tokenize to <s> + minimal content + </s>; if
    /// the caller wants to treat them as empty, it's the caller's job to
    /// filter.
    pub fn embed(&mut self, text: &str) -> Result<Embedding>;

    /// Empty *slice* returns Ok(Vec::new()). Any empty string in the batch
    /// returns Error::EmptyInput (with index for diagnostics).
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>>;

    pub fn warmup(&mut self) -> Result<()>;
}
```

**Concurrency model:** `TextEncoder` is `Send` but **not `Sync`** (matches `AudioEncoder`; same thread-per-core
rationale, same independent-per-worker construction story). Encoder-owned scratch (`Vec<i64>` for token ids
and attention mask) is grown via `Vec::resize_with` and reused; the hot path is allocation-free after warmup.

**Tokenizer truncation max_length is taken from `tokenizer.json` at construction** — no `max_length` knob is
exposed. Whether that value is 77 (CLAP's typical setting) or 512 (RoBERTa's default; some Xenova exports
keep this) depends entirely on the supplied `tokenizer.json`, and is recorded into `golden_params.json` by
§3.1. Users wanting overrides build their own `Tokenizer` (with the exact padding and truncation they want)
and use `from_ort_session`.

**Position-ids:** RoBERTa computes positions as `pad_id + 1 + cumsum(non_pad_mask)`. This is *typically*
inlined into Xenova's ONNX export, in which case the textencoder feeds only `input_ids` and `attention_mask`.
But some Xenova exports externalize `position_ids` as a third input. §3.2 confirms which by inspecting the
ONNX graph; the implementation matches whatever the graph says. The attention mask is always critical
regardless — if positions are inlined, the mask drives the cumsum; if externalized, the mask still drives
attention.

### 7.5 `Embedding`

```rust
impl Embedding {
    pub fn dim(&self) -> usize;            // 512 for 0.1.0; runtime-queryable, future-proof

    // Borrow-only access — supports append_slice into Arrow's MutableBuffer
    // without an intermediate Vec.
    pub fn as_slice(&self) -> &[f32];

    // Owned conversion.
    pub fn to_vec(&self) -> Vec<f32>;

    /// Reconstruct from a stored unit vector. Validates length AND norm
    /// (release-mode check: `(norm² − 1).abs() < 1e-4`, matching the
    /// integration-test budget for produced embeddings; this allows for
    /// fp16 storage round-trip in lancedb without false positives).
    pub fn try_from_unit_slice(s: &[f32]) -> Result<Self>;

    /// Construct from any non-zero slice; always re-normalizes to unit length
    /// (idempotent for input that's already unit-norm). Validates length and
    /// rejects all-zero input via Error::EmbeddingZero.
    pub fn from_slice_normalizing(s: &[f32]) -> Result<Self>;

    // Similarity (== for unit vectors).
    pub fn dot(&self, other: &Embedding) -> f32;
    pub fn cosine(&self, other: &Embedding) -> f32;
}

impl AsRef<[f32]> for Embedding;          // delegates to as_slice()

// Custom Debug — does NOT dump 512 floats. Format:
//   Embedding { dim: 512, head: [0.0123, -0.0456, 0.0789, ..] }
impl fmt::Debug for Embedding;

// derives: Clone, PartialEq.
// No Eq / Hash. Bit-pattern equality is incompatible with f32's PartialEq
// semantics around +0/-0 and NaN; storing embeddings in a HashMap is not a
// supported pattern (use the ANN index for similarity lookups).
#[cfg(feature = "serde")] // serializes as a sequence of 512 f32 values.
```

**No public method exposes a fixed-size array.** Internal storage is `[f32; 512]` for 0.1.0 (cheap,
stack-friendly); that detail can change to `Box<[f32]>` later to support 1024-dim CLAP variants without
breaking the public API. **There is no `pub const DIM`** — code that needs the dimension calls `dim()` or
`embedding.as_slice().len()`.

**PartialEq caveat:** `Embedding: PartialEq` compares the full 512-component f32 arrays bit-exact after L2
normalization. Two embeddings produced from the same input may not compare equal across runs, threads, OSes,
or hardware due to **floating-point non-determinism** — reduce order, ORT kernel choice, BLAS path differences
(MLAS on Linux/Windows vs Accelerate on macOS), x86-vs-ARM FMA fusion, AVX-512 vs AVX2 ULP-level differences.
Tests that compare embeddings should run with `Options::new().with_intra_threads(1)` (§11.5) and accept that
tolerances absorb hardware variance.

**Invariant:** every `Embedding` returned by this crate is L2-normalized to unit length. Internal constructors
divide raw ONNX projection by its L2 norm; `try_from_unit_slice` validates the invariant; `from_slice_normalizing`
re-establishes it. The un-normalized projections used internally by `embed_chunked`'s centroid math never
escape to the public API.

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
ideal for in-thread top-k. `LabeledScoreOwned` is for serialization, cross-thread send, or DB row construction.

### 7.7 `Options`

```rust
#[derive(Default)]
impl Options {
    pub fn new() -> Self;                                                 // delegates to Default
    pub fn with_graph_optimization_level(self, level: GraphOptimizationLevel) -> Self;
    pub fn graph_optimization_level(&self) -> GraphOptimizationLevel;

    /// 0 = inherit ORT's default thread count (typically num_cpus).
    /// 1 = single-threaded; deterministic across runs on the same hardware.
    /// >1 = multi-threaded with reduce-order non-determinism (§7.5 caveat).
    pub fn with_intra_threads(self, n: usize) -> Self;
    pub fn intra_threads(&self) -> usize;
}
```

`GraphOptimizationLevel` is re-exported from `ort`.

### 7.8 `ChunkingOptions`

```rust
#[derive(Default)]
impl ChunkingOptions {
    pub fn new() -> Self;        // window=480_000, hop=480_000, batch_size=8
    pub fn with_window_samples(self, n: usize) -> Self;
    pub fn window_samples(&self) -> usize;
    pub fn with_hop_samples(self, n: usize) -> Self;
    pub fn hop_samples(&self) -> usize;
    pub fn with_batch_size(self, n: usize) -> Self;
    pub fn batch_size(&self) -> usize;
}
```

Aggregation strategy is **fixed to centroid-of-un-normalized-projections + L2 normalize** in 0.1.0 — there is
no `Aggregation` enum and no `with_aggregation` setter. A second strategy (max, attention pooling, etc.) would
be added together with the enum and setter in a minor-version bump; until then a single-variant enum carries
no information and clutters the API. See §14.

Validation runs at use, not at build (matches silero `SpeechOptions`): `embed_chunked` returns
`Error::ChunkingConfig { field }` if any of `window_samples`, `hop_samples`, or `batch_size` is `0`, with
`ChunkingField` identifying which.

## 8. Audio inference pipeline

### 8.1 Mel-spectrogram extractor (`src/mel.rs`)

`MelExtractor` is `pub(crate)` — never appears in the public API. Parameters baked in (subject to §3.1
verification — values in this table are *expected*, the recorded values in `golden_params.json` are
authoritative; only the truncation row is intentionally chosen differently):

| Parameter            | Value                                             |
|----------------------|---------------------------------------------------|
| Sample rate          | 48 000 Hz                                         |
| Target samples       | 480 000 (10 s)                                    |
| `n_fft`              | 1024                                              |
| Hop length           | 480                                               |
| Window               | **Hann, periodic, length 1024**                   |
| Frame count `T`      | **TBD by §3.1** (HF claims max_frames=1000; STFT centering convention determines actual) |
| Mel bins             | 64                                                |
| Mel scale            | **Slaney**                                        |
| Filterbank norm      | **Slaney** (per-filter bandwidth normalization)   |
| Frequency range      | 50 – 14 000 Hz                                    |
| Power spectrum       | `|X|²` (squared magnitude, not magnitude)         |
| Mel→dB transform     | **`10 · log10(max(amin, x))` with `amin = 1e-10`, `ref = 1.0`, no `top_db` clip; applied exactly once after the mel filterbank, never before** |
| Padding mode         | repeatpad (tile input to 480 000 samples)         |
| Truncation mode      | head (deterministic, first 480 000 samples) — *intentionally* differs from HF's rand_trunc; see §3.1 |
| HTSAT input norm     | **TBD by §3.2** (whether the ONNX graph contains AudioSet mean/std subtraction near the input — if not, mel.rs may need to apply it post-log-mel) |

Pipeline per call:

```
samples (f32, 48 kHz mono, length L, finite)
  → pad-or-truncate to 480_000 samples           (repeatpad if L < target; head-truncate if L > target)
  → STFT (n_fft=1024, hop=480, periodic Hann)    via rustfft RealFftPlanner → [513 freq bins × T frames]
  → |·|² (power spectrogram)                     → [513 × T]
  → mel filterbank multiply (Slaney scale, Slaney norm)  → [64 × T]
  → 10 · log10(max(amin, x)) with amin=1e-10     → [64 × T]
  → (if §3.2 requires) HTSAT AudioSet normalization (mean/std) → [64 × T]
  → write into caller-provided [64 × T] f32 buffer (row-major, time-major contiguous)
```

State allocated once in `new()`, owned by the `MelExtractor`, owned by the `AudioEncoder`:
- Hann window (`Vec<f32>`, len 1024). Generated as the first 1024 samples of a length-1025 symmetric Hann
  (the periodic convention).
- Mel filterbank (`Vec<f32>`, len 64 × 513). Generated using Slaney's MEL-scale formula and Slaney
  normalization.
- The real-FFT plan handle from `RealFftPlanner<f32>`.

#### 8.1.1 Filterbank-correctness unit test

`mel.rs` ships a unit test that compares filter row 0 (the lowest-frequency filter) and filter row 32 (a
mid-band filter) against pre-computed reference rows extracted from
`librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64, fmin=50, fmax=14000, htk=False, norm='slaney')` at
design time and committed as `tests/fixtures/filterbank_row_0.npy` and `filterbank_row_32.npy`. Tolerance:
`max_abs_diff < 1e-6`. This catches filterbank-construction bugs without needing the full ONNX stack.

#### 8.1.2 Power-to-dB single-application test

Separate unit test asserts: feeding a known input through `MelExtractor::extract_into` and a hand-written
"apply power_to_dB twice" reference produces visibly different output. Confirms the floor is applied exactly
once after the filterbank, never before, never twice.

### 8.2 `AudioEncoder` orchestration

> **§3.2 backfill required:** the audio-input tensor name and shape (`[batch, 1, 64, T]` *or*
> `[batch, 64, T]`) and HTSAT normalization presence are confirmed by `inspect_onnx.py` before
> implementation. Below assumes a 4-D input with the channel dim; the implementation drops the `1` channel
> dim if the export uses 3-D.

**Internal helper `embed_raw(samples) -> Result<[f32; 512]>`** runs the full forward pipeline (mel → ONNX)
and returns the **un-normalized** projection. `embed`, `embed_batch`, and `embed_chunked` build on top of it.
This helper is `pub(crate)` only.

**`embed(samples)`:**
1. `samples.is_empty()` → `Error::EmptyAudio { clip_index: None }`.
2. `samples.len() > 480_000` → `Error::AudioTooLong { got, max: 480_000 }`.
3. Resize encoder mel scratch to `[64 × T]`; mel extractor writes there.
4. Run ONNX, take row 0 of the projection output.
5. L2-normalize → `Embedding`.

**`embed_batch(clips)`:**
1. Empty slice → `Ok(Vec::new())`.
2. For each clip `i`: `clips[i].is_empty()` → `Error::EmptyAudio { clip_index: Some(i) }`;
   `clips[i].len() > 480_000` → `Error::AudioTooLong`. (No equal-length requirement.)
3. Resize encoder mel scratch to `[N × 64 × T]` (grows on first call, reused thereafter via `resize_with`).
4. For each clip: mel extractor independently repeat-pads to 480_000 samples and writes its mel features
   into row `i` of the scratch. After this step every row of the batch tensor has the same `[64 × T]`
   shape regardless of input length.
5. One ONNX call with `[N, 1, 64, T]`.
6. Row-by-row L2-normalize → `Vec<Embedding>`.

**`embed_chunked(samples, opts)`:**
1. `samples.is_empty()` → `Error::EmptyAudio { clip_index: None }`.
2. Validate `window_samples > 0 && hop_samples > 0 && batch_size > 0` → otherwise
   `Error::ChunkingConfig { field }` with `ChunkingField::Window`/`Hop`/`BatchSize`.
3. Compute chunk offsets: `0, hop, 2·hop, …` while `offset < samples.len()`. (Step 1 prevents zero-chunk
   cases.)
4. Each chunk is `samples[offset .. min(offset + window, len)]`. Trailing short chunk goes through
   repeat-pad. Single-chunk case (input shorter than window) handled identically.
5. **Collect un-normalized projections** by running `embed_raw` on each chunk (in groups of `batch_size`
   via the same internal batching path that `embed_batch` uses, but skipping the final L2 normalize).
6. Aggregate: **component-wise mean of the un-normalized projection vectors** (centroid), then L2-normalize
   the centroid → `Embedding`. Single chunk: skip aggregation (the lone projection is normalized directly).

Why centroid-before-normalize and not mean-of-unit-vectors-then-normalize: averaging *un-normalized*
projections yields the linear average of the model's pre-projection direction-and-magnitude information.
Re-normalizing once at the end lands the result on the unit sphere as required by the `Embedding` invariant.
Mean-of-unit-vectors is the spherical mean — geometrically defensible but discards magnitude information that
encodes the model's confidence per chunk. **This is textclap's chunking convention. It is not LAION-reference
compatible.** LAION's reference for the unfused model uses single-window rand_trunc; multi-window aggregation
is a textclap addition (and the analogous aggregation in the *fused* model happens before projection, so
even fused-model embeddings are not directly interchangeable). Cross-tool retrieval requires both indexing
and querying through textclap.

**Numerical edge case:** if chunk projections are nearly orthogonal in *direction* (extreme content variation
across windows), their centroid has small norm and the L2-normalize step amplifies floating-point noise. In
practice this never occurs for natural audio sharing a single provenance. The implementation does not
special-case it; the resulting direction will be numerically unstable but the embedding remains unit-norm.

**Allocation budget per call (after warmup):**
- `embed`: only the output `Embedding`. Mel scratch, FFT scratch, and ONNX input backing live on the encoder.
- `embed_batch(N)`: `Vec<Embedding>` of N entries; mel scratch grows once on the first batch of a new max
  size, reused thereafter.
- `embed_chunked(L, batch=B)`: scratch sized to `B` once, reused; per-call cost is the `Vec<[f32; 512]>` of
  `ceil(L/hop)` un-normalized projections during aggregation.

`warmup()` runs a single `embed` (480 000 samples of silence) which sizes the steady-state scratch and
triggers ORT operator specialization — production paths see allocation-free, fully-specialized inference from
request 1.

## 9. Text inference pipeline

### 9.1 Tokenizer

Loaded once at construction via `tokenizers::Tokenizer::from_bytes` / `from_file`. textclap inspects the
tokenizer at construction to cache:
- `pad_id: i64` — resolved as
  `tokenizer.get_padding().map(|p| p.pad_id).unwrap_or_else(|| tokenizer.token_to_id("<pad>").unwrap_or(1))`
  (RoBERTa's pad id is 1; the explicit fallback chain matches HF Python's behavior).
- `max_length: usize` — from the tokenizer's truncation configuration.

If the loaded `tokenizer.json` lacks padding configuration, textclap calls `Tokenizer::with_padding(...)` to
enable batch-longest padding using the resolved `pad_id`. Padding then happens inside `encode_batch` — not
re-applied manually by the encoder.

### 9.2 `TextEncoder` orchestration

> **§3.2 backfill required:** the exact tensor names and dtypes (`input_ids`, `attention_mask`, possibly
> `position_ids`) are confirmed by `inspect_onnx.py` before implementation. The implementation supplies
> exactly the inputs the graph expects.

**`embed(text)`:**
1. Reject empty `text` with `Error::EmptyInput { batch_index: None }`.
2. `tokenizer.encode(text, add_special_tokens=true)` → `Encoding` (input_ids, attention_mask as `Vec<u32>`).
3. Resize encoder-owned `ids: Vec<i64>` and `mask: Vec<i64>` to `T`; cast u32→i64 and copy.
4. ort tensors `input_ids: [1, T] i64`, `attention_mask: [1, T] i64` (and `position_ids` if §3.2 says so).
5. `session.run()` → output `[1, 512]` un-normalized projection.
6. L2-normalize → `Embedding`.

**`embed_batch(texts)`:**
1. Empty slice → `Ok(Vec::new())`.
2. For each `texts[i]`: empty string → `Error::EmptyInput { batch_index: Some(i) }`.
3. `tokenizer.encode_batch(texts)` → `Vec<Encoding>` — **pre-padded to longest in batch** by the tokenizer
   itself (no manual padding step). Every encoding's `input_ids` has the same length `T_max`.
4. Resize encoder-owned ids and mask scratch to `[N × T_max]`; copy each encoding's already-padded
   `input_ids` and `attention_mask` into the corresponding row.
5. ort tensors `[N, T_max] i64` × 2 (or × 3 if `position_ids` is externalized).
6. `session.run()` → output `[N, 512]` un-normalized projections.
7. Row-by-row L2-normalize → `Vec<Embedding>`.

## 10. Error type

Single `thiserror` enum, exposed at crate root, `#[non_exhaustive]` for additive evolution.

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("failed to load ONNX model: {0}")]
    OnnxLoad(#[source] ort::Error),

    /// Tokenizer failed to load. The wrapped error is from the `tokenizers` crate;
    /// it is type-erased via Box because the upstream type is awkward to expose
    /// directly, but the source chain is preserved.
    #[error("failed to load tokenizer")]
    TokenizerLoad(#[source] Box<dyn std::error::Error + Send + Sync>),

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

    #[error("invalid chunking option: {field:?} must be > 0")]
    ChunkingConfig { field: ChunkingField },

    /// Tokenization failed (rare; mostly malformed Unicode or vocabulary issues).
    #[error("tokenization failed")]
    Tokenize(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// Empty `&str` passed to TextEncoder::embed, or empty string at the given
    /// index inside a batch passed to embed_batch. Empty *batch slices* are
    /// not an error — they return an empty Vec.
    #[error("input text is empty (batch index: {batch_index:?})")]
    EmptyInput { batch_index: Option<usize> },

    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch { expected: usize, got: usize },

    /// Slice passed to from_slice_normalizing was all zeros (or numerically
    /// indistinguishable from zero). Distinct from EmbeddingNotUnitNorm,
    /// which signals a stored unit vector failed its norm check.
    #[error("embedding is the zero vector (dim: {dim})")]
    EmbeddingZero { dim: usize },

    #[error("embedding norm out of tolerance: |norm² − 1| = {deviation:.3e}")]
    EmbeddingNotUnitNorm { deviation: f32 },

    #[error("unexpected tensor shape: expected {expected}, got {got}")]
    UnexpectedTensorShape { expected: String, got: String },

    #[error("ONNX runtime error: {0}")]
    Onnx(#[from] ort::Error),
}
```

## 11. Engineering robustness

### 11.1 ort version coupling

`ort = "=2.0.0-rc.12"` is exact. silero, soundevents, and textclap must all pin to the same RC. Bumping
requires a coordinated change across the trio; otherwise a downstream user combining them will fail to
compile. The README documents this coupling explicitly.

### 11.2 Model file integrity

The README publishes the SHA256 of each known-good model artifact (also recorded in
`tests/fixtures/MODELS.md` per §3.3):
- `audio_model_quantized.onnx` (33 MB)
- `text_model_quantized.onnx` (121 MB)
- `tokenizer.json` (2.0 MB)

If a user supplies a different file, behavior is undefined — typically a runtime tensor-shape error or, worse,
silent embedding drift. The README warns prominently. Xenova ships fp32, fp16, and int8 variants under similar
URLs; picking the wrong one is easy.

### 11.3 Quantization variant compatibility

textclap 0.1.0 is verified against the **INT8-quantized** export specifically. Other precisions are expected
to work (same I/O contract) but their golden-test tolerances would differ:

| Variant | Audio embedding tolerance vs Python int8 reference | Notes              |
|---------|----------------------------------------------------|--------------------|
| int8    | < 1e-4 (verified target — see §12.2)               | This release       |
| fp16    | likely < 5e-3                                      | Not verified       |
| fp32    | likely < 1e-2                                      | Not verified       |

The README states: "0.1.0 is verified against the int8 quantized export. Other precisions should work but
have not been tested." A quantization-tolerance matrix populated for fp16 and fp32 is a §14 follow-up.

### 11.4 Cold-start latency

First `session.run()` after construction includes ORT operator specialization, which can be 5–20× slower than
steady-state inference. `AudioEncoder::warmup(&mut self)`, `TextEncoder::warmup(&mut self)`, and
`Clap::warmup(&mut self)` run a single dummy forward each — 480 000 samples of silence for audio,
`"hello world"` for text — so production paths see steady-state latency from the first real request. The
dummy forward also sizes the encoder-owned scratch buffers to the steady-state for batch size 1.

**Caveat:** `warmup()` only sizes scratch for `N=1`. Workloads that batch will see one extra scratch growth
on the first batched call. A `warmup_for_batch(audio_n, text_n)` variant is a §14 follow-up. README
quick-start example calls `clap.warmup()?` after construction.

### 11.5 Test determinism and platform variance

Integration tests construct sessions with `Options::new().with_intra_threads(1)` so reduce-order variability
across thread schedules doesn't introduce flake. Real users should not set this — the production default is
whatever ort decides.

**Hardware and platform variance is intrinsic.** ORT's CPU EP differs across OSes (MLAS on Linux/Windows,
Accelerate on macOS); FMA fusion and vectorization differ between x86 and ARM, and between AVX-512 and AVX2.
Even with `intra_threads=1`, embedding values can differ at the ULP level across runners. CI tolerances
(§12.2) are calibrated to absorb this — they are not zero.

### 11.6 Model attribution and license compliance

The crate ships with no model files, but the README's "Model attribution" section states clearly that
**downstream users redistributing model files take on the attribution responsibilities** of the upstream
licenses:
- LAION CLAP weights are **CC-BY 4.0** — attribution required when redistributing.
- Xenova ONNX export is **Apache-2.0**.
- HTSAT and CLAP papers have **citation requirements** (BibTeX in README).

The README points to original LAION model card, Xenova export page, and paper citations.

## 12. Testing strategy

### 12.1 Unit tests (per module)

- **`mel.rs`:**
  - Hann window numerical correctness (periodic convention; first 1024 samples of length-1025 symmetric).
  - Filter rows 0 and 32 vs `filterbank_row_0.npy` / `filterbank_row_32.npy` (tolerance 1e-6).
  - power_to_dB applied exactly once after the mel filterbank (§8.1.2).
  - Repeat-pad behavior: `len < target` tiles correctly; `len == 0` is rejected upstream so not exercised
    here.
  - eps clamp on silence input (no NaN/Inf in the log transform).
  - Output buffer shape is exactly `[64 × T]` where `T` matches `golden_params.json`.
- **`audio.rs`:**
  - `AudioTooLong` boundary at exactly 480_001 samples (480_000 must succeed).
  - `EmptyAudio` rejection on `embed(&[])` / `embed_chunked(&[], ..)` and batch-with-empty-clip (with correct
    `clip_index`).
  - `embed_batch` with **uneven-length** clips succeeds (per §8.2 auto-pad).
  - Empty batch slice returns empty `Vec`.
  - Chunked windowing offsets and chunk counts for representative `(L, window, hop)` triples.
- **`text.rs`:**
  - `EmptyInput` for empty `&str` and for a batch containing an empty string at index `i`.
  - Empty batch slice returns empty `Vec`.
  - Batch tokenizer pads to longest-in-batch automatically (assert all `Encoding::ids` lengths equal after
    `encode_batch`); no manual `pad_id` filling.
- **`clap.rs`:**
  - `classify` returns top-k in score-descending order.
  - `classify_all` returns one entry per input label.
  - Stable ordering on tied scores.
  - `LabeledScore::to_owned()` preserves label and score.
- **`options.rs`:**
  - Builder methods round-trip through accessors.
  - Defaults match documented values.
  - `with_intra_threads(0)` round-trips (does not silently coerce to 1).
- **`Embedding`:**
  - `from_slice_normalizing` always produces unit-norm output for non-zero input.
  - `from_slice_normalizing` rejects all-zero input with `Error::EmbeddingZero`.
  - `try_from_unit_slice` rejects wrong lengths *and* non-unit-norm input (release-mode check at 1e-4).
  - `dot` equals `cosine` for unit inputs (within fp32 ULP).
  - `to_vec` and `as_slice` are byte-equal.
  - No `pub const DIM` exists (compile-time test that `Embedding::DIM` doesn't resolve).
  - Custom `Debug` output does not contain 512 floats; format is `Embedding { dim: 512, head: [..] }`.

### 12.2 Integration test (`tests/clap_integration.rs`)

Gated on `TEXTCLAP_MODELS_DIR` env var (skip with `eprintln!` if unset, do not fail). Models are not committed.

Sessions constructed with `intra_threads(1)` for determinism (§11.5).

Fixtures (committed):
- `tests/fixtures/sample.wav` — public-domain dog-bark WAV, ≤10 s, 48 kHz mono. Provenance and license
  attribution in `tests/fixtures/README.md`.
- `tests/fixtures/golden_params.json` — parameters dumped from `ClapFeatureExtractor`.
- `tests/fixtures/golden_onnx_io.json` — ONNX graph IO inspection.
- `tests/fixtures/golden_mel.npy` — `[64, T]` HF reference mel features.
- `tests/fixtures/golden_audio_proj.npy` — `[512]` int8 ONNX audio projection (un-normalized).
- `tests/fixtures/golden_audio_emb.npy` — `[512]` int8 ONNX audio embedding (L2-normalized).
- `tests/fixtures/golden_text_projs.npy` — `[5, 512]` int8 ONNX text projections (un-normalized).
- `tests/fixtures/golden_text_embs.npy` — `[5, 512]` int8 ONNX text embeddings (L2-normalized).
- `tests/fixtures/regen_golden.py` — pinned-version Python that produced the goldens.
- `tests/fixtures/inspect_onnx.py` — ONNX graph IO dumper.
- `tests/fixtures/MODELS.md` — SHA256s + HF revision + URLs.
- `tests/fixtures/README.md` — provenance + license attribution for sample.wav.

Assertions:

| Check                                                          | Tolerance (`max_abs_diff`)         |
|----------------------------------------------------------------|-------------------------------------|
| Rust mel features vs `golden_mel.npy`                          | < 1e-4                              |
| Rust audio projection vs `golden_audio_proj.npy`               | < 1e-4 (un-normalized, post-HTSAT) |
| Rust audio embedding vs `golden_audio_emb.npy`                 | < 1e-4 (matches mel propagation reality after L2 normalize) |
| Rust text embeddings vs `golden_text_embs.npy`                 | < 1e-5 (text input is integer ids; no upstream drift) |
| `classify_all` discrimination check (see below)                | structural, not absolute            |

**Why audio is 1e-4 and text is 1e-5.** Mel features drift up to 1e-4 (the mel test budget); that drift
propagates through HTSAT (many fmadds, even quantized) and compounds before L2 normalize. 1e-5 is not
physically achievable downstream of 1e-4 mel. Text inputs are integer token ids — no upstream drift — so
1e-5 is realistic and tighter to catch RoBERTa wiring bugs. Both budgets absorb ORT cross-binding floating-
point determinism only (intra_threads(1) eliminates reduce-order variance) plus hardware ULP variance
across CI runners.

**Discrimination check (replaces "ranks #1 exactly"):** `classify_all` is run with the five labels above. The
test asserts:
1. `"a dog barking"` ranks in the top 2.
2. `score("a dog barking") - score("music") > 0.05` (the irrelevant baseline).

Tie-breaks between dog-bark and "speech" can plausibly swap under int8 quantization, so the test does not
require exact #1 — it requires the model to *discriminate* (large margin against unrelated labels), which is
the property that actually matters for retrieval.

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
- **doctest** (Linux, all features)
- **coverage** (tarpaulin → codecov, Linux)
- **integration job** (Linux only) — fetches model files into a runner cache before tests, sets
  `TEXTCLAP_MODELS_DIR`, runs `cargo test --test clap_integration`

Removed from the original template's CI: WASM/RISC-V/PowerPC64 cross-compile targets `ort` doesn't support;
Miri/ASAN/LSAN/MSAN/TSAN/Loom (we're `#![forbid(unsafe_code)]` and write no custom sync primitives).

Tolerances in §12.2 absorb cross-OS ORT EP variance (MLAS vs Accelerate). If a runner consistently exceeds a
budget by a fixed amount, that's a hardware/EP signal, not a Rust bug — investigated, then either widened
with documentation or papered over with a per-OS budget table.

## 13. Migration from current template

textclap is currently the bare `al8n/template-rs` scaffold (single "Initial commit", `src/lib.rs` is 11
lines of lint config, version 0.0.0, no deps).

### Replace
- `Cargo.toml` (identity, deps, dev-deps including silero/rubato, features, MSRV, version 0.1.0, exact `ort` pin).
- `README.md` — purpose, install, quick-start (`Clap::from_files` → `warmup()` → audio embed + text embed
  + zero-shot classify), model-acquisition note pointing to HuggingFace **with SHA256s** and HF revision pin,
  explicit warning that `tokenizer.json` must come from the same Xenova export (not from
  `laion/clap-htsat-unfused` directly — they differ subtly and produce token-id mismatches that pass tests
  on common English but break on edge cases), model-attribution-on-downstream section (§11.6), ort-coupling
  note, license, lancedb integration snippet, deployment note that thread-per-core means each worker calls
  `from_files` once at startup with **150–300 MB resident per worker** (measure on your hardware), pipeline
  example showing 48-vs-16 kHz handling and the role of CLAP for non-speech context only (§1.2).
- `src/lib.rs` (keep crate-level lints; replace body with module decls and re-exports).
- `tests/foo.rs` → delete; replaced by per-module unit tests + `tests/clap_integration.rs`.
- `benches/foo.rs` → delete; replaced by the three Criterion benches.
- `examples/foo.rs` → delete; add `examples/index_and_search.rs` and `examples/vad_to_clap.rs`.
- `CHANGELOG.md` → reset to Keep-a-Changelog stub starting at `[0.1.0]`.

### Keep
- `build.rs` — adds a one-line comment documenting that it sets `cfg(tarpaulin_include)` based on
  `CARGO_CFG_TARGET_OS`/coverage env vars so coverage runs can selectively include/exclude blocks. Used by
  sibling crates' coverage runs.
- License files (`LICENSE-MIT`, `LICENSE-APACHE`, `COPYRIGHT`); update copyright holder/year.
- `.github/workflows/` skeleton, with deletions per §12.5.

### Add
- `src/error.rs`, `src/options.rs`, `src/mel.rs`, `src/audio.rs`, `src/text.rs`, `src/clap.rs`.
- `tests/fixtures/` contents per §3 / §12.2 (including `README.md` for sample.wav provenance and `MODELS.md`
  for model SHA256s).
- `examples/index_and_search.rs` — pipeline shape, lancedb stubbed.
- `examples/vad_to_clap.rs` — full audio path: 48 kHz source → `rubato` resample to 16 kHz → silero VAD →
  segment time ranges (using `mediatime::TimeRange`) → slice the *original* 48 kHz audio at those times →
  `textclap::AudioEncoder::embed`. Demonstrates the bridging concern from §1.1 explicitly. The example
  imports `mediatime` types where silero exposes them; the audio data into textclap is a plain `&[f32]`.
- This spec under `docs/superpowers/specs/`.

### lancedb integration snippet (for README)

```rust
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use textclap::Embedding;

// Ingest (clap is owned mutably by this worker thread):
let embedding: Embedding = clap.audio_mut().embed(&pcm_48khz_mono)?;
let dim = embedding.as_slice().len() as i32;          // dimension-agnostic
let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
builder.values().append_slice(embedding.as_slice()); // copies into Arrow's MutableBuffer
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
- An optional **strict LAION-reference mode** for `embed_chunked` that does single-window rand_trunc with a
  caller-provided RNG seed instead of multi-window aggregation, for cross-tool retrieval interop. Today's
  workaround: callers do `embed(&samples[..480_000.min(len)])` themselves.
- A doctest on `Embedding::cosine` showing the lancedb round-trip specifically (`try_from_unit_slice` of a
  stored vector → `cosine` against the query embedding); sketched in §13 README snippet but not yet a
  runnable doctest.
- `tracing` feature for service-tier observability (per-call timing, batch sizes, warmup spans).
