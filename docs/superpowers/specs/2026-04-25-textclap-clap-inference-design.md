# textclap — CLAP Inference Library Design

**Status:** Draft (revision 6, post-rev-5 review)
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
exports are expected to work (same I/O contract) but have not been measured. See §11.3.

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
textclap does not resample. Realistic deployment: the source audio arrives in some native rate (commonly
48 kHz from microphones or 44.1 kHz from media files); the caller resamples to 16 kHz for silero; silero
returns segment boundaries in *time* (e.g. via `mediatime::TimeRange`); the caller slices the *original
48 kHz* audio at those boundaries and feeds that slice to textclap. `examples/vad_to_clap.rs` demonstrates
this with `rubato`.

### 1.2 What CLAP does — and doesn't do — on speech segments

**Domain-of-training mismatch.** CLAP-HTSAT-unfused was trained on AudioSet plus general-audio captions, not
on conversational speech. Speech segments embedded through the audio encoder cluster tightly in CLAP-audio
space and do not discriminate well *between speech contents* — that's not the model's job. In the pipeline
above, the audio embedding captures **non-speech acoustic context accompanying the speech** (background
sounds, ambience, music, dog barks behind a conversation, traffic noise); discrimination between *what was
said* lives on the **text** branch (Whisper transcript → CLAP text encoder).

**Short-segment artifacts.** CLAP expects 10 s of audio. textclap's repeat-pad fills shorter clips by tiling,
which produces real periodicity artifacts in the mel spectrogram: a 0.5 s segment tiled to 10 s has a 2 Hz
periodicity and 20 identical positional patches in HTSAT's input. **Recommended minimum is ~2 s of original
content**. Below that, embeddings occupy a region of CLAP-audio space the model wasn't trained on and
results degrade unpredictably. silero VAD routinely emits 200–400 ms segments — callers should merge or
drop those before reaching `embed`. A `pad_mode: silence` alternative is a §14 follow-up.

## 2. Non-goals

- **Audio resampling.** Input must be 48 kHz mono `f32` PCM. Caller's responsibility.
- **Streaming inference.** CLAP isn't streaming.
- **Vector store integration.** Embeddings are emitted; storage and ANN search live in the caller.
- **Model bundling or download helpers.** No models in the crate, no network at build or runtime.
- **Async / runtime ownership.** Synchronous library; no in-flight cancellation. Callers wrap in their own
  cancellation scope (timeout, `tokio::select!`, etc.) if needed.
- **Multi-variant CLAP support in 0.1.0.** Only the 512-dim `Xenova/clap-htsat-unfused` export is verified
  (against INT8). The public API does not lock to this dimension (§7.5).
- **NaN/Inf-safe arithmetic.** Non-finite samples are detected and rejected up front (§7.3, §10); they do
  not propagate into the model.
- **Cross-tool embedding interop for chunked audio.** textclap's `embed_chunked` is a textclap-specific
  convention, not LAION-reference compatible (§7.3, §8.2). Single-window `embed` (≤10 s) does match the
  LAION reference within the verified tolerance.

## 3. Pre-implementation prerequisites

Several parameters in the audio preprocessing pipeline cannot be safely guessed; they must be measured against
the actual model files before any Rust is written. This section drives the prerequisite scripts and the
spec-update gate that follows them.

### 3.1 Reference-parameter dump and golden generation

`tests/fixtures/regen_golden.py` (pinned `transformers` / `optimum` / `onnxruntime` / `torch` / `librosa`
versions in a header comment):

1. Loads the test audio fixture (`tests/fixtures/sample.wav`, ≤10 s, 48 kHz mono — provenance and license
   in `tests/fixtures/README.md`).
2. Constructs `ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")` and writes
   `tests/fixtures/golden_params.json` capturing every numerical-fidelity parameter, **read directly from
   the constructed extractor object** (do not trust documentation):
   - `sampling_rate`, `feature_size` (mel bins), `fft_window_size`, `hop_length`, `max_length_s`
   - `mel_scale`, filterbank `norm` argument
   - `power_to_db`: `amin`, `ref`, **`top_db` (read directly from `extractor.top_db`; do not assume `None`)**
   - Window function: periodic vs symmetric (read by computing the actual array and inspecting it)
   - **Frame count `T`** — the time dimension produced by the extractor on a 480_000-sample input.
     With `n_fft=1024`, `hop=480`: `center=False` ⇒ 998 frames; `center=True` ⇒ 1001. The HF
     preprocessor_config claims `max_frames=1000`. The actual value, the centering flag, and the pad mode
     all go into `golden_params.json`. The spec stops asserting "1000" anywhere; mel.rs uses `T` from
     `golden_params.json`.
   - `padding` mode, `truncation` mode (the latter intentionally diverges in Rust — see §8.1).
   - `frequency_min` / `frequency_max`.
3. Runs the extractor; saves resulting `[64, T]` mel features to `golden_mel.npy`.
4. **Audio model goldens.** Loads `audio_model_quantized.onnx` via `onnxruntime.InferenceSession`; runs it on
   the golden mel features. Saves the **raw model output** as `golden_audio_proj.npy`. Then computes the
   L2-normalized embedding using the *exact* formula
   ```python
   x = raw_output.astype(np.float32)
   norm = np.linalg.norm(x).astype(np.float32)
   embedding = (x / norm).astype(np.float32)
   ```
   (deliberately *not* `torch.nn.functional.normalize`, which can differ in summation order) and saves to
   `golden_audio_emb.npy`. **If §3.2 detects an internal L2-normalize at the ONNX graph tail, raw_output is
   already unit-norm and `golden_audio_proj.npy` is redundant.** In that case the script omits it and the
   spec's "embed_chunked uses un-normalized projections" branch falls back to spherical-mean aggregation
   (see §8.2). The decision is recorded in `golden_onnx_io.json`.
5. **Text model goldens.** Loads `text_model_quantized.onnx` and `tokenizer.json`. Runs five fixed labels
   (`["a dog barking", "rain", "music", "silence", "door creaking"]` — note "speech" is *deliberately not*
   in the label set, since CLAP does not discriminate well within speech, §1.2). Saves
   `golden_text_projs.npy` and `golden_text_embs.npy` with the same `np.linalg.norm` formula.

**Why goldens come from the int8 ONNX, not the fp32 PyTorch path:** the Rust crate runs the int8 ONNX.
Goldens must run the same int8 ONNX in Python — otherwise the test tolerance has to absorb both quantization
drift *and* implementation differences, indistinguishably. Mel goldens still come from `ClapFeatureExtractor`
(fp32 NumPy).

**To verify a non-int8 setup,** users regenerate the goldens by pointing `regen_golden.py` at an alternate
ONNX file. Mel goldens don't change; audio/text goldens are recomputed; tolerances may need to widen
(§11.3).

### 3.2 ONNX graph IO inspection — and functional verification

`tests/fixtures/inspect_onnx.py` does both static graph inspection and a functional end-to-end check.

**Static inspection.** For each ONNX file, dump `graph.input` / `graph.output` (name, dtype, shape with
dynamic dims marked) and the first/last 20 graph nodes into `tests/fixtures/golden_onnx_io.json`. From this
file the spec answers:

- **Audio input shape:** `[batch, 1, 64, T]` *vs* `[batch, 64, T]` (channel dim present?).
- **Audio output L2-normalize?** Examine the last 5 graph nodes for an `LpNormalization` op (axis=-1, p=2)
  or the equivalent `ReduceL2` + `Div` pattern. Record `audio_output_is_unit_norm: true|false`. This drives
  whether `golden_audio_proj.npy` is generated (§3.1) and which aggregation path `embed_chunked` uses (§8.2).
- **Text input names and dtypes:** `input_ids: [batch, T] i64`, `attention_mask: [batch, T] i64`, **plus**
  whether `position_ids` appears as a third input. The implementation matches whatever the graph requires.
- **Text output L2-normalize?** Same check as audio.
- **Text truncation max_length** — read from the `tokenizers` Python binding's `tokenizer.truncation`
  property. Some Xenova exports keep RoBERTa's default 512 instead of CLAP's expected 77.
- **Audio output / text output names and shapes** (`[batch, 512]` expected; verify names).

**Functional verification of HTSAT input normalization.** Static inspection alone is insufficient — absence
of `Sub`/`Div`/`BatchNorm` near the input doesn't prove the model accepts raw log-mels. The script also runs:

1. Load `laion/clap-htsat-unfused` in PyTorch (fp32) via `transformers.ClapModel.from_pretrained`.
2. `pt_emb = pt_model.get_audio_features(extractor(sample.wav))` — fp32 reference, normalized.
3. `ort_raw = onnxruntime_audio_session.run({input: extractor(sample.wav)})[0]` — int8 export with the
   same extractor output, then L2-normalize externally if §3.2's static check said the output is *not*
   already unit-norm.
4. Compare `pt_emb` and `ort_emb` (both unit-norm) by `max_abs_diff`. If `< 1e-2`, the int8 export accepts
   raw log-mels — Rust mel.rs needs no extra normalization. If `≥ 1e-2`, the int8 export expects external
   normalization that the extractor doesn't provide; the script then probes per-mel-bin mean/std
   normalization (starting from AudioSet stats `mean=−4.27`, `std=4.57` if available, or computed from a
   small reference audio batch) until the embeddings agree.
5. Whatever transformation makes them agree is recorded in `golden_params.json` as
   `htsat_input_normalization: { type: "none" | "global_mean_std", mean: f32, std: f32 }` and applied
   by Rust's mel.rs post-log-mel step.

This is the *only* way to catch a silent normalization mismatch. Skipping it puts the implementation at
risk of producing embeddings in the wrong distribution with no test that fails loudly enough to notice.

### 3.3 Model SHA256 acquisition

Before §3.1 / §3.2, the maintainer downloads the three model artifacts from a pinned Hugging Face revision
(commit hash recorded in `tests/fixtures/MODELS.md`) and computes:

```
shasum -a 256 audio_model_quantized.onnx text_model_quantized.onnx tokenizer.json
```

The SHA256s, the HF revision hash, and the URL are recorded in `tests/fixtures/MODELS.md` and reproduced in
the README. Re-running `regen_golden.py` against a different revision recomputes goldens; mismatched SHA256s
in user setups produce undefined results (loud README warning).

### 3.4 Spec-update commit sequence

§3.1–§3.3 produce a multi-commit sequence:

1. **Scripts commit:** `regen_golden.py` and `inspect_onnx.py` source.
2. **Generated-fixtures commit:** `golden_params.json`, `golden_onnx_io.json`, `golden_*.npy`, `MODELS.md`,
   `tests/fixtures/README.md`.
3. **Spec-update commit:** §8.1 mel parameter table, §8.2 / §9.2 tensor names and shapes, §7.4
   attention-mask / position_ids description, §12.2 tolerance table — all replaced with values consistent
   with the generated fixtures.
4. **Rust implementation commits.**

Steps 1–3 must land before any Rust source.

## 4. Crate layout

```
textclap/
├── Cargo.toml
├── build.rs                        # sets cfg(tarpaulin_include) under coverage builds; no-op otherwise
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
│       ├── golden_params.json
│       ├── golden_onnx_io.json
│       ├── golden_mel.npy
│       ├── golden_audio_proj.npy   # only present if ONNX audio output is NOT unit-norm
│       ├── golden_audio_emb.npy
│       ├── golden_text_projs.npy   # only present if ONNX text output is NOT unit-norm
│       ├── golden_text_embs.npy
│       ├── filterbank_row_0.npy    # librosa-precomputed mel filterbank row 0
│       ├── filterbank_row_10.npy   # row near 1 kHz (Slaney inflection — discriminates Slaney vs HTK)
│       ├── filterbank_row_32.npy   # mid-band reference row
│       ├── regen_golden.py
│       └── inspect_onnx.py
├── benches/
│   ├── bench_mel.rs
│   ├── bench_audio_encode.rs
│   └── bench_text_encode.rs
├── examples/
│   ├── index_and_search.rs         # end-to-end pipeline shape (lancedb stubbed)
│   └── vad_to_clap.rs              # silero (16 kHz) + rubato resample → textclap (48 kHz)
└── docs/superpowers/specs/
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

| Crate        | Version | Used by                                |
|--------------|---------|----------------------------------------|
| `criterion`  | `^0.5`  | `benches/`                             |
| `silero`     | path    | `examples/vad_to_clap.rs`              |
| `mediatime`  | path    | `examples/vad_to_clap.rs`              |
| `rubato`     | `^0.16` | `examples/vad_to_clap.rs` (resampling) |
| `npyz`       | `^0.8`  | `tests/clap_integration.rs` (.npy reader) |
| `hound`      | `^3`    | `tests/clap_integration.rs` (WAV reader) |

`silero` / `mediatime` are referenced via path-deps. `examples/` are marked `publish = false` so the
published crate's `cargo build --examples` does not depend on the workspace layout.

### Excluded (deliberate)

- No `tokio`, no async — synchronous library.
- No `download` feature — no network, no `ureq`/`sha2`/`reqwest`.
- No model bundling — no `bundled` feature.
- No BLAS / `ndarray` — the mel filterbank multiply is small enough to write by hand.
- No `tracing` for 0.1.0 — observability is a §14 follow-up.
- No `num_cpus` — `with_intra_threads(0)` is forwarded to ORT verbatim (see §7.7).

## 6. Toolchain & metadata

- **Rust edition:** 2024
- **MSRV:** 1.85
- **License:** MIT OR Apache-2.0
- **Crate-level lints:** `#![deny(missing_docs)]`, `#![forbid(unsafe_code)]`
- **Initial version:** `0.1.0`

## 7. Public API

All public structs use private fields and accessor methods. Builder-style `with_*` methods return `Self` by
value; getters return references or `Copy` values. Field-less unit enums (`Error` variants, `ChunkingField`)
are public-as-data. **No public signature exposes `[f32; 512]`** — this keeps the door open for swapping in
larger CLAP variants.

### 7.1 Top-level types

```rust
pub struct Clap          { /* AudioEncoder + TextEncoder */ }
pub struct AudioEncoder  { /* ort::Session + MelExtractor + encoder-owned scratch */ }
pub struct TextEncoder   { /* ort::Session + Tokenizer + cached pad_id + encoder-owned scratch */ }

pub struct Embedding     { /* invariant: L2-normalized, internal storage [f32; 512] */ }

pub struct LabeledScore<'a>      { /* private; borrows label */ }
pub struct LabeledScoreOwned     { /* private; owns label */ }

#[derive(Default)]
pub struct Options       { /* private */ }

#[derive(Default)]
pub struct ChunkingOptions { /* private */ }

#[non_exhaustive]
pub enum ChunkingField { Window, Hop, BatchSize }   // identifies which field violated ChunkingConfig

pub type Result<T, E = Error> = std::result::Result<T, E>;
```

`AudioEncoder` and `TextEncoder` own their `ort::Session` and (for audio) the mel-extractor state by value.
There is no internal `Arc`, no clone-with-shared-session, and no cross-thread session sharing. **The
deployment model is thread-per-core**: each worker thread loads its own encoder once at startup. Memory cost
is **150–300 MB resident per worker** for both encoders together (33 MB int8 audio model + 121 MB int8 text
model on disk, plus ORT working buffers and weight layout — measure on your hardware). README recommends
**sequential** worker construction at startup to avoid transient 2× peak memory during ORT weight
reformatting (M6 from rev-5 review).

`ChunkingField` is re-exported from `lib.rs` since it appears in `Error::ChunkingConfig`.

**Drop / shutdown.** Encoders implement `Drop` safely. They are `!Sync` and owned by exactly one thread; on
drop the ORT session releases its weights and the scratch `Vec`s deallocate.

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
    pub fn warmup(&mut self) -> Result<()>;

    // Single ~10 s clip (LAION-reference compatible)
    pub fn classify<'a>(&mut self, samples: &[f32], labels: &'a [&str], k: usize)
        -> Result<Vec<LabeledScore<'a>>>;
    pub fn classify_all<'a>(&mut self, samples: &[f32], labels: &'a [&str])
        -> Result<Vec<LabeledScore<'a>>>;

    /// Long clip via textclap-specific chunking (NOT LAION-reference compatible
    /// — see §7.3 embed_chunked docs and §8.2).
    pub fn classify_chunked<'a>(
        &mut self, samples: &[f32], labels: &'a [&str], k: usize, opts: &ChunkingOptions,
    ) -> Result<Vec<LabeledScore<'a>>>;
}
```

**`classify` edge cases:**
- `labels.is_empty()` → `Ok(Vec::new())`.
- `k == 0` → `Ok(Vec::new())`.
- `k > labels.len()` → clamps to `labels.len()` (returns full ranking, no error).

`classify` is `classify_all` followed by heap-based top-k. Score is **cosine similarity** between
L2-normalized audio and text embeddings (range ≈ `[-1, 1]`); higher is more relevant. Order is descending by
score; tie-break is input-label order (stable).

### 7.3 `AudioEncoder`

```rust
impl AudioEncoder {
    pub fn from_file<P: AsRef<Path>>(onnx_path: P, opts: Options) -> Result<Self>;
    pub fn from_memory(onnx_bytes: &[u8], opts: Options) -> Result<Self>;  // bytes copied

    /// Wraps a pre-built ORT session. The session must conform to the input
    /// and output schema in tests/fixtures/golden_onnx_io.json — name and
    /// dtype are checked at construction; mismatches return Error::SessionSchema.
    pub fn from_ort_session(session: ort::session::Session, opts: Options) -> Result<Self>;

    /// Single clip, length 0 < len ≤ 480_000 samples (10 s @ 48 kHz):
    ///   - len == 0           → Error::EmptyAudio { clip_index: None }
    ///   - len > 480_000      → Error::AudioTooLong { got, max: 480_000 } (use embed_chunked)
    ///   - any non-finite     → Error::NonFiniteAudio { first_index } (caught up front)
    ///   - 0 < len < 480_000  → repeat-padded to 10 s by the mel extractor
    ///   - len == 480_000     → passes through without padding/truncation
    pub fn embed(&mut self, samples: &[f32]) -> Result<Embedding>;

    /// Batch of clips of any lengths in 0 < len ≤ 480_000. Each clip is
    /// repeat-padded to 10 s independently — no equal-length requirement.
    /// Empty *slice* returns Ok(Vec::new()). Any clip with len == 0 returns
    /// Error::EmptyAudio with its index. Any non-finite sample anywhere in
    /// the batch returns Error::NonFiniteAudio with the offending clip's
    /// index encoded into first_index (high bits) — see error docs.
    pub fn embed_batch(&mut self, clips: &[&[f32]]) -> Result<Vec<Embedding>>;

    /// Arbitrary-length input via textclap's chunking convention.
    ///
    /// **WARNING — not LAION-reference compatible.** LAION's reference for the
    /// unfused model uses single-window rand_trunc, not multi-window
    /// aggregation. Aggregation belongs to the *fused* CLAP variant, which
    /// does fusion inside the network. textclap aggregates by either
    /// (a) centroid-of-un-normalized-projections + L2-normalize, or
    /// (b) spherical-mean (mean of unit vectors) + L2-normalize.
    /// Which one is used depends on whether the ONNX export already
    /// L2-normalizes its output (§3.2 / golden_onnx_io.json determines this);
    /// the choice is made at construction and recorded.
    /// Embeddings produced by either variant are textclap-specific.
    /// Cross-tool retrieval requires both sides to use textclap.
    pub fn embed_chunked(&mut self, samples: &[f32], opts: &ChunkingOptions)
        -> Result<Embedding>;

    pub fn warmup(&mut self) -> Result<()>;
}
```

**Concurrency model:** `AudioEncoder` is `Send` but **not `Sync`**. Each worker thread owns its own
`AudioEncoder`. Encoder-owned mel feature buffer, FFT scratch, and ONNX input tensor backing are growable
`Vec<f32>`s; sized on the first call (or amortized via `warmup()`); reused thereafter via `Vec::resize_with`,
which preserves capacity — the hot path performs no heap allocation after warmup.

#### 7.3.1 Scratch lifecycle contract (UB prevention)

`#![forbid(unsafe_code)]` blocks `unsafe` blocks in textclap, but **`ort 2.x`'s `TensorRef::from_array_view`
constructs views that the ORT C++ runtime borrows during `session.run()`**. If the underlying scratch
buffer is reallocated (e.g. `Vec::resize_with` or `Vec::reserve` triggering capacity growth) after the tensor
view is bound and before `session.run()` returns, the C++ side accesses freed memory — undefined behavior
through the FFI boundary that Rust's safety lints cannot catch.

The implementation **must** obey this order in every public encoder method:

1. **Resize all scratch buffers** to the final size required for this call. This is the only point at which
   reallocation is permitted.
2. **Construct ORT tensor views** (`TensorRef::from_array_view`) over the now-stable buffers.
3. **Call `session.run()`**.
4. **Read outputs** into separate result types (`Vec<Embedding>` etc.).
5. **Drop tensor views**, typically by end of scope.

In particular: do not resize a buffer mid-call, do not call any helper that might resize between steps 2
and 5, and do not bind a tensor view to a buffer whose required final size is computed during step 2.

Each encoder method has a unit test that runs a small batch then a larger batch in sequence; the larger
batch must succeed without panic and produce embeddings that match independently-computed single-call
results within the §12.2 tolerance. This is the structural protection against accidental resize-during-run
regressions.

### 7.4 `TextEncoder`

```rust
impl TextEncoder {
    pub fn from_files<P: AsRef<Path>>(onnx_path: P, tokenizer_json_path: P, opts: Options) -> Result<Self>;
    pub fn from_memory(onnx_bytes: &[u8], tokenizer_json_bytes: &[u8], opts: Options) -> Result<Self>;

    /// Wraps a pre-built ORT session and Tokenizer. Both are validated:
    /// session schema vs golden_onnx_io.json; tokenizer must have padding
    /// configured as Padding::BatchLongest (or unconfigured, in which case
    /// textclap configures it). Padding::Fixed is rejected with
    /// Error::TokenizerLoad — fixed padding silently sends full-MaxLength
    /// batches and produces ~6× perf regressions.
    pub fn from_ort_session(
        session: ort::session::Session, tokenizer: tokenizers::Tokenizer, opts: Options,
    ) -> Result<Self>;

    /// Empty &str returns Error::EmptyInput. Whitespace-only strings are
    /// accepted as-is — they tokenize to <s> + minimal content + </s>.
    pub fn embed(&mut self, text: &str) -> Result<Embedding>;

    /// Empty *slice* returns Ok(Vec::new()). Any empty string in the batch
    /// returns Error::EmptyInput with its batch_index.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>>;

    pub fn warmup(&mut self) -> Result<()>;
}
```

**Concurrency model:** `Send` but **not `Sync`**. Encoder-owned scratch (`Vec<i64>` for token ids and
attention mask) is grown via `Vec::resize_with` and reused; the hot path is allocation-free after warmup.
The §7.3.1 scratch lifecycle contract applies identically here.

**Tokenizer truncation max_length is taken from `tokenizer.json` at construction** — no `max_length` knob is
exposed. The actual value (typically 77 for CLAP, sometimes 512 for RoBERTa default) is recorded in
`golden_params.json` by §3.1. Long inputs are silently truncated; this is documented loudly in the function's
rustdoc.

**Padding mode is forced to `BatchLongest`** at construction regardless of what `tokenizer.json` declares.
If the input JSON has `Padding::Fixed`, textclap replaces it. Rationale: fixed padding sends full-MaxLength
batches and produces ~6× perf regressions in typical Whisper-transcript workloads where most inputs are far
shorter than 77 tokens. The replacement is a pure performance fix; it does not change embedding values, since
the model's own attention mask handles variable lengths.

**Position-ids:** RoBERTa computes positions as `pad_id + 1 + cumsum(non_pad_mask)`. This is *typically*
inlined into Xenova's ONNX export; in that case the encoder feeds only `input_ids` and `attention_mask`. If
§3.2 finds `position_ids` as an externalized input, the encoder computes it explicitly using the resolved
`pad_id` (§9.1) — *not* the literal `1` — and feeds it as a third tensor. The attention mask is critical
either way.

### 7.5 `Embedding`

```rust
impl Embedding {
    pub fn dim(&self) -> usize;            // 512 for 0.1.0; runtime-queryable, future-proof

    // Borrow-only access — supports append_slice into Arrow's MutableBuffer.
    pub fn as_slice(&self) -> &[f32];

    // Owned conversion.
    pub fn to_vec(&self) -> Vec<f32>;

    /// Reconstruct from a stored unit vector. Validates length AND norm
    /// (release-mode check: `(norm² − 1).abs() < 5e-5`). The numerical
    /// budget reflects per-component drift propagating to norm² via
    /// `Δ(norm²) ≈ 2 · √(dim · Δ_max_per_component²)` — a 5e-5 norm² check
    /// corresponds to ~5e-6 max-per-component drift, tighter than the
    /// integration-test budget so storage round-trips don't trigger
    /// false positives. fp16 storage round-trip is OUT OF SCOPE for this
    /// constructor — fp16's ulp(1.0) ≈ 9.77e-4 makes the check fail; users
    /// converting through fp16 should use from_slice_normalizing.
    pub fn try_from_unit_slice(s: &[f32]) -> Result<Self>;

    /// Construct from any non-zero slice; always re-normalizes to unit length
    /// (idempotent for input that's already unit-norm). Validates length and
    /// rejects all-zero input via Error::EmbeddingZero.
    pub fn from_slice_normalizing(s: &[f32]) -> Result<Self>;

    // Similarity (== for unit vectors, modulo fp32 ULP).
    pub fn dot(&self, other: &Embedding) -> f32;
    pub fn cosine(&self, other: &Embedding) -> f32;

    /// Approximate equality test for use in user tests across runs/threads/
    /// hardware. Returns true if `(self - other).max_abs() < tol`. Replaces
    /// PartialEq, which is intentionally NOT derived.
    pub fn is_close(&self, other: &Embedding, tol: f32) -> bool;
}

impl AsRef<[f32]> for Embedding;          // delegates to as_slice()

// Custom Debug — does NOT dump 512 floats. Format:
//   Embedding { dim: 512, head: [0.0123, -0.0456, 0.0789, ..] }
impl fmt::Debug for Embedding;

// derives: Clone.
// NO PartialEq, Eq, or Hash. Bit-pattern equality across runs/threads/OSes/
// hardware is not reliable for f32 outputs of ML models (FMA fusion, ORT
// kernel choice, BLAS path, x86-vs-ARM, AVX-512-vs-AVX2 all produce ULP-level
// differences). Use `is_close(other, tol)` for tests; use the ANN index for
// similarity lookups in production.
#[cfg(feature = "serde")] // serializes as a sequence of 512 f32 values.
```

**No public method exposes a fixed-size array.** Internal storage is `[f32; 512]` for 0.1.0 (cheap,
stack-friendly); that detail can change to `Box<[f32]>` later to support 1024-dim CLAP variants without
breaking the public API. **There is no `pub const DIM`** — code that needs the dimension calls `dim()` or
`embedding.as_slice().len()`.

**Invariant:** every `Embedding` returned by this crate is L2-normalized to unit length within fp32 ULP.
Internal constructors divide raw model output by its L2 norm; `try_from_unit_slice` validates the invariant;
`from_slice_normalizing` re-establishes it. The un-normalized projections used internally by `embed_chunked`
never escape to the public API.

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
ideal for in-thread top-k. `LabeledScoreOwned` is for serialization / cross-thread send / DB rows.

### 7.7 `Options`

```rust
impl Options {
    pub fn new() -> Self;                                                 // == Self::default()
    pub fn with_graph_optimization_level(self, level: GraphOptimizationLevel) -> Self;
    pub fn graph_optimization_level(&self) -> GraphOptimizationLevel;

    /// Forwarded verbatim to ORT's intra-op thread count.
    /// 0 = inherit ORT's default (typically num_cpus). Verified against
    ///     ort=2.0.0-rc.12 — meaning has not changed across ORT 1.x → 2.x.
    /// 1 = single-threaded; deterministic across runs on the same hardware.
    /// >1 = multi-threaded with reduce-order non-determinism (§7.5 caveat).
    pub fn with_intra_threads(self, n: usize) -> Self;
    pub fn intra_threads(&self) -> usize;
}
```

`GraphOptimizationLevel` is re-exported from `ort`. `Default` is on the *struct* (§7.1), not the impl block.

### 7.8 `ChunkingOptions`

```rust
impl ChunkingOptions {
    pub fn new() -> Self;        // == Self::default(); window=480_000, hop=480_000, batch_size=8
    pub fn with_window_samples(self, n: usize) -> Self;
    pub fn window_samples(&self) -> usize;
    pub fn with_hop_samples(self, n: usize) -> Self;
    pub fn hop_samples(&self) -> usize;
    pub fn with_batch_size(self, n: usize) -> Self;
    pub fn batch_size(&self) -> usize;
}
```

Aggregation strategy is **fixed in 0.1.0** — the choice between centroid-of-un-normalized-projections and
spherical-mean is determined at encoder construction by §3.2's static graph inspection (whether the ONNX
output is already unit-norm), not user-selectable. There is no `Aggregation` enum and no `with_aggregation`
setter; both are §14 follow-ups.

Validation runs at use, not at build: `embed_chunked` returns `Error::ChunkingConfig { field }` if any of
`window_samples`, `hop_samples`, or `batch_size` is `0`, with `ChunkingField` identifying which.

Trailing chunks shorter than `window_samples / 4` (i.e. < 2.5 s for the 10 s default window) are **skipped**
to avoid dragging the centroid into noise from the trailing repeat-pad. If skipping would leave zero chunks
(input shorter than `window/4`), the single chunk is processed anyway — the user did ask.

## 8. Audio inference pipeline

### 8.1 Mel-spectrogram extractor (`src/mel.rs`)

`MelExtractor` is `pub(crate)`. Parameters expected (subject to §3.1 verification — values in this table are
*expected*; recorded values in `golden_params.json` are authoritative; only the truncation row is
intentionally chosen differently):

| Parameter            | Value                                             |
|----------------------|---------------------------------------------------|
| Sample rate          | 48 000 Hz                                         |
| Target samples       | 480 000 (10 s)                                    |
| `n_fft`              | 1024                                              |
| Hop length           | 480                                               |
| Window               | **Hann, periodic, length 1024**                   |
| Frame count `T`      | **TBD by §3.1**                                   |
| Mel bins             | 64                                                |
| Mel scale            | **Slaney**                                        |
| Filterbank norm      | **Slaney** (per-filter bandwidth normalization)   |
| Frequency range      | 50 – 14 000 Hz                                    |
| Power spectrum       | `|X|²` (squared magnitude)                        |
| Mel→dB transform     | **`10 · log10(max(amin, x))` with `amin = 1e-10`, `ref = 1.0`, `top_db = TBD by §3.1`; applied exactly once after the mel filterbank** |
| Padding mode         | repeatpad                                         |
| Truncation mode      | head (deterministic; intentionally differs from HF rand_trunc) |
| HTSAT input norm     | **TBD by §3.2 functional check**                  |

Pipeline per call:

```
samples (f32, 48 kHz mono, length L, finite — caller checked upstream)
  → pad-or-truncate to 480_000 samples           (repeatpad if L < target; head-truncate if L > target)
  → STFT (n_fft=1024, hop=480, periodic Hann)    via rustfft RealFftPlanner → [513 × T]
  → |·|² (power spectrogram)                     → [513 × T]
  → mel filterbank multiply (Slaney / Slaney)    → [64 × T]
  → 10 · log10(max(amin, x))                     → [64 × T]
  → (if §3.2 says so) global mean/std norm       → [64 × T]
  → write into caller-provided [64 × T] f32 buffer (row-major, time-major contiguous)
```

State allocated once in `new()`, owned by the `MelExtractor`:
- Hann window (`Vec<f32>`, len 1024, periodic convention).
- Mel filterbank (`Vec<f32>`, len 64 × 513).
- `RealFftPlanner<f32>` instance.

The ONNX input tensor `[N, 1, 64, T]` is built as a *view* over the mel feature scratch — the channel dim is
added at tensor-construction time with no data movement.

#### 8.1.1 Filterbank-correctness unit test

`mel.rs` ships unit tests that compare filter rows 0, 10, and 32 against pre-computed reference rows
(`librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64, fmin=50, fmax=14000, htk=False, norm='slaney')`)
committed as `tests/fixtures/filterbank_row_0.npy` etc. Tolerance: `max_abs_diff < 1e-6`. **Row 10 is
included specifically because it lands near the 1 kHz Slaney inflection point** — rows 0 and 32 alone do
not discriminate Slaney from HTK construction.

#### 8.1.2 Power-to-dB single-application test

Separate unit test asserts that feeding a known input through `MelExtractor::extract_into` differs visibly
from a hand-written "apply power_to_dB twice" reference. Confirms the floor is applied exactly once after
the filterbank, never before, never twice.

### 8.2 `AudioEncoder` orchestration

> §3.2 backfill: audio-input tensor name and shape, plus whether the audio output is already unit-norm,
> are confirmed by `inspect_onnx.py` before implementation.

**Internal helper.** A single `pub(crate)` method runs the full forward pipeline (mel → ONNX) for any
non-empty batch and writes raw model outputs into a caller-provided buffer:

```rust
pub(crate) fn embed_projections_batched(
    &mut self,
    clips: &[&[f32]],         // 1..=N clips, each 1..=480_000 samples
    out: &mut Vec<[f32; 512]>,
) -> Result<()>;
```

Single-clip operations call this with `clips = &[samples]`. There is no separate "single-clip raw" helper —
keeping one path eliminates the perf-profile ambiguity from rev-5.

**§7.3.1 scratch-lifecycle contract applies to this helper.** All scratch resizes happen up front (mel
scratch sized to `[N × 64 × T]`, ORT input tensor backing sized to match). Tensor views are bound *after*
all resizes complete. `session.run()` returns. Outputs are copied into `out`. Tensor views drop at end of
scope.

**`embed(samples)`:**
1. `samples.is_empty()` → `Error::EmptyAudio { clip_index: None }`.
2. `samples.len() > 480_000` → `Error::AudioTooLong { got, max: 480_000 }`.
3. **Finiteness scan:** SIMD pass over `samples` for the first non-finite value. On hit:
   `Error::NonFiniteAudio { first_index }`. Cost ≈ 50 µs over 480 k floats — dwarfed by ONNX work.
4. Call `embed_projections_batched(&[samples], &mut self.proj_scratch)`.
5. Take `self.proj_scratch[0]`; L2-normalize → `Embedding` (or, if §3.2 says outputs are already unit-norm,
   skip the normalize and just construct via the trusted constructor; debug_assert unit-norm).

**`embed_batch(clips)`:**
1. Empty slice → `Ok(Vec::new())`.
2. For each clip `i`: empty → `Error::EmptyAudio { clip_index: Some(i) }`; too-long → `AudioTooLong`;
   finiteness scan → `NonFiniteAudio` (the offending clip's index is encoded into `first_index` per
   the error's docs).
3. Call `embed_projections_batched(clips, &mut self.proj_scratch)`.
4. Row-by-row L2-normalize → `Vec<Embedding>`.

**`embed_chunked(samples, opts)`:**
1. `samples.is_empty()` → `Error::EmptyAudio { clip_index: None }`.
2. Validate `window_samples > 0 && hop_samples > 0 && batch_size > 0` → otherwise
   `Error::ChunkingConfig { field }`.
3. Finiteness scan over `samples`.
4. Compute chunk offsets `0, hop, 2·hop, …` while `offset < samples.len()`. Trailing chunks shorter than
   `window_samples / 4` are skipped (§7.8) unless the input itself is shorter than `window/4`.
5. For each group of `batch_size` chunks: call `embed_projections_batched(group, &mut tmp_proj)`, append
   raw projections to a per-call `Vec<[f32; 512]>`.
6. **Aggregate.** Two paths — selected at construction by §3.2:
   - **Centroid path** (ONNX outputs un-normalized projections): component-wise mean of the raw projections,
     then L2-normalize the centroid → `Embedding`.
   - **Spherical-mean path** (ONNX outputs already unit-norm — the raw projections are unit vectors):
     component-wise mean of the unit vectors, then L2-normalize → `Embedding`.

   Both paths are computationally equivalent (one extra normalize-per-chunk in the spherical-mean path,
   negligible). Single-chunk case skips aggregation entirely.

**Why aggregation at all and not single-window like LAION's reference:** the user's pipeline must handle
audio segments of arbitrary length even when no LAION-reference path exists for them; we acknowledge the
divergence loudly (§7.3 docstring) and pick a defensible, simple aggregator. The centroid-vs-spherical-mean
distinction is a property of the model artifact, not a user choice. **Both are textclap-specific. Cross-tool
retrieval requires both indexing and querying through textclap.**

**Numerical edge case:** if chunk projections / unit vectors are nearly orthogonal (extreme content variation
across windows), their centroid / mean has small norm and the L2-normalize step amplifies floating-point
noise. In practice this never occurs for natural audio sharing a single provenance; the implementation does
not special-case it.

**Allocation budget per call (after warmup):**
- `embed`: only the output `Embedding`. Mel scratch, FFT scratch, ONNX input backing live on the encoder.
- `embed_batch(N)`: `Vec<Embedding>` of N entries; mel scratch grows once per new max size, reused thereafter.
- `embed_chunked(L, batch=B)`: scratch sized to `B`, reused; per-call cost is the chunk-projection
  `Vec<[f32; 512]>` (~`ceil(L/hop)` × 2 KB).

`warmup()` runs a single `embed` (480 000 samples of silence) which sizes steady-state scratch and triggers
ORT operator specialization.

## 9. Text inference pipeline

### 9.1 Tokenizer

Loaded once at construction via `tokenizers::Tokenizer::from_bytes` / `from_file`. textclap inspects the
tokenizer at construction to cache:

- **`pad_id: i64`** — resolved as:
  ```
  pad_id = tokenizer.get_padding().map(|p| p.pad_id)
       .or_else(|| tokenizer.token_to_id("<pad>"))
       .ok_or_else(|| Error::TokenizerLoad("no pad token; supply tokenizer.json with padding configuration"))?
  ```
  **No literal-1 fallback.** Hardcoding `1` is correct only for RoBERTa; it's wrong for BART (1=BOS), GPT-2
  (1=different token), BERT-base-uncased (1=UNK), etc. Silent miscompute of attention masks (and externalized
  position_ids, where applicable) is not a tradeoff worth making for unverified compatibility.

- **`max_length: usize`** — from the tokenizer's truncation configuration (typically 77 or 512).

**Padding is forced to `BatchLongest`** at construction:
```
tokenizer.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::BatchLongest,
    pad_id,
    pad_token: "<pad>".to_string(),  // or whatever resolved to pad_id
    ...
}))
```
Replaces whatever padding the JSON declared. `Padding::Fixed` is replaced silently; users wanting
fixed-length batches build their own `Tokenizer` outside textclap and apply padding before calling
`embed_batch`.

### 9.2 `TextEncoder` orchestration

> §3.2 backfill: tensor names and dtypes (`input_ids`, `attention_mask`, possibly `position_ids`) confirmed
> by `inspect_onnx.py` before implementation.

§7.3.1 scratch-lifecycle contract applies.

**`embed(text)`:**
1. `text.is_empty()` → `Error::EmptyInput { batch_index: None }`.
2. `tokenizer.encode(text, add_special_tokens=true)` → `Encoding`.
3. Resize encoder-owned `ids: Vec<i64>` and `mask: Vec<i64>` to `T = encoding.len()`; cast u32→i64 and copy.
4. If §3.2 says `position_ids` is externalized: compute it from `mask` and `pad_id` (using the resolved
   `pad_id`, *not* `1`), store in `pos: Vec<i64>`.
5. Bind tensor views, run, drop views.
6. L2-normalize the `[1, 512]` output (or skip if §3.2 says outputs are already unit-norm) → `Embedding`.

**`embed_batch(texts)`:**
1. Empty slice → `Ok(Vec::new())`.
2. For each `texts[i]`: empty → `Error::EmptyInput { batch_index: Some(i) }`.
3. `tokenizer.encode_batch(texts)` → all encodings already padded to `T_max` (BatchLongest applied by
   the tokenizer per §9.1).
4. Resize encoder-owned ids/mask scratch to `[N × T_max]`; copy in-place. If `position_ids` externalized,
   resize and compute.
5. Bind tensor views, run, drop views.
6. Row-by-row L2-normalize (or trust unit-norm output) → `Vec<Embedding>`.

## 10. Error type

Single `thiserror` enum, exposed at crate root, `#[non_exhaustive]` for additive evolution.

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("failed to load ONNX model: {0}")]
    OnnxLoad(#[source] ort::Error),

    #[error("failed to load tokenizer: {message}")]
    TokenizerLoad {
        message: &'static str,
        #[source] source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    #[error("ONNX session schema mismatch: {detail}")]
    SessionSchema { detail: String },

    #[error("failed to read file {path}: {source}")]
    Io { path: PathBuf, #[source] source: std::io::Error },

    #[error("audio input length {got} exceeds maximum {max} samples (10 s @ 48 kHz)")]
    AudioTooLong { got: usize, max: usize },

    #[error("audio input is empty (clip index: {clip_index:?})")]
    EmptyAudio { clip_index: Option<usize> },

    /// Audio sample at first_index is non-finite (NaN, +Inf, -Inf). For batch
    /// input, the offending clip's index is encoded into the upper 32 bits of
    /// first_index (clip_index << 32 | sample_index).
    #[error("audio input contains non-finite sample at first_index = {first_index}")]
    NonFiniteAudio { first_index: u64 },

    #[error("invalid chunking option: {field:?} must be > 0")]
    ChunkingConfig { field: ChunkingField },

    #[error("tokenization failed: {message}")]
    Tokenize {
        message: &'static str,
        #[source] source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    #[error("input text is empty (batch index: {batch_index:?})")]
    EmptyInput { batch_index: Option<usize> },

    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch { expected: usize, got: usize },

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

Error variants that wrap upstream errors use the `Box<dyn Error + Send + Sync + 'static>` pattern
explicitly with the `'static` bound, which `thiserror`'s `#[source]` attribute requires.

## 11. Engineering robustness

### 11.1 ort version coupling

`ort = "=2.0.0-rc.12"` is exact across silero, soundevents, and textclap. Bumping requires a coordinated
change across the trio.

### 11.2 Model file integrity

The README publishes SHA256s (also recorded in `tests/fixtures/MODELS.md` per §3.3) for
`audio_model_quantized.onnx`, `text_model_quantized.onnx`, `tokenizer.json`. Mismatched files produce
undefined results — typically a runtime tensor-shape error or, worse, silent embedding drift.

textclap does not re-verify SHA256s at runtime. Callers serving content over NFS, symlinks, or mmap who
need stricter integrity guarantees should verify before construction.

### 11.3 Quantization variant compatibility

textclap 0.1.0 is verified against the **INT8-quantized** export specifically. Tolerances against the
**Python int8 reference**:

| Variant | Audio embedding tolerance vs Python int8 reference | Notes              |
|---------|----------------------------------------------------|--------------------|
| int8    | < 5e-4 (verified target — see §12.2)               | This release       |
| fp16    | likely < 5e-3                                      | Not verified       |
| fp32    | likely < 1e-2                                      | Not verified       |

The fp32-vs-int8 column entry is intentionally looser than int8-vs-int8 — the comparison is across
quantization regimes, not within. fp32-vs-fp32 (same-precision goldens regenerated against the fp32 export)
would be tighter than int8-vs-int8.

### 11.4 Cold-start latency

`AudioEncoder::warmup`, `TextEncoder::warmup`, and `Clap::warmup` run a single dummy forward each — 480 000
samples of silence for audio, **`"a quick brown fox jumps over the lazy dog repeatedly to size the scratch"`**
for text (a longer string than `"hello world"` so the first real long-text batch doesn't reallocate token
scratch). Sizes scratch for batch size 1; first batched call still grows scratch once (`warmup_for_batch` is
a §14 follow-up).

### 11.5 Test determinism and platform variance

Integration tests construct sessions with `Options::new().with_intra_threads(1)` so reduce-order variability
across thread schedules doesn't introduce flake.

**Hardware and platform variance is intrinsic and non-trivial.** ORT's CPU EP differs across OSes (MLAS on
Linux/Windows, Accelerate on macOS); FMA fusion and vectorization differ between x86 and ARM, and between
AVX-512 and AVX2. Even with `intra_threads=1`, embedding values can differ at the ULP level across runners.
CI tolerances (§12.2) are calibrated to absorb this.

**Bench warmup.** `benches/` Criterion harnesses each call `warmup()` in their setup closure before the
`iter` loop, so first-sample cold-start cost doesn't skew the median.

### 11.6 Model attribution and license compliance

The crate ships with no model files. The README's "Model attribution" section states that **downstream
users redistributing model files take on the attribution responsibilities** of the upstream licenses:
- LAION CLAP weights are **CC-BY 4.0** — attribution required when redistributing.
- Xenova ONNX export is **Apache-2.0**.
- HTSAT and CLAP papers have **citation requirements** (BibTeX in README).

## 12. Testing strategy

### 12.1 Unit tests (per module)

- **`mel.rs`:**
  - Hann window numerical correctness (periodic convention).
  - Filter rows 0, 10, and 32 vs librosa references at `max_abs_diff < 1e-6`. Row 10 specifically catches
    Slaney-vs-HTK construction errors.
  - power_to_dB applied exactly once after the mel filterbank (§8.1.2).
  - Repeat-pad behavior: `len < target` tiles correctly.
  - eps clamp on silence input — no NaN/Inf in the log transform.
  - Output buffer shape matches `T` from `golden_params.json`.
- **`audio.rs`:**
  - Boundary tests: `embed(&[])` → `EmptyAudio`; `embed(&[0.0; 480_001])` → `AudioTooLong`;
    `embed(&[0.0; 480_000])` succeeds.
  - `embed(&[f32::NAN, ...])` → `NonFiniteAudio { first_index: 0 }`. Same for `+Inf`, `-Inf`.
  - `embed_batch` with **uneven-length** clips succeeds (auto-pad).
  - `embed_batch` with one empty clip in the middle → `EmptyAudio { clip_index: Some(i) }`.
  - `embed_batch` with one non-finite clip → `NonFiniteAudio` with encoded clip index.
  - Empty batch slice → empty `Vec`.
  - Chunked windowing offsets and chunk counts, including trailing-chunk-skip rule.
  - **Scratch lifecycle stress test:** small batch then large batch in sequence; results must match
    independently-computed singles within §12.2 tolerance. Reverse order also works.
- **`text.rs`:**
  - `EmptyInput` for empty `&str` and empty string at index `i` in batch.
  - Empty batch slice → empty `Vec`.
  - Tokenizer with `Padding::Fixed` is replaced by `BatchLongest` at construction.
  - Tokenizer with no pad config and no `<pad>` token → `TokenizerLoad`.
  - `from_ort_session` with mismatched session schema → `SessionSchema`.
- **`clap.rs`:**
  - `classify(&samples, &[], k)` → `Ok(vec![])`.
  - `classify(&samples, &labels, 0)` → `Ok(vec![])`.
  - `classify(&samples, &labels, 1000)` → returns all `labels.len()` entries.
  - `classify` returns top-k descending; `classify_all` returns all labels.
  - Stable ordering on tied scores.
  - `LabeledScore::to_owned()` preserves label and score.
- **`options.rs`:**
  - Builder methods round-trip through accessors.
  - Defaults match documented values.
  - `Options::default() == Options::new()`.
- **`Embedding`:**
  - `from_slice_normalizing` always produces unit-norm output for non-zero input.
  - `from_slice_normalizing` rejects all-zero input → `EmbeddingZero`.
  - `try_from_unit_slice` rejects wrong lengths (`EmbeddingDimMismatch`) and non-unit-norm input
    (`EmbeddingNotUnitNorm`) at the 5e-5 norm² budget.
  - `dot ≈ cosine` for unit inputs (within fp32 ULP — this *only* holds for vectors that are exactly
    unit-norm; vectors that pass `try_from_unit_slice` at the 5e-5 budget may see dot/cosine diverge by
    ~5e-6 because cosine re-normalizes at compute time).
  - `is_close(other, tol)` returns true for self-comparison at any tol > 0.
  - `to_vec` and `as_slice` are byte-equal.
  - **No `pub const DIM`** — compile-time test that `Embedding::DIM` doesn't resolve.
  - **Custom `Debug` output does not contain 512 floats.**
  - **No `PartialEq` derived** — compile-time test that `Embedding: PartialEq` doesn't resolve.

### 12.2 Integration test (`tests/clap_integration.rs`)

Gated on `TEXTCLAP_MODELS_DIR` env var (skip with `eprintln!` if unset, do not fail).

Sessions constructed with `intra_threads(1)`.

Assertions:

| Check                                                          | Tolerance (`max_abs_diff`)         |
|----------------------------------------------------------------|-------------------------------------|
| Rust mel features vs `golden_mel.npy`                          | < 1e-4                              |
| Rust audio raw projection vs `golden_audio_proj.npy` (if generated) | < 1e-3 (un-normalized; HTSAT contractive amplification of mel drift) |
| Rust audio embedding vs `golden_audio_emb.npy`                 | < 5e-4 (post-L2; tightens projection drift) |
| Rust text embeddings vs `golden_text_embs.npy`                 | < 1e-5 (text input is integer ids; no upstream drift) |
| `classify_all` discrimination check (see below)                | structural                          |

**Tolerance origin.** Mel drift up to 1e-4 propagates through HTSAT (~14 transformer blocks × ~50 INT8 ops)
with typical 5–50× contractive amplification, putting realistic raw-projection drift at 5e-4 to 5e-3. L2
normalization tightens this — direction is more stable than magnitude. **The 5e-4 audio-embedding budget is
the opening tolerance**; it will be calibrated downward only if real-run measurement supports it. Per-OS
budget tables (Linux MLAS vs macOS Accelerate) may be needed; the spec accepts that and treats per-OS
relaxation as a documented finding, not a failure.

Text embeddings have no upstream drift; 1e-5 catches RoBERTa wiring bugs.

`try_from_unit_slice`'s norm² budget (5e-5) is intentionally tighter than the audio-embedding budget — a
stored unit vector that was unit-norm at write time should *not* drift in fp32 storage; tightening the
round-trip check catches accidental mutation of stored vectors.

**Discrimination check (replaces "ranks #1 exactly"):** `classify_all` is run with the labels
`["a dog barking", "rain", "music", "silence", "door creaking"]` (note: no "speech" — §1.2). The test asserts:
1. `"a dog barking"` ranks in the top 2.
2. `score("a dog barking") - score("music") > 0.05` (irrelevant baseline).

Tie-breaks between dog-bark and acoustically similar labels can swap under int8 quantization; the test
requires the model to *discriminate* (large margin against unrelated labels), which is the property that
matters for retrieval.

### 12.3 Doctests

Every public function on `Clap`, `AudioEncoder`, `TextEncoder`, `Embedding` ships a runnable rustdoc
example. `Embedding` examples are runnable; encoder examples use `# no_run`.

### 12.4 Benches (`benches/`)

Three Criterion benchmarks. Each `setup` closure calls `warmup()` before `iter`. No correctness assertions:
- `bench_mel.rs` — `MelExtractor::extract_into` on a 10 s buffer.
- `bench_audio_encode.rs` — full encode (mel + ONNX) for batch sizes 1, 4, 8.
- `bench_text_encode.rs` — text encode for batch sizes 1, 8, 32.

### 12.5 CI

- **rustfmt** (Linux)
- **clippy** (Linux/macOS/Windows × default features × all features)
- **build + test** matrix (Linux/macOS/Windows × stable Rust)
- **doctest** (Linux, all features)
- **coverage** (tarpaulin → codecov, Linux)
- **integration job** (Linux only) — fetches model files into a runner cache, sets `TEXTCLAP_MODELS_DIR`,
  runs `cargo test --test clap_integration`.

Removed from the original template's CI: WASM/RISC-V/PowerPC64; Miri/ASAN/LSAN/MSAN/TSAN/Loom.

If a runner consistently exceeds a §12.2 budget by a fixed amount, that's a hardware/EP signal, not a Rust
bug — investigated, then either widened with documentation or papered over with a per-OS budget table.

## 13. Migration from current template

textclap is currently the bare `al8n/template-rs` scaffold.

### Replace
- `Cargo.toml` (identity, deps, dev-deps including silero/mediatime/rubato, features, MSRV, version 0.1.0,
  exact `ort` pin, `examples` marked `publish = false`).
- `README.md` — purpose, install, quick-start (`Clap::from_files` → `warmup()` → audio embed + text embed +
  zero-shot classify), model-acquisition note pointing to HuggingFace **with SHA256s** and HF revision pin,
  warning that `tokenizer.json` must come from the same Xenova export (not from
  `laion/clap-htsat-unfused` directly), model-attribution-on-downstream section (§11.6), ort-coupling note,
  license, lancedb integration snippet, deployment note that thread-per-core means each worker calls
  `from_files` **sequentially at startup** with **150–300 MB resident per worker** (measure on your hardware),
  pipeline example showing 48-vs-16 kHz handling, **§1.2 short-segment minimum recommendation (~2 s) and
  speech-domain caveat**.
- `src/lib.rs` (keep crate-level lints; replace body with module decls and re-exports).
- `tests/foo.rs` → delete.
- `benches/foo.rs` → delete.
- `examples/foo.rs` → delete; add `examples/index_and_search.rs` and `examples/vad_to_clap.rs`.
- `CHANGELOG.md` → reset to Keep-a-Changelog stub starting at `[0.1.0]`.

### Keep
- `build.rs` — adds a one-line comment documenting that it sets `cfg(tarpaulin_include)` based on
  `CARGO_CFG_TARGET_OS`/coverage env vars so coverage runs can selectively include/exclude blocks.
- License files (update copyright holder/year).
- `.github/workflows/` skeleton, with deletions per §12.5.

### Add
- `src/error.rs`, `src/options.rs`, `src/mel.rs`, `src/audio.rs`, `src/text.rs`, `src/clap.rs`.
- `tests/fixtures/` contents per §3 / §12.2 (including `README.md` for sample.wav provenance and `MODELS.md`
  for model SHA256s).
- `examples/index_and_search.rs` — pipeline shape, lancedb stubbed.
- `examples/vad_to_clap.rs` — full audio path: 48 kHz source → `rubato` resample to 16 kHz → silero VAD →
  segment time ranges (`mediatime::TimeRange`) → slice the *original* 48 kHz audio at those times →
  `textclap::AudioEncoder::embed`. Demonstrates the bridging concern from §1.1 explicitly.
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

- `serde` round-trip tests for `Embedding`, `Options`, `ChunkingOptions`.
- 1024-dim CLAP variants (`larger_clap_general`, `larger_clap_music`, `clap-htsat-fused`).
- Quantization-tolerance matrix populated for fp16 and fp32 exports (§11.3).
- Optional execution-provider configuration (CUDA, CoreML) layered on top of `from_ort_session`.
- `warmup_for_batch(audio_n: usize, text_n: usize)` if profiling shows the first batched call's scratch
  growth costs measurable latency.
- A second chunking-aggregation strategy (max, attention pooling, mean-of-logits) if a real CLAP use case
  demonstrates value. Adding it brings back the `Aggregation` enum + `ChunkingOptions::with_aggregation`.
- A `pad_mode: silence` option in `ChunkingOptions` to replace repeat-pad with zero-pad for short clips,
  addressing §1.2's periodicity-artifact concern.
- An optional **strict LAION-reference mode** for `embed_chunked` that does single-window rand_trunc with a
  caller-provided RNG seed.
- A doctest on `Embedding::cosine` showing the lancedb round-trip specifically.
- `tracing` feature for service-tier observability.
- `try_reserve_exact` on scratch resizes to surface OOM as `Error::ScratchAlloc` instead of panic.
- `Options::with_truncation_warn_threshold(usize)` to log when text inputs hit the silent truncation cap.
- In-flight cancellation: not feasible with the synchronous `session.run()` API; would require ORT's
  async API or a coarse drop-the-encoder pattern.
