# textclap — CLAP Inference Library Design

**Status:** Draft (revision 9 — text encoder is query-time only, not part of indexing)
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

textclap exposes two encoders that work as the **indexing** and **query** halves of an audio-search system.
The model treats them as a contrastive pair — audio embeddings and text embeddings live in the same 512-dim
space — but **they are used at different times in the pipeline, not as parallel indexing paths**.

**Indexing path (write side, runs continuously while audio is captured):**

```text
audio frames (native rate, e.g. 44.1 / 48 kHz)
  → decoder → resample to 48 kHz → buffer 10 s
  → AudioEncoder::embed → 512-dim audio embedding
  → lancedb.write { audio_embedding, ts_start, ts_end, ... }
```

The audio encoder runs every 10 s of input. There is no VAD — CLAP is a general-audio model trained on
AudioSet (speech, music, ambient sounds, alarms, animals, traffic, all of it), so the right input is the
whole audio stream chunked at the model's training window. Pre-segmenting to "speech only" or "non-silence"
would discard exactly the non-speech context CLAP excels at recognizing.

**Query path (read side, on demand when a user submits a search):**

```text
user query text (e.g. "dog barking near a door")
  → TextEncoder::embed → 512-dim text embedding
  → lancedb cosine-similarity search against the audio_embedding column
  → ranked audio windows
```

The text encoder runs **once per search query**. It does *not* sit in the indexing path; it does *not*
embed Whisper transcripts or any STT output. Its sole job is to convert a free-form text query into a vector
in the same 512-dim space as the indexed audio embeddings, so cosine similarity finds matching audio.

**Out of textclap's scope (but worth flagging because it's adjacent to a real deployment).** A user pipeline
may run silero VAD + Whisper STT on the same source audio to produce transcripts and store them in a
*separate* lancedb column for caption display, BM25 / FTS keyword search, or other text-based recall. That
branch runs in parallel with textclap and does **not** route through CLAP's text encoder. It needs nothing
from textclap.

**Sample-rate handling is the caller's problem.** CLAP requires 48 kHz mono `f32` PCM; the decoder
resamples to that. textclap does not resample. The 16 kHz path that silero needs is independent and runs
out-of-band if the user pipeline does any STT.

**Asymmetric encoder lifetime in deployment.** Because the audio encoder runs once per 10 s of input and
the text encoder runs once per user search, indexing workers and query workers face very different load
profiles. The API already supports loading only one encoder per process — `AudioEncoder::from_files(...)`
and `TextEncoder::from_files(...)` are independent. An indexing-worker process can save the 121 MB text-model
load by skipping `Clap::from_files` and constructing only `AudioEncoder`; a low-rate query process can
similarly skip the audio model. Use `Clap::from_files` only when one process needs both. README documents
the deployment pattern.

`examples/audio_window_to_clap.rs` demonstrates the indexing path (decoder → 48 kHz resample → 10 s buffer
→ `AudioEncoder::embed` → stubbed lancedb write). `examples/index_and_search.rs` shows the read side
(text query → `TextEncoder::embed` → stubbed lancedb search) plus the indexing side at a glance.

### 1.2 What CLAP recognizes — and doesn't

**Domain-of-training.** CLAP-HTSAT-unfused was trained on AudioSet plus general-audio captions: it
discriminates speech-vs-music-vs-ambient, recognizes specific sound categories (dog barks, alarms, traffic,
machinery, water, applause), and tracks coarse acoustic scene attributes. It is suited to descriptive text
queries like *"rain on a metal roof,"* *"applause in a stadium,"* *"engine starting,"* *"speech with a
loud crowd in the background."*

**It is NOT suited to within-speech content queries** like *"the meeting where Alice mentioned Q3
revenue."* Conversations cluster tightly in CLAP-audio space because their acoustic features (speech-band
energy, voiced/unvoiced patterns, prosody) are similar regardless of what was said. For that kind of recall,
the user's pipeline indexes Whisper transcripts as plain text in a separate column (BM25 / FTS / vector search
with a separate text model); textclap is not involved.

**Short-clip artifacts (off the recommended path).** If callers feed `embed()` with clips shorter than 10 s
— e.g. per-VAD-segment for a different use case — textclap's repeat-pad fills the window by tiling, which
produces real periodicity artifacts in the mel spectrogram (e.g. a 1 s clip tiled to 10 s creates a 1 Hz
repetition pattern and 10 identical positional patches in HTSAT's input). Recommended minimum for that
non-default flow is ~2.5 s of original content (matching the §7.8 trailing-chunk-skip threshold of
`window/4`). For the recommended fixed-window flow (§1.1), this never triggers — every input is exactly
10 s. A `pad_mode: silence` alternative for short clips is a §14 follow-up.

## 2. Non-goals

- **Audio resampling.** Input must be 48 kHz mono `f32` PCM. Caller's responsibility.
- **Streaming inference.** CLAP isn't streaming.
- **Vector store integration.** Embeddings are emitted; storage and ANN search live in the caller.
- **Model bundling or download helpers.** No models in the crate, no network at build or runtime.
- **Async / runtime ownership.** Synchronous library; no in-flight cancellation in 0.1.0 (see §14 — feasible
  via ORT's `RunOptions::terminate()`, deferred for now).
- **Multi-variant CLAP support in 0.1.0.** Only the 512-dim `Xenova/clap-htsat-unfused` export is verified
  (against INT8). The public API does not lock to this dimension (§7.5).
- **NaN/Inf-safe arithmetic.** Non-finite samples are detected and rejected up front (§7.3, §10).
- **Cross-tool embedding interop for chunked audio.** textclap's `embed_chunked` is a textclap-specific
  convention, not LAION-reference compatible (§7.3, §8.2). Single-window `embed` (≤10 s) does match the
  LAION reference within the verified tolerance.
- **Thread / EP tuning knobs on `Options`.** Sibling convention (silero, soundevents) deliberately omits
  these — deployment-specific runtime policy is configured one layer up by building an `ort::Session`
  directly and passing it via `from_ort_session`. textclap follows.

## 3. Pre-implementation prerequisites

Several parameters in the audio preprocessing pipeline cannot be safely guessed; they must be measured against
the actual model files before any Rust is written.

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
   - Window function: periodic vs symmetric (read by computing the actual array and inspecting it).
   - **Frame count `T`** — the time dimension produced by the extractor on a 480_000-sample input.
     With `n_fft=1024`, `hop=480`: `center=False` ⇒ 998 frames; `center=True` ⇒ 1001. The HF
     preprocessor_config claims `max_frames=1000`. The actual value, the centering flag, and the pad mode
     all go into `golden_params.json`. The spec stops asserting "1000" anywhere; mel.rs uses `T` from
     `golden_params.json`.
   - `padding` mode, `truncation` mode (the latter intentionally diverges in Rust — see §8.1).
   - `frequency_min` / `frequency_max`.
3. Runs the extractor; saves resulting `[64, T]` mel features to `golden_mel.npy`.
4. **Audio model golden.** Loads `audio_model_quantized.onnx` via `onnxruntime.InferenceSession`; runs it
   on the golden mel features. Computes the L2-normalized embedding using the *exact* formula
   ```python
   x = raw_output.astype(np.float32)
   norm = np.linalg.norm(x).astype(np.float32)
   embedding = (x / norm).astype(np.float32)
   ```
   (deliberately *not* `torch.nn.functional.normalize`, which can differ in summation order) and saves to
   `golden_audio_emb.npy`. Raw projection goldens are NOT generated — see §12.2 for why (the projection
   tolerance arithmetic is inverted vs the embedding check, and the projection check is redundant for
   single-window inference; chunking has no Python reference at all).
5. **Text model golden.** Loads `text_model_quantized.onnx` and `tokenizer.json`. Runs five fixed labels
   (`["a dog barking", "rain", "music", "silence", "door creaking"]` — note "speech" is *deliberately not*
   in the label set, since CLAP does not discriminate well within speech, §1.2). Saves L2-normalized
   embeddings `[5, 512]` to `golden_text_embs.npy`.

**Why goldens come from the int8 ONNX, not the fp32 PyTorch path:** the Rust crate runs the int8 ONNX.
Goldens must run the same int8 ONNX in Python — otherwise the test tolerance has to absorb both quantization
drift *and* implementation differences, indistinguishably. Mel goldens still come from `ClapFeatureExtractor`
(fp32 NumPy).

**To verify a non-int8 setup,** users regenerate goldens by pointing `regen_golden.py` at the alternate ONNX
file. Mel goldens don't change; audio/text goldens are recomputed; tolerances may need to widen (§11.3).

### 3.2 ONNX graph IO inspection — and functional verification

`tests/fixtures/inspect_onnx.py` does both static graph inspection and a functional end-to-end check.

**Static inspection.** For each ONNX file, dump `graph.input` / `graph.output` (name, dtype, shape with
dynamic dims marked) and the first/last 20 graph nodes into `tests/fixtures/golden_onnx_io.json`. From this
file the spec answers:

- **Audio input shape:** `[batch, 1, 64, T]` *vs* `[batch, 64, T]` (channel dim present?).
- **Audio output L2-normalize?** Examine the last 5 graph nodes for an `LpNormalization` op (axis=-1, p=2)
  or the equivalent `ReduceL2` + `Div` pattern. Record `audio_output_is_unit_norm: true|false`. This drives
  which aggregation path `embed_chunked` uses (§8.2).
- **Text input names and dtypes:** `input_ids: [batch, T] i64`, `attention_mask: [batch, T] i64`, **plus**
  whether `position_ids` appears as a third input. Implementation matches whatever the graph requires.
- **Text output L2-normalize?** Same check as audio.
- **Text truncation max_length** — read from the `tokenizers` Python binding's `tokenizer.truncation`
  property. Some Xenova exports keep RoBERTa's default 512 instead of CLAP's expected 77.
- **Audio output / text output names and shapes** (`[batch, 512]` expected; verify names).

**Functional verification of HTSAT input normalization.** Static inspection alone is insufficient. The script
runs both transformations and picks the lower-error one:

```python
features = extractor(audio, sampling_rate=48000, return_tensors="pt")  # returns BatchFeature dict
pt_emb = pt_model.get_audio_features(**features)                       # fp32 reference, normalized
pt_emb = pt_emb / pt_emb.norm()

# Try BOTH input transforms; pick whichever agrees better.
for transform_name, transform_fn in [("none", lambda x: x), ("global_mean_std", apply_audioset_norm)]:
    ort_input = transform_fn(features["input_features"].numpy())
    ort_raw = audio_session.run(None, {"input_features": ort_input})[0]
    ort_emb = ort_raw / np.linalg.norm(ort_raw)
    drift = np.max(np.abs(pt_emb - ort_emb))
    record(transform_name, drift)

# Decision rule:
#   < 5e-3       → pass; record this transform
#   5e-3 .. 2e-2 → yellow zone; pick whichever transform produced less drift; warn in stdout
#   ≥ 2e-2       → reject (both transforms); something else is wrong, investigate before continuing
```

The script records the chosen transform in `golden_params.json` as
`htsat_input_normalization: { type: "none" | "global_mean_std", mean: f32, std: f32 }` and Rust's mel.rs
applies it (or doesn't) accordingly. The 5e-3 / 2e-2 thresholds are calibrated against the typical
quantization drift between fp32 PyTorch and int8 ONNX (~1e-3 to 5e-3) — a real normalization mismatch
produces 3e-3 to 3e-2 drift, overlapping the noise floor enough that a single-tier check at 1e-2 (rev-6's
choice) couldn't tell them apart.

Probing both transforms (rather than stopping at the first that "passes") catches the case where `none`
happens to land at 4.5e-3 by accident while `global_mean_std` would have given 1e-3.

### 3.3 Model SHA256 acquisition

Before §3.1 / §3.2, the maintainer downloads the three model artifacts from a pinned Hugging Face revision
(commit hash recorded in `tests/fixtures/MODELS.md`) and computes:

```
shasum -a 256 audio_model_quantized.onnx text_model_quantized.onnx tokenizer.json
```

The SHA256s, the HF revision hash, and the URL are recorded in `tests/fixtures/MODELS.md` and reproduced in
the README. Mismatched SHA256s in user setups produce undefined results (loud README warning).

### 3.4 Spec-update commit sequence

§3.1–§3.3 produce an explicit multi-commit sequence:

0. **Models commit:** `tests/fixtures/MODELS.md` (SHA256s + HF revision pin + URLs).
1. **Scripts commit:** `regen_golden.py` and `inspect_onnx.py` source.
2. **Generated-fixtures commit:** `golden_params.json`, `golden_onnx_io.json`, `golden_*.npy`,
   `tests/fixtures/README.md`.
3. **Spec-update commit:** §8.1 mel parameter table, §8.2 / §9.2 tensor names and shapes, §7.4
   attention-mask / position_ids description, §12.2 tolerance table — all replaced with values consistent
   with the generated fixtures.
4. **Rust implementation commits.**

Steps 0–3 must land before any Rust source.

## 4. Crate layout

```
textclap/
├── Cargo.toml                       # see §5 for [lints.rust], [package.metadata.docs.rs], include shape
├── build.rs                         # emits cargo:rustc-cfg=tarpaulin when CARGO_FEATURE_TARPAULIN /
│                                    # CARGO_TARPAULIN / CARGO_CFG_TARPAULIN is set; copied verbatim
│                                    # from the sibling crates
├── README.md
├── CHANGELOG.md
├── LICENSE-MIT / LICENSE-APACHE / COPYRIGHT
├── src/
│   ├── lib.rs                       # crate-level docs + module decls + the re-exports below
│   ├── error.rs                     # Error enum (thiserror)
│   ├── options.rs                   # Options, ChunkingOptions
│   ├── mel.rs                       # MelExtractor: STFT → mel filterbank → log-mel
│   ├── audio.rs                     # AudioEncoder
│   ├── text.rs                      # TextEncoder
│   └── clap.rs                      # Clap, Embedding, LabeledScore, LabeledScoreOwned
├── tests/
│   ├── clap_integration.rs          # gated on TEXTCLAP_MODELS_DIR env var
│   └── fixtures/
│       ├── README.md                # provenance + license attribution for sample.wav
│       ├── MODELS.md                # SHA256s + HF revision + download URLs
│       ├── sample.wav               # public-domain dog-bark WAV, ≤10 s, 48 kHz mono
│       ├── golden_params.json
│       ├── golden_onnx_io.json
│       ├── golden_mel.npy
│       ├── golden_audio_emb.npy
│       ├── golden_text_embs.npy
│       ├── filterbank_row_0.npy     # librosa-precomputed mel filterbank row 0
│       ├── filterbank_row_10.npy    # row near 1 kHz (Slaney inflection — discriminates Slaney vs HTK)
│       ├── filterbank_row_32.npy    # mid-band reference row
│       ├── regen_golden.py
│       └── inspect_onnx.py
├── benches/
│   ├── bench_mel.rs
│   ├── bench_audio_encode.rs
│   └── bench_text_encode.rs
├── examples/
│   ├── index_and_search.rs          # end-to-end pipeline shape (lancedb stubbed)
│   └── audio_window_to_clap.rs      # decoder → 48 kHz resample → 10 s window → AudioEncoder::embed
└── docs/superpowers/specs/
```

**`lib.rs` re-exports** (enumerated to match silero/soundevents convention):

```rust
pub use crate::audio::AudioEncoder;
pub use crate::clap::{Clap, Embedding, LabeledScore, LabeledScoreOwned};
pub use crate::error::{Error, Result};
pub use crate::options::{ChunkingOptions, GraphOptimizationLevel, Options};
pub use crate::text::TextEncoder;
```

**Test infrastructure divergence from siblings:** silero ships a 1.8 MB ONNX model in-tree under
`tests/fixtures/`; soundevents has no separate integration directory at all. textclap's models are
33+121 MB and cannot reasonably be committed; integration tests are gated on `TEXTCLAP_MODELS_DIR` env var
(§12.2). This is the only test-infra divergence from sibling convention and is justified by size.

## 5. Dependencies

### Default

| Crate         | Version          | Purpose                                          |
|---------------|------------------|--------------------------------------------------|
| `ort`         | `2.0.0-rc.12`    | ONNX Runtime Rust bindings (matches sibling caret-pin) |
| `rustfft`     | `^6`             | Real-input STFT for mel extraction               |
| `tokenizers`  | `^0.20`          | HF tokenizer.json loader (RoBERTa BPE)           |
| `thiserror`   | `^2`             | Error derives                                    |

`ort = "2.0.0-rc.12"` is a caret pin matching silero and soundevents' `Cargo.toml` exactly. Cross-RC bumps
require a coordinated change across the sibling crates.

### Optional features

- **`serde`** — `Serialize` / `Deserialize` derives on `Options`, `ChunkingOptions`,
  `LabeledScore`, `LabeledScoreOwned`, and `Embedding` (sequence form, length 512).

### Dev-dependencies

| Crate        | Version | Used by                                                    |
|--------------|---------|------------------------------------------------------------|
| `criterion`  | `^0.5`  | `benches/`                                                 |
| `rubato`     | `^0.16` | `examples/audio_window_to_clap.rs` (44.1 → 48 kHz resample) |
| `npyz`       | `^0.8`  | `tests/clap_integration.rs` (.npy reader)                  |
| `hound`      | `^3`    | `tests/clap_integration.rs` (WAV reader)                   |

`examples/` are marked `publish = false` so the published crate's `cargo build --examples` does not need
non-published path-deps. `silero` and `mediatime` are not direct dev-deps in 0.1.0 — the kept example shows
only the audio path (no VAD), and the STT branch (which would use silero / Whisper) is a §14 follow-up
example.

### Cargo.toml shape (matching siblings)

In addition to deps and features, the manifest carries:

- **`include = [...]`** — a whitelist excluding `tests/fixtures/` (~MBs of `.npy` and the WAV) from the
  published crate.
- **`[lints.rust]`** block: `rust_2018_idioms`, `single_use_lifetimes`,
  `unexpected_cfgs = { level = "warn", check-cfg = ['cfg(tarpaulin)', 'cfg(all_tests)'] }`.
- **`[package.metadata.docs.rs]`**: `all-features = true` and `rustdoc-args = ["--cfg", "docsrs"]`.

These three blocks are copied from silero/soundevents `Cargo.toml` directly (no per-crate divergence).

### Excluded (deliberate)

- No `tokio`, no async — synchronous library.
- No `download` feature — no network, no `ureq`/`sha2`/`reqwest`.
- No model bundling — no `bundled` feature.
- No BLAS / `ndarray` — the mel filterbank multiply is small enough to write by hand.
- No `tracing` — observability is a §14 follow-up.
- No `num_cpus` — `Options` does not expose thread counts (§7.7); ORT defaults are inherited or overridden
  via `from_ort_session`.

## 6. Toolchain & metadata

- **Rust edition:** 2024
- **MSRV:** 1.85
- **License:** MIT OR Apache-2.0
- **Crate-level lints:** `#![deny(missing_docs)]`, `#![forbid(unsafe_code)]`
- **Initial version:** `0.1.0`

## 7. Public API

All public structs use private fields and accessor methods. Builder-style `with_*` returns `Self` by value;
`set_*` mirrors take `&mut self` and return `&mut Self`; getters are `pub const fn` and `#[inline(always)]`
under non-coverage builds. Field-less unit enums are public-as-data. **No public signature exposes
`[f32; 512]`** — keeps the door open for swapping in larger CLAP variants.

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

pub type Result<T, E = Error> = std::result::Result<T, E>;
```

`AudioEncoder` and `TextEncoder` own their `ort::Session` and (for audio) the mel-extractor state by value.
**The deployment model is thread-per-core**: each worker thread loads its own encoder once at startup.
Memory cost is **150–300 MB resident per worker** for both encoders together (33 MB int8 audio model +
121 MB int8 text model on disk, plus ORT working buffers and weight layout — measure on your hardware).
README recommends **sequential** worker construction at startup to avoid transient 2× peak memory during
ORT weight reformatting.

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
L2-normalized audio and text embeddings; range ≈ `[-1, 1]`, higher more relevant. Order is descending;
tie-break is input-label order (stable).

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
    ///   - any non-finite     → Error::NonFiniteAudio { clip_index: None, sample_index } (caught up front)
    ///   - 0 < len < 480_000  → repeat-padded to 10 s by the mel extractor
    ///   - len == 480_000     → passes through without padding/truncation
    pub fn embed(&mut self, samples: &[f32]) -> Result<Embedding>;

    /// Batch of clips of any lengths in 0 < len ≤ 480_000. Each clip is
    /// repeat-padded to 10 s independently — no equal-length requirement.
    /// Empty *slice* returns Ok(Vec::new()). Any clip with len == 0 returns
    /// Error::EmptyAudio with its index. Any non-finite sample returns
    /// Error::NonFiniteAudio with both clip_index and sample_index.
    ///
    /// **Performance note.** Compute scales with N × full-window regardless
    /// of input length — 8 clips of 0.3 s cost the same as 8 clips of 10 s.
    /// Group very short clips and pre-concatenate for embed_chunked if
    /// per-clip latency is dominated by the repeat-pad waste.
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
    /// the choice is made at construction.
    /// Embeddings produced by either variant are textclap-specific.
    /// Cross-tool retrieval requires both sides to use textclap.
    pub fn embed_chunked(&mut self, samples: &[f32], opts: &ChunkingOptions)
        -> Result<Embedding>;

    pub fn warmup(&mut self) -> Result<()>;
}
```

**Concurrency model:** `AudioEncoder` is `Send` but **not `Sync`**. Each worker thread owns its own
`AudioEncoder`. Encoder-owned mel feature buffer, FFT scratch, and ONNX input tensor backing are growable
`Vec<f32>`s; sized on the first call (or amortized via `warmup()`); reused thereafter — the hot path
performs no heap allocation after warmup.

#### 7.3.1 Scratch lifecycle contract (UB prevention)

`#![forbid(unsafe_code)]` blocks `unsafe` blocks in textclap, but **`ort 2.x`'s `TensorRef::from_array_view`
constructs views that the ORT C++ runtime borrows during `session.run()`**. If the underlying scratch
buffer is reallocated after the tensor view is bound and before `session.run()` returns, the C++ side
accesses freed memory — undefined behavior through the FFI boundary.

The implementation adopts silero's production pattern from `silero/src/session.rs` (the same hazard exists
there; silero's working code documents the working solution by example):

1. **`scratch.clear()`** — drops the previous contents but preserves capacity.
2. **`scratch.reserve(required_size)`** — grows capacity if needed; this is the *only* point at which
   reallocation can occur. The caller still owns the `Vec`; nothing borrows it yet.
3. **`scratch.extend_from_slice(...)`** *or* **`scratch.resize(required_size, 0.0)`** — fills to the final
   length. Capacity is now committed.
4. **`TensorRef::from_array_view(...as_slice()...)`** — borrows `&[f32]` from the `Vec`. From this point on,
   the borrow checker prevents *any* mutation of the `Vec` until the borrow ends; reallocation is
   structurally impossible until step 6.
5. **`session.run()`**.
6. **Tensor views drop** at end of scope. The borrow on `&[f32]` ends; the `Vec` is mutable again.

`extend_from_slice` is preferred over `resize_with` for "fill from sources" cases (it copies efficiently and
matches silero); `resize` with a fill value is appropriate for genuine zero-padding (silero uses this in
session.rs:319 and 361). The borrow checker — not unit tests, not documentation — is the structural
protection. Each encoder method ships a unit test that runs a small batch then a larger batch in sequence,
verifying that growth-then-reuse paths produce embeddings within §12.2 tolerance, but the test is a
liveness check, not the safety guarantee.

After `session.run()`, every output tensor's shape is validated against the expected shape via a
`validate_shape(tensor: &'static str, expected: &[i64], actual: &[i64])` helper (mirrors silero
`validate_shape` at `session.rs`). Mismatch → `Error::UnexpectedTensorShape`. Cost is negligible; catches
ORT version skew and model artifact swaps at the friendly boundary.

### 7.4 `TextEncoder`

```rust
impl TextEncoder {
    pub fn from_files<P: AsRef<Path>>(onnx_path: P, tokenizer_json_path: P, opts: Options) -> Result<Self>;
    pub fn from_memory(onnx_bytes: &[u8], tokenizer_json_bytes: &[u8], opts: Options) -> Result<Self>;

    /// Wraps a pre-built ORT session and Tokenizer. Both are validated:
    /// - session schema vs golden_onnx_io.json (Error::SessionSchema on mismatch);
    /// - tokenizer must NOT have Padding::Fixed (Error::TokenizerLoad). The
    ///   caller built this Tokenizer deliberately, so a Fixed padding setting
    ///   is treated as a user bug rather than silently rewritten.
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

**Padding::Fixed handling is asymmetric across constructors:**
- `from_files` / `from_memory`: `Padding::Fixed` in the loaded `tokenizer.json` is **silently rewritten** to
  `BatchLongest`. JSON authors rarely set `Fixed` intentionally; the silent rewrite avoids a ~6× perf
  regression on typical Whisper-transcript inputs without changing semantics (the model's attention mask
  handles variable lengths regardless).
- `from_ort_session`: `Padding::Fixed` is **rejected** with `Error::TokenizerLoad`. The caller built this
  `Tokenizer` deliberately; surface the surprise rather than mutate it. Callers wanting fixed-length batches
  pre-pad upstream.

**Concurrency model:** `Send` but **not `Sync`**. §7.3.1 scratch lifecycle contract applies identically.

**Tokenizer truncation max_length is taken from `tokenizer.json` at construction** — no `max_length` knob is
exposed. The actual value (typically 77 for CLAP, sometimes 512 for RoBERTa default) is recorded in
`golden_params.json` by §3.1. Long inputs are silently truncated; this is documented loudly in
`embed`/`embed_batch` rustdoc.

**Position-ids:** RoBERTa computes positions as `pad_id + 1 + cumsum(non_pad_mask)`. This is *typically*
inlined into Xenova's ONNX export. If §3.2 finds `position_ids` as an externalized input, the encoder
computes it explicitly using the resolved `pad_id` (§9.1) — *not* the literal `1` — and feeds it as a third
tensor.

### 7.5 `Embedding`

```rust
impl Embedding {
    pub const fn dim(&self) -> usize;            // 512 for 0.1.0; runtime-queryable, future-proof

    // Borrow-only access — supports append_slice into Arrow's MutableBuffer.
    pub fn as_slice(&self) -> &[f32];

    // Owned conversion.
    pub fn to_vec(&self) -> Vec<f32>;

    /// Reconstruct from a stored unit vector. Validates length AND norm
    /// (release-mode check: `(norm² − 1).abs() < 5e-5`). The numerical
    /// budget reflects per-component drift propagating to norm² via
    /// `Δ(norm²) ≈ 2 · √(dim · Δ_max_per_component²)` — a 5e-5 norm² check
    /// corresponds to ~5e-6 max-per-component drift, tighter than the
    /// integration-test embedding budget so storage round-trips don't trigger
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

    /// Approximate equality test — raw float drift, max-abs metric.
    /// Use for unit tests checking implementation determinism.
    /// Recommended tol: see §12.2 tolerance table.
    pub fn is_close(&self, other: &Embedding, tol: f32) -> bool;

    /// Approximate equality test — semantic (cosine) metric.
    /// Returns true if `1 - self.cosine(other) < tol`.
    /// Use for tests asserting semantic equivalence (e.g. round-trip
    /// through serialization). For unit vectors of dim 512, a max-abs
    /// drift of ε corresponds to a worst-case rotation of arcsin(ε·√dim);
    /// is_close_cosine is the geometrically meaningful metric.
    pub fn is_close_cosine(&self, other: &Embedding, tol: f32) -> bool;
}

impl AsRef<[f32]> for Embedding;          // delegates to as_slice()

// Custom Debug — does NOT dump 512 floats.
//   Embedding { dim: 512, head: [0.0123, -0.0456, 0.0789, ..] }
impl fmt::Debug for Embedding;

// derives: Clone.
// NO PartialEq, Eq, or Hash. Bit-pattern equality across runs/threads/OSes/
// hardware is unreliable for f32 outputs of ML models. Use is_close /
// is_close_cosine for tests; use the ANN index for similarity in production.
#[cfg(feature = "serde")] // serializes as a sequence of 512 f32 values.
```

**No public method exposes a fixed-size array.** Internal storage is `[f32; 512]` for 0.1.0 (cheap,
stack-friendly); the API is dimension-agnostic at signature level. **There is no `pub const DIM`** — code
calls `dim()` or `as_slice().len()`.

**Invariant:** every `Embedding` returned by this crate is L2-normalized to unit length within fp32 ULP.
The un-normalized projections used internally by `embed_chunked` never escape to the public API.

### 7.6 `LabeledScore` and `LabeledScoreOwned`

```rust
impl<'a> LabeledScore<'a> {
    pub const fn label(&self) -> &'a str;
    pub const fn score(&self) -> f32;
    pub fn to_owned(&self) -> LabeledScoreOwned;
}

impl LabeledScoreOwned {
    pub fn label(&self) -> &str;
    pub const fn score(&self) -> f32;
    pub fn into_label(self) -> String;
}
```

### 7.7 `Options`

```rust
impl Options {
    pub fn new() -> Self;                                                 // == Self::default()

    pub fn with_graph_optimization_level(self, level: GraphOptimizationLevel) -> Self;
    pub fn set_graph_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self;
    pub const fn graph_optimization_level(&self) -> GraphOptimizationLevel;
}
```

`GraphOptimizationLevel` is re-exported from `ort`. **There is no `with_intra_threads` knob.** Sibling
convention (silero `options.rs:128–145`, soundevents): "deployment-specific runtime policy such as
intra_threads / inter_threads should normally be configured one layer up, then passed down via
`Session::from_ort_session`." Callers needing thread tuning, EP selection (CUDA, CoreML), or other ORT
runtime knobs build their own `ort::Session` via `ort::session::Session::builder()...build()` and pass it
to `from_ort_session`.

### 7.8 `ChunkingOptions`

```rust
impl ChunkingOptions {
    pub fn new() -> Self;        // window=480_000, hop=480_000, batch_size=8

    pub fn with_window_samples(self, n: usize) -> Self;
    pub fn set_window_samples(&mut self, n: usize) -> &mut Self;
    pub const fn window_samples(&self) -> usize;

    pub fn with_hop_samples(self, n: usize) -> Self;
    pub fn set_hop_samples(&mut self, n: usize) -> &mut Self;
    pub const fn hop_samples(&self) -> usize;

    pub fn with_batch_size(self, n: usize) -> Self;
    pub fn set_batch_size(&mut self, n: usize) -> &mut Self;
    pub const fn batch_size(&self) -> usize;
}
```

Aggregation strategy is fixed in 0.1.0 (centroid or spherical-mean, chosen at construction per §3.2);
no enum or setter, see §14.

Validation runs at use, not at build: `embed_chunked` returns
`Error::ChunkingConfig { window_samples, hop_samples, batch_size }` if **any** of these holds:
- `window_samples == 0`
- `hop_samples == 0`
- `batch_size == 0`
- `hop_samples > window_samples` (rejected because gapped chunking is rarely intentional and complicates the
  trailing-chunk-skip rule).

The error carries all three values so the caller can see the full configuration that was rejected (matches
soundevents convention).

Trailing chunks shorter than `window_samples / 4` are **skipped** to avoid dragging the centroid into noise
from the trailing repeat-pad. With `hop_samples ≤ window_samples` enforced at validation, this rule
applies cleanly to the final position only: `if min(window_samples, samples_remaining_at_offset) < window_samples / 4, skip`.

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
| Filterbank norm      | **Slaney**                                        |
| Frequency range      | 50 – 14 000 Hz                                    |
| Power spectrum       | `|X|²`                                            |
| Mel→dB transform     | **`10 · log10(max(amin, x))` with `amin = 1e-10`, `ref = 1.0`, `top_db = TBD by §3.1`; applied exactly once after the mel filterbank** |
| Padding mode         | repeatpad                                         |
| Truncation mode      | head (deterministic; intentionally differs from HF rand_trunc) |
| HTSAT input norm     | **TBD by §3.2 functional check**                  |

State allocated once in `new()`, owned by the `MelExtractor`:
- Hann window (`Vec<f32>`, len 1024, periodic convention).
- Mel filterbank (`Vec<f32>`, len 64 × 513).
- `RealFftPlanner<f32>` instance.

The ONNX input tensor `[N, 1, 64, T]` is built as a *view* over the mel feature scratch — the channel dim is
added at tensor-construction time with no data movement.

#### 8.1.1 Filterbank-correctness unit test

`mel.rs` ships unit tests that compare filter rows 0, 10, and 32 against pre-computed reference rows
(`librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64, fmin=50, fmax=14000, htk=False, norm='slaney')`)
committed as `tests/fixtures/filterbank_row_0.npy` etc. Tolerance: `max_abs_diff < 1e-6`. **Row 10 lands
near the 1 kHz Slaney inflection** — necessary to discriminate Slaney from HTK construction (rows 0 and 32
alone do not).

#### 8.1.2 Power-to-dB single-application test

Separate unit test asserts that `MelExtractor::extract_into` differs visibly from a hand-written "apply
power_to_dB twice" reference. Confirms the floor is applied exactly once after the filterbank.

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

Single-clip operations call this with `clips = &[samples]`. One path; no perf-profile ambiguity.

§7.3.1 scratch-lifecycle contract applies to every `session.run()` here. The implementation pattern, mirroring
silero:

```text
self.mel_scratch.clear();
self.mel_scratch.reserve(N * 64 * T);
self.mel_scratch.resize(N * 64 * T, 0.0);              // single grow + zero-fill
for (i, clip) in clips.iter().enumerate() {
    self.mel.extract_into(clip, &mut self.mel_scratch[i*64*T .. (i+1)*64*T]);
}
let input = TensorRef::from_array_view(self.mel_scratch.as_slice(), &[N, 1, 64, T])?;
let outputs = self.session.run(ort::inputs!["input_features" => input])?;
let raw = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;
validate_shape("audio_output", &[N, 512], raw.shape())?;
out.clear();
out.reserve(N);
for n in 0..N { out.push(*raw.slice(s![n, ..]).as_array_ref()); }
// tensor views drop here at end of scope
```

Per-call shape validation runs after `try_extract_tensor` (matches silero `session.rs:185, 188-192`).

**`embed(samples)`:**
1. `samples.is_empty()` → `Error::EmptyAudio { clip_index: None }`.
2. `samples.len() > 480_000` → `Error::AudioTooLong { got, max: 480_000 }`.
3. **Finiteness scan:** SIMD pass over `samples` for the first non-finite value. On hit:
   `Error::NonFiniteAudio { clip_index: None, sample_index }`.
4. Call `embed_projections_batched(&[samples], &mut self.proj_scratch)`.
5. Take `self.proj_scratch[0]`; L2-normalize → `Embedding` (or, if §3.2 says outputs are already unit-norm,
   skip the normalize and construct via the trusted path; debug_assert unit-norm).

**`embed_batch(clips)`:**
1. Empty slice → `Ok(Vec::new())`.
2. For each clip `i`: empty → `Error::EmptyAudio { clip_index: Some(i) }`; too-long → `AudioTooLong`;
   finiteness scan → `Error::NonFiniteAudio { clip_index: Some(i), sample_index }`.
3. Call `embed_projections_batched(clips, &mut self.proj_scratch)`.
4. Row-by-row L2-normalize → `Vec<Embedding>`.

**`embed_chunked(samples, opts)`:**
1. `samples.is_empty()` → `Error::EmptyAudio { clip_index: None }`.
2. Validate `opts` per §7.8 → otherwise `Error::ChunkingConfig { window_samples, hop_samples, batch_size }`.
3. Finiteness scan over `samples`.
4. Compute chunk offsets `0, hop, 2·hop, …` while `offset < samples.len()`. Trailing chunks shorter than
   `window_samples / 4` are skipped (§7.8) unless the input itself is shorter than `window/4`.
5. For each group of `batch_size` chunks: call `embed_projections_batched(group, &mut tmp_proj)`, append
   raw outputs to a per-call `Vec<[f32; 512]>`.
6. **Aggregate.** Two paths — selected at construction by §3.2:
   - **Centroid path** (ONNX outputs un-normalized projections): component-wise mean of the raw projections,
     then L2-normalize the centroid → `Embedding`.
   - **Spherical-mean path** (ONNX outputs already unit-norm — the raw projections are unit vectors):
     component-wise mean of the unit vectors, then L2-normalize → `Embedding`.

   Both paths are computationally equivalent. Single-chunk case skips aggregation entirely.

**Justification for aggregation at all:** the user pipeline must handle audio of arbitrary length even where
no LAION-reference path exists for it; we acknowledge the divergence loudly (§7.3 docstring) and pick a
defensible, simple aggregator. The centroid-vs-spherical-mean distinction is a property of the model
artifact, not a user choice. **Both are textclap-specific. Cross-tool retrieval requires both indexing and
querying through textclap.**

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
       .ok_or_else(|| Error::TokenizerLoad { message: "no pad token; supply tokenizer.json with padding configuration", source: None })?
  ```
  **No literal-1 fallback.** Hardcoding `1` is correct only for RoBERTa; it's wrong for BART (1=BOS), GPT-2,
  BERT-base-uncased (1=UNK), etc.

- **`max_length: usize`** — from the tokenizer's truncation configuration (typically 77 or 512).

**Padding-mode handling depends on the construction path** (§7.4):
- `from_files` / `from_memory`: `Padding::Fixed` is silently rewritten to `BatchLongest`. `Padding::None` /
  no padding configured: textclap calls `with_padding(...)` to enable `BatchLongest` using the resolved
  `pad_id`.
- `from_ort_session`: `Padding::Fixed` → `Error::TokenizerLoad`. Other padding modes are accepted as-is.

### 9.2 `TextEncoder` orchestration

> §3.2 backfill: tensor names and dtypes (`input_ids`, `attention_mask`, possibly `position_ids`) confirmed
> by `inspect_onnx.py` before implementation.

§7.3.1 scratch-lifecycle contract applies (clear → reserve → extend → bind → run → drop views).
Per-call shape validation after `try_extract_tensor` matches silero pattern.

**`embed(text)`:**
1. `text.is_empty()` → `Error::EmptyInput { batch_index: None }`.
2. `tokenizer.encode(text, add_special_tokens=true)` → `Encoding`.
3. `ids: Vec<i64>` — clear, reserve(T), extend_from_slice from cast u32 ids. Same for `mask: Vec<i64>`.
4. If §3.2 says `position_ids` is externalized: clear/reserve/extend `pos: Vec<i64>` computed from `mask`
   and the resolved `pad_id`.
5. Bind tensor views, run, validate output shape `[1, 512]`, copy out, drop views.
6. L2-normalize the output (or skip if §3.2 says outputs are already unit-norm) → `Embedding`.

**`embed_batch(texts)`:**
1. Empty slice → `Ok(Vec::new())`.
2. For each `texts[i]`: empty → `Error::EmptyInput { batch_index: Some(i) }`.
3. `tokenizer.encode_batch(texts)` → all encodings already padded to `T_max` (BatchLongest applied per §9.1).
4. Resize encoder-owned ids/mask scratch via clear/reserve/resize to `[N × T_max]`; copy in-place.
5. Bind tensor views, run, validate output shape `[N, 512]`, copy out, drop views.
6. Row-by-row L2-normalize → `Vec<Embedding>`.

## 10. Error type

Single `thiserror` enum, exposed at crate root. **No `#[non_exhaustive]`** — sibling convention
(silero `error.rs:5–6`, soundevents `lib.rs:324–325`) treats new variants as minor-version breaks.

```rust
#[derive(Debug, thiserror::Error)]
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

    #[error("audio sample at index {sample_index} (clip {clip_index:?}) is non-finite")]
    NonFiniteAudio { clip_index: Option<usize>, sample_index: usize },

    #[error("invalid chunking options: window={window_samples}, hop={hop_samples}, batch={batch_size}; \
             all must be > 0 and hop ≤ window")]
    ChunkingConfig { window_samples: usize, hop_samples: usize, batch_size: usize },

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

    #[error("embedding norm out of tolerance: |norm² − 1| = {norm_sq_deviation:.3e}")]
    EmbeddingNotUnitNorm { norm_sq_deviation: f32 },

    #[error("unexpected tensor shape for {tensor}: expected {expected:?}, got {actual:?}")]
    UnexpectedTensorShape { tensor: &'static str, expected: Vec<i64>, actual: Vec<i64> },

    #[error("ONNX runtime error: {0}")]
    Onnx(#[from] ort::Error),
}
```

`UnexpectedTensorShape` carries structured, allocation-cheap fields (mirrors silero `UnexpectedOutputShape`).

## 11. Engineering robustness

### 11.1 ort version coupling

`ort = "2.0.0-rc.12"` (caret) matches silero/soundevents byte-identically. Bumping requires a coordinated
change across the trio.

### 11.2 Model file integrity

The README publishes SHA256s (also in `tests/fixtures/MODELS.md` per §3.3) for the three model artifacts.
Mismatched files produce undefined results — typically a runtime tensor-shape error (caught by §7.3.1
shape validation) or, worse, silent embedding drift.

textclap does not re-verify SHA256s at runtime. NFS/symlink/mmap deployments wanting stricter integrity
guarantees verify before construction.

### 11.3 Quantization variant compatibility

textclap 0.1.0 is verified against the **INT8-quantized** export specifically. Tolerances against the
**Python int8 reference**:

| Variant  | Audio embedding tolerance vs Python int8 reference | Notes              |
|----------|----------------------------------------------------|--------------------|
| int8     | < 5e-4 (verified target — see §12.2)               | This release       |
| fp16     | likely < 5e-3                                      | Not verified       |
| fp32     | likely < 1e-2                                      | Not verified       |

The fp32-vs-int8 column entry is intentionally looser than int8-vs-int8 — the comparison is across
quantization regimes. fp32-vs-fp32 (same-precision goldens regenerated against the fp32 export) would be
tighter than int8-vs-int8.

### 11.4 Cold-start latency

`AudioEncoder::warmup`, `TextEncoder::warmup`, and `Clap::warmup` run a single dummy forward each — 480 000
samples of silence for audio, **a ~60-token synthetic string for text** (the pangram "the quick brown fox
jumps over the lazy dog" repeated three times — long enough that typical Whisper transcripts of 40–77 tokens
don't reallocate token scratch on the first real call). Sizes scratch for batch size 1; first batched call
still grows scratch once. `warmup_for_batch(audio_n, text_n)` is a §14 follow-up.

### 11.5 Test determinism and platform variance

Tests rely on the same ORT default threading as production. Tests that need single-threaded determinism
build their own `ort::Session` with `intra_op_threads=1` and pass it via `from_ort_session` — sibling
convention places this configuration outside the crate's API.

**Hardware and platform variance is intrinsic and non-trivial.** ORT's CPU EP differs across OSes (MLAS on
Linux/Windows, Accelerate on macOS); FMA fusion and vectorization differ between x86 and ARM, and between
AVX-512 and AVX2. Embedding values can differ at the ULP level across runners; CI tolerances (§12.2)
absorb this.

**Bench warmup.** `benches/` Criterion harnesses each call `warmup()` in their setup closure before the
`iter` loop, so first-sample cold-start cost doesn't skew the median.

### 11.6 Model attribution and license compliance

The crate ships with no model files. The README states that **downstream users redistributing model files
take on the attribution responsibilities** of the upstream licenses:
- LAION CLAP weights: **CC-BY 4.0** — attribution required when redistributing.
- Xenova ONNX export: **Apache-2.0**.
- HTSAT and CLAP papers: citation required (BibTeX in README).

## 12. Testing strategy

### 12.1 Unit tests (per module)

- **`mel.rs`:**
  - Hann window numerical correctness (periodic convention).
  - Filter rows 0, 10, 32 vs librosa references at `max_abs_diff < 1e-6`.
  - power_to_dB applied exactly once after the mel filterbank (§8.1.2).
  - Repeat-pad behavior on `len < target`.
  - eps clamp on silence input — no NaN/Inf in the log transform.
  - Output buffer shape matches `T` from `golden_params.json`.
- **`audio.rs`:**
  - Boundary tests: `embed(&[])` → `EmptyAudio`; `embed(&[0.0; 480_001])` → `AudioTooLong`;
    `embed(&[0.0; 480_000])` succeeds.
  - `embed(&[f32::NAN, ...])` → `NonFiniteAudio { clip_index: None, sample_index: 0 }`. Same for ±Inf.
  - `embed_batch` with **uneven-length** clips succeeds (auto-pad).
  - `embed_batch` with one empty clip → `EmptyAudio { clip_index: Some(i) }`.
  - `embed_batch` with one non-finite clip → `NonFiniteAudio { clip_index: Some(i), sample_index }`.
  - Empty batch slice → empty `Vec`.
  - Chunked windowing offsets and chunk counts, including trailing-chunk-skip rule.
  - `ChunkingOptions { hop > window }` → `Error::ChunkingConfig` with all three values reported.
  - **Scratch lifecycle stress test:** small batch then large batch in sequence; results match
    independently-computed singles within §12.2 tolerance. Reverse order also works. Liveness check
    only — the borrow-checker enforcement of the §7.3.1 contract is the structural guarantee.
- **`text.rs`:**
  - `EmptyInput` for empty `&str` and empty string at index `i` in batch.
  - Empty batch slice → empty `Vec`.
  - `from_files` with `Padding::Fixed` JSON: tokenizer is rewritten to `BatchLongest`; assert
    `tokenizer.get_padding().strategy == BatchLongest`.
  - `from_ort_session` with caller-supplied tokenizer using `Padding::Fixed`: returns `TokenizerLoad`.
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
  - Builder methods round-trip through accessors for both `with_*` and `set_*` mirrors.
  - `Options::default() == Options::new()`.
- **`Embedding`:**
  - `from_slice_normalizing` always produces unit-norm output for non-zero input.
  - `from_slice_normalizing` rejects all-zero input → `EmbeddingZero`.
  - `try_from_unit_slice` rejects wrong lengths (`EmbeddingDimMismatch`) and non-unit-norm input
    (`EmbeddingNotUnitNorm`) at the 5e-5 norm² budget.
  - `dot ≈ cosine` for unit inputs (within fp32 ULP).
  - `is_close(&self, &self, 0.0)` returns true.
  - `is_close_cosine(&self, &self, 0.0)` returns true.
  - `to_vec` and `as_slice` are byte-equal.
  - **No `pub const DIM`** — compile-time test that `Embedding::DIM` doesn't resolve.
  - **Custom `Debug` output** does not contain 512 floats.
  - **No `PartialEq` derived** — compile-time test that `Embedding: PartialEq` doesn't resolve.

### 12.2 Integration test (`tests/clap_integration.rs`)

Gated on `TEXTCLAP_MODELS_DIR` env var (skip with `eprintln!` if unset, do not fail).

Tests construct sessions via `from_ort_session` with `intra_op_threads=1` (deterministic).

**Tolerance reference table** (use these values when calling `is_close` / `is_close_cosine` in tests):

| Comparison                                      | `is_close` (max-abs) | `is_close_cosine` (1−cos) |
|-------------------------------------------------|----------------------|----------------------------|
| Rust audio embedding vs golden (int8 vs int8)   | 5e-4                 | 1e-6                       |
| Rust text embedding vs golden (int8 vs int8)    | 1e-5                 | 1e-9                       |
| Cross-quantization audio (fp32 reference)       | 1e-2                 | 1e-3                       |
| Cross-thread same-encoder reproducibility       | 1e-5                 | 1e-9                       |

Assertions:

| Check                                                          | Tolerance (`max_abs_diff`)         |
|----------------------------------------------------------------|-------------------------------------|
| Rust mel features vs `golden_mel.npy`                          | < 1e-4                              |
| Rust audio embedding vs `golden_audio_emb.npy`                 | < 5e-4 (post-L2)                    |
| Rust text embeddings vs `golden_text_embs.npy`                 | < 1e-5                              |
| `classify_all` discrimination check (see below)                | structural                          |

**Why no projection-vs-projection check.** rev-6 had an `audio raw projection vs golden_audio_proj.npy` row
at 1e-3. The arithmetic is inverted: pre-L2 projections from HTSAT have norm ≈ 10, so a post-normalize
embedding drift of 5e-4 corresponds to per-component pre-normalize drift of ~5e-3 — i.e. the projection
budget should be ~10× *looser* than the embedding budget, not tighter. Worse: the projection check is
redundant for the single-window path (the embedding check covers the same ground), and there is no Python
reference for the chunking path (no equivalent of the centroid math runs in LAION's reference). Drop it.

**Tolerance origin (audio).** Mel drift up to 1e-4 propagates through HTSAT (~14 transformer blocks) with
typical 5–50× contractive amplification. L2 normalization tightens this — direction is more stable than
magnitude. **5e-4 is the opening tolerance**; calibrated downward only after real-run measurement supports
it. Per-OS budget tables (Linux MLAS vs macOS Accelerate) may be needed.

**Tolerance origin (text).** Integer token ids → no upstream drift; 1e-5 catches RoBERTa wiring bugs.

**`try_from_unit_slice`'s norm² budget (5e-5)** is intentionally tighter than the audio-embedding budget —
a stored unit vector should not drift in fp32 storage; tightening the round-trip check catches accidental
mutation of stored vectors.

**Discrimination check:** `classify_all` is run with the labels
`["a dog barking", "rain", "music", "silence", "door creaking"]`. The test asserts:
1. `"a dog barking"` ranks in the top 2.
2. `score("a dog barking") - score("music") > 0.05` (irrelevant baseline).

Tie-breaks between dog-bark and acoustically similar labels can swap under int8 quantization; the test
requires the model to *discriminate* (large margin against unrelated labels).

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

If a runner consistently exceeds a §12.2 budget by a fixed amount, that's a hardware/EP signal — investigated,
then either widened with documentation or papered over with a per-OS budget table.

## 13. Migration from current template

textclap is currently the bare `al8n/template-rs` scaffold.

### Replace
- `Cargo.toml` (identity, deps, dev-deps including rubato/npyz/hound/criterion, features, MSRV, version 0.1.0,
  caret-pinned `ort`, `examples` marked `publish = false`, `[lints.rust]` block, `[package.metadata.docs.rs]`,
  `include = [...]` whitelist excluding `tests/fixtures/`).
- `README.md` — purpose, install, quick-start (`Clap::from_files` → `warmup()` → audio embed + text embed +
  zero-shot classify), model-acquisition note pointing to HuggingFace **with SHA256s** and HF revision pin,
  warning that `tokenizer.json` must come from the same Xenova export (not from
  `laion/clap-htsat-unfused` directly), model-attribution-on-downstream section (§11.6), ort-coupling note,
  license, lancedb integration snippet, deployment note that thread-per-core means each worker calls
  `from_files` **sequentially at startup** with **150–300 MB resident per worker** (or use the
  single-encoder constructors `AudioEncoder::from_files` / `TextEncoder::from_files` to load only one
  side per process — see the deployment-pattern guidance in §1.1), **the §1.1 indexing-vs-query pipeline
  diagram** (audio encoder runs at indexing time on fixed 10 s windows; text encoder runs at query time
  on user search text only; STT/transcripts are out of textclap's scope), §1.2 domain-of-training and
  short-clip caveats, thread-tuning note pointing users to `from_ort_session` for thread/EP configuration.
- `src/lib.rs` (keep crate-level lints; replace body with module decls and the explicit re-exports listed
  in §4).
- `tests/foo.rs` → delete.
- `benches/foo.rs` → delete.
- `examples/foo.rs` → delete; add `examples/index_and_search.rs` and `examples/audio_window_to_clap.rs`.
- `CHANGELOG.md` → reset to Keep-a-Changelog stub starting at `[0.1.0]`.

### Keep
- `build.rs` — copied verbatim from sibling crates (emits `cargo:rustc-cfg=tarpaulin` when
  `CARGO_FEATURE_TARPAULIN`/`CARGO_TARPAULIN`/`CARGO_CFG_TARPAULIN` is set).
- License files (update copyright holder/year).
- `.github/workflows/` skeleton, with deletions per §12.5.

### Add
- `src/error.rs`, `src/options.rs`, `src/mel.rs`, `src/audio.rs`, `src/text.rs`, `src/clap.rs`.
- `tests/fixtures/` contents per §3 / §12.2 (including `README.md` for sample.wav provenance and `MODELS.md`
  for model SHA256s).
- `examples/index_and_search.rs` — both halves at a glance: an indexing loop (audio frames →
  `AudioEncoder::embed` → stubbed `lancedb.write`) followed by a query (`"dog barking near a door"` →
  `TextEncoder::embed` → stubbed `lancedb.search`). Shows the asymmetric encoder lifetime — indexing runs
  many times per minute, query runs on demand.
- `examples/audio_window_to_clap.rs` — the §1.1 indexing path in isolation: source frames at native rate
  (the example uses 44.1 kHz) → `rubato` resample to 48 kHz → buffer 10 s of samples →
  `AudioEncoder::embed` → 512-dim `Embedding` → push to a stubbed lancedb writer. No VAD; no per-segment
  slicing.
- This spec under `docs/superpowers/specs/`.

### lancedb integration snippet (for README)

```rust
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use textclap::Embedding;

// ── Indexing side: every 10 s of buffered 48 kHz mono audio ──
let embedding: Embedding = clap.audio_mut().embed(&pcm_48khz_mono_10s)?;
let dim = embedding.as_slice().len() as i32;          // dimension-agnostic
let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
builder.values().append_slice(embedding.as_slice()); // copies into Arrow's MutableBuffer
builder.append(true);
// (build a record batch with this column + timestamps + metadata → table.add(...))

// ── Query side: when a user submits a text search ──
// The text encoder runs ONCE per query, not per indexed item.
let query: Embedding = clap.text_mut().embed("dog barking near a door")?;
let _ = table.search(query.to_vec()).limit(10).execute().await?;

// ── Read-back (rare; lancedb usually computes similarity for you) ──
let raw: Vec<f32> = row.get("audio_embedding")?;
let stored = Embedding::try_from_unit_slice(&raw)?;   // validates len AND norm
let sim = query.cosine(&stored);
```

## 14. Known follow-ups (out of scope for 0.1.0)

- `serde` round-trip tests for `Embedding`, `Options`, `ChunkingOptions`.
- 1024-dim CLAP variants (`larger_clap_general`, `larger_clap_music`, `clap-htsat-fused`).
- Quantization-tolerance matrix populated for fp16 and fp32 exports (§11.3).
- Optional execution-provider configuration (CUDA, CoreML) layered on top of `from_ort_session`.
- `warmup_for_batch(audio_n: usize, text_n: usize)`.
- `Clap::warmup_text(&str)` if the synthetic warmup string proves too short for some workloads (§11.4).
- A second chunking-aggregation strategy (max, attention pooling, mean-of-logits) if a real CLAP use case
  demonstrates value. Adding it brings back the `Aggregation` enum + `ChunkingOptions::with_aggregation`.
- A `pad_mode: silence` option in `ChunkingOptions` to replace repeat-pad with zero-pad for short clips
  (addresses §1.2 periodicity-artifact concern).
- An optional **strict LAION-reference mode** for `embed_chunked` — single-window rand_trunc with a
  caller-provided RNG seed.
- A doctest on `Embedding::cosine` showing the lancedb round-trip specifically.
- `tracing` feature for service-tier observability.
- `try_reserve_exact` on scratch resizes to surface OOM as `Error::ScratchAlloc` instead of panic.
- `Options::with_truncation_warn_threshold(usize)` to log when text inputs hit the silent truncation cap.
- **Pre-allocation of scratch to a fixed `MAX_BATCH × 64 × T` at construction**, which would eliminate the
  resize-during-inference class structurally instead of relying on the §7.3.1 borrow-checker pattern.
  Trade: API rigidity (`embed_batch(N > MAX_BATCH)` becomes an error or silently re-allocates anyway). The
  current design preserves dynamic batch sizing; adopt this if profiling or fuzz-style stress testing
  surfaces resize-related issues.
- **In-flight cancellation.** ORT 2.x exposes `RunOptions::new()?` and `Arc<RunOptions>::terminate()` for
  cross-thread cancellation. Implementing this would require threading an `Arc<RunOptions>` through every
  `embed*` call and exposing a `CancelHandle` type. Deferred — feature decision, not infeasibility.
- An AddressSanitizer CI job (`RUSTFLAGS="-Z sanitizer=address" cargo +nightly test`) — Miri can't cross
  the FFI boundary, but ASan does. Only worth adding if the §7.3.1 contract ever needs empirical
  re-validation beyond what the borrow checker enforces statically.
