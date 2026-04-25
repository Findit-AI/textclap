# textclap — CLAP Inference Library Design

**Status:** Draft (revision 11, post-rev-10 review)
**Date:** 2026-04-25
**Target version:** 0.1.0

## 1. Purpose

textclap is a Rust inference library for **CLAP** (Contrastive Language-Audio Pre-training). It loads the
audio (HTSAT) and text (RoBERTa) ONNX encoders of LAION's `clap-htsat-unfused` model — typically the
`Xenova/clap-htsat-unfused` export — and exposes them alongside a zero-shot classification helper. It
follows the API conventions of the sibling crates `silero` (VAD), `soundevents` (sound classification), and
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
  → caller-supplied decoder → resample to 48 kHz → buffer 10 s of f32 mono PCM
  → AudioEncoder::embed → 512-dim audio embedding
  → lancedb.write { audio_embedding, ts_start, ts_end, ... }
```

The audio encoder runs every 10 s of input. There is no VAD — CLAP is a general-audio model trained on
AudioSet (speech, music, ambient sounds, alarms, animals, traffic, all of it), so the right input is the
whole audio stream chunked at the model's training window. **Decoding and resampling are out of scope:**
textclap accepts 48 kHz mono `f32` PCM as the slice handed to `embed`; the caller wires up whatever decoder
+ resampler suits their source format.

**Windowing commitment.** At startup, do not emit until the first 10 s buffer is full (no half-window
embeddings). At shutdown / EOF, the trailing partial buffer is dropped. This commits the windowing semantics
that real deployments will need anyway, so the spec doesn't punt the question.

**Silent / low-energy windows are a caller concern.** Hours of silence at 48 kHz produce hundreds of
near-identical embeddings clogging the vector store. textclap performs **no** energy gating; callers may
apply a coarse RMS / dBFS threshold before calling `embed` to skip dead windows.

**Query path (read side, on demand when a user submits a search):**

```text
user query text (e.g. "dog barking near a door")
  → TextEncoder::embed → 512-dim text embedding
  → lancedb cosine-similarity search against the audio_embedding column
  → ranked audio windows
```

The text encoder runs **once per search query**. It does *not* sit in the indexing path; it does *not*
embed Whisper transcripts or any STT output. Its sole job is to convert a free-form text query into a
vector in the same 512-dim space as the indexed audio embeddings, so cosine similarity finds matching audio.

**Out of textclap's scope (but worth flagging because it's adjacent to a real deployment).** A user
pipeline may run silero VAD + Whisper STT on the same source audio to produce transcripts and store them in
a *separate* lancedb column for caption display, BM25 / FTS keyword search, or other text-based recall.
That branch runs in parallel with textclap and does **not** route through CLAP's text encoder.

### 1.2 Use cases beyond live indexing + live query

The §1.1 split is the recommended live deployment, but the API also serves three secondary cases:

- **Offline single-clip embedding (`embed_chunked`).** For long-form audio — a 30-min podcast, a film
  scene, a half-hour field recording — when the caller wants *one* embedding describing the whole clip,
  not one per 10 s window. `embed_chunked` is for this. **Caveat:** its aggregation is textclap-specific,
  not LAION-reference compatible (§7.3, §8.2). For live indexing, use `embed`.

- **Offline batch embedding (`embed_batch`).** For backfilling an index after first-time setup or
  re-indexing after a model update — N pre-collected 10 s windows go in, N independent embeddings come out.
  Live indexing uses `embed`, not `embed_batch`, since one window is ready every 10 s of wall-clock.

- **Ad-hoc / diagnostic classification (`Clap::classify*`).** Zero-shot tagging of a single clip against
  a fixed label set, mostly for spot checks ("is this audio file actually a dog barking?") and the
  discrimination test in §12.2. Not part of the live pipeline.

The audio encoder is therefore called via `embed` (live), `embed_batch` (offline backfill), `embed_chunked`
(offline single-clip aggregation), or `classify*` (diagnostic). The text encoder is called only via
`embed` (per query) or `classify*` (per diagnostic call).

### 1.3 What CLAP recognizes — and doesn't

**Domain-of-training.** CLAP-HTSAT-unfused was trained on AudioSet plus general-audio captions: it
discriminates speech-vs-music-vs-ambient, recognizes specific sound categories (dog barks, alarms, traffic,
machinery, water, applause), and tracks coarse acoustic scene attributes. It is suited to descriptive text
queries like *"rain on a metal roof,"* *"applause in a stadium,"* *"engine starting,"* *"speech with a loud
crowd in the background."*

**It is NOT suited to within-speech content queries** like *"the meeting where Alice mentioned Q3
revenue."* Conversations cluster tightly in CLAP-audio space because their acoustic features are similar
regardless of what was said. For that kind of recall, the user's pipeline indexes Whisper transcripts as
plain text in a separate column (BM25 / FTS / a separate sentence-embedding model); textclap is not
involved.

**Short-clip artifacts (off the recommended path).** If callers feed `embed()` clips shorter than 10 s —
e.g. per-event-onset slicing for some non-default flow — textclap's repeat-pad fills the window by tiling,
which produces real periodicity artifacts in the mel spectrogram (a 1 s clip tiled to 10 s creates a 1 Hz
repetition pattern and 10 identical positional patches in HTSAT's input). Recommended minimum for that
non-default flow is ~2.5 s of original content (matching the §7.8 trailing-chunk-skip threshold of
`window/4`). For the recommended fixed-window indexing flow (§1.1), this never triggers — every input is
exactly 10 s. A `pad_mode: silence` alternative is a §14 follow-up.

## 2. Non-goals

- **Audio decoding and resampling.** Input must be 48 kHz mono `f32` PCM. Caller's responsibility.
- **Streaming inference.** CLAP isn't streaming.
- **Vector store integration.** Embeddings are emitted; storage and ANN search live in the caller.
- **Model bundling or download helpers.** No models in the crate, no network at build or runtime.
- **Async / runtime ownership.** Synchronous library; no in-flight cancellation in 0.1.0 (see §14).
- **Multi-variant CLAP support in 0.1.0.** Only the 512-dim `Xenova/clap-htsat-unfused` export is verified.
  The public API does not lock to this dimension (§7.5).
- **NaN/Inf-safe arithmetic.** Non-finite samples are detected and rejected up front (§7.3, §10).
- **Energy gating / VAD.** Caller's responsibility (§1.1).
- **Cross-tool embedding interop for chunked audio.** textclap's `embed_chunked` is a textclap-specific
  convention (§7.3, §8.2). Single-window `embed` does match the LAION reference within the verified
  tolerance.
- **Thread / EP tuning knobs on `Options`.** Sibling convention (silero, soundevents) deliberately omits
  these — runtime policy is configured one layer up by building an `ort::Session` directly and passing it
  via `from_ort_session`.

## 3. Pre-implementation prerequisites

Several parameters in the audio preprocessing pipeline cannot be safely guessed; they must be measured
against the actual model files before any `src/` Rust is written.

### 3.1 Reference-parameter dump and golden generation

`tests/fixtures/regen_golden.py` (pinned `transformers` / `optimum` / `onnxruntime` / `torch` / `librosa`
versions in a header comment):

1. Loads the test audio fixture (`tests/fixtures/sample.wav`, ≤10 s, 48 kHz mono — provenance and license
   in `tests/fixtures/README.md`).
2. Constructs `ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")` and writes
   `tests/fixtures/golden_params.json` capturing every numerical-fidelity parameter, **read directly from
   the constructed extractor object**:
   - `sampling_rate`, `feature_size` (mel bins), `fft_window_size`, `hop_length`, `max_length_s`
   - `mel_scale`, filterbank `norm` argument
   - `power_to_db`: `amin`, `ref`, `top_db` (read directly from `extractor.top_db`).
   - Window function: periodic vs symmetric (read by computing the actual array and inspecting it).
   - **Frame count `T`** — the time dimension produced by the extractor on a 480_000-sample input. With
     `n_fft=1024`, `hop=480`: `center=False` ⇒ 998 frames; `center=True` ⇒ 1001. The actual value, the
     centering flag, and the pad mode all go into `golden_params.json`. mel.rs uses `T` from there.
   - `padding` mode, `truncation` mode (the latter intentionally diverges in Rust — see §8.1).
   - `frequency_min` / `frequency_max`.
   - **Warmup token target.** The script generates the warmup string deterministically: starting from the
     pangram `"the quick brown fox jumps over the lazy dog "`, repeat it the smallest integer `k` times
     such that the post-tokenizer count is ≥80 BPE tokens (above CLAP's typical `max_length=77`). The
     resulting `pangram * k` string is recorded as `warmup_text` in `golden_params.json` along with the
     measured token count. This is reproducible across maintainers — no hand iteration. The literal
     string is what `Clap::warmup` feeds to the text encoder (§11.4).
3. Runs the extractor; saves resulting `[64, T]` mel features to `golden_mel.npy`.
4. **Audio model golden.** Loads `audio_model_quantized.onnx` via `onnxruntime.InferenceSession`; runs it
   on the golden mel features. Computes the L2-normalized embedding using the *exact* formula
   ```python
   x = raw_output.astype(np.float32)
   norm = np.linalg.norm(x).astype(np.float32)
   embedding = (x / norm).astype(np.float32)
   ```
   (deliberately *not* `torch.nn.functional.normalize`, which can differ in summation order) and saves to
   `golden_audio_emb.npy`.
5. **Text model golden.** Loads `text_model_quantized.onnx` and `tokenizer.json`. Runs five fixed labels
   (`["a dog barking", "rain", "music", "silence", "door creaking"]`). Saves L2-normalized embeddings
   `[5, 512]` to `golden_text_embs.npy`.

**Why goldens come from the int8 ONNX, not the fp32 PyTorch path:** the Rust crate runs the int8 ONNX.
Goldens must run the same int8 ONNX in Python — otherwise the test tolerance has to absorb both
quantization drift *and* implementation differences, indistinguishably.

### 3.2 ONNX graph IO inspection — and functional verification

`tests/fixtures/inspect_onnx.py` does both static graph inspection and a functional end-to-end check.

**Static inspection.** For each ONNX file, dump `graph.input` / `graph.output` (name, dtype, shape with
dynamic dims marked) and the first/last 20 graph nodes into `tests/fixtures/golden_onnx_io.json`. From this
file the spec answers:

- **Audio input shape:** `[batch, 1, 64, T]` *vs* `[batch, 64, T]` (channel dim present?).
- **Audio output L2-normalize?** Examine the last 5 graph nodes for an `LpNormalization` op (axis=-1, p=2)
  or the equivalent `ReduceL2` + `Div` pattern. Record `audio_output_is_unit_norm: true|false`.
- **Text input names and dtypes:** `input_ids: [batch, T] i64`, `attention_mask: [batch, T] i64`, plus
  whether `position_ids` appears as a third input.
- **Text output L2-normalize?** Same check.
- **Text truncation max_length** — read from the `tokenizers` Python binding's `tokenizer.truncation`
  property.
- **Audio output / text output names** — recorded as `audio_output_name`, `text_output_name` for use as
  Rust constants by §8.2 / §9.2.

**Functional verification of HTSAT input normalization.** Static inspection alone is insufficient. The
script runs both transformations and picks the lower-error one:

```python
import json
import numpy as np
import torch
import torch.nn.functional as F

# Names recorded by the static-inspection pass earlier in this script.
onnx_io = json.load(open("tests/fixtures/golden_onnx_io.json"))
AUDIO_INPUT_NAME = onnx_io["audio_input_name"]      # e.g. "input_features"
AUDIO_OUTPUT_NAME = onnx_io["audio_output_name"]    # e.g. "audio_embeds"

# AudioSet stats per AST/HTSAT convention; computed in dB space (post-power_to_db).
# Source: LAION CLAP repo / Xenova export config — confirm against the actual checkpoint.
AUDIOSET_MEAN = -4.27
AUDIOSET_STD  =  4.57

def apply_audioset_norm(x):
    """Per-element global mean/std normalization in dB space."""
    return (x - AUDIOSET_MEAN) / AUDIOSET_STD

# Single-clip input — assert batch_size == 1 so .norm() is well-defined.
features = extractor(audio, sampling_rate=48000, return_tensors="pt")  # BatchFeature dict
assert features["input_features"].shape[0] == 1, "verification expects batch_size=1"

pt_emb = pt_model.get_audio_features(**features)        # [1, 512] fp32
pt_emb = F.normalize(pt_emb, dim=-1)                    # robust to any batch size

# Try BOTH input transforms; pick whichever agrees better.
results = {}
for name, fn in [("none", lambda x: x), ("global_mean_std", apply_audioset_norm)]:
    ort_input = fn(features["input_features"].numpy()).astype(np.float32)
    ort_raw = audio_session.run([AUDIO_OUTPUT_NAME], {AUDIO_INPUT_NAME: ort_input})[0]
    ort_emb = ort_raw / np.linalg.norm(ort_raw).astype(np.float32)
    drift = float(np.max(np.abs(pt_emb.numpy() - ort_emb)))
    results[name] = drift

# Decision rule:
#   < 5e-3       → pass; record this transform
#   5e-3 .. 2e-2 → yellow zone; pick whichever produced less drift; tiebreak: prefer "none"
#                  (simpler and auditable); warn in stdout
#   ≥ 2e-2       → reject (both transforms); something else is wrong, investigate
chosen, drift = min(results.items(), key=lambda kv: kv[1])

# Write to golden_params.json (read/modify/write by this script — see §3.1).
golden_params = json.load(open("tests/fixtures/golden_params.json"))
golden_params["htsat_input_normalization"] = {
    "type": chosen,
    "mean": AUDIOSET_MEAN if chosen == "global_mean_std" else None,
    "std":  AUDIOSET_STD  if chosen == "global_mean_std" else None,
}
golden_params["htsat_norm_drift"] = drift
json.dump(golden_params, open("tests/fixtures/golden_params.json", "w"), indent=2)
```

The 5e-3 / 2e-2 thresholds are **heuristic**: calibrated against published int8 quantization drift estimates
(~1e-3 to 5e-3) plus a margin for HTSAT's contractive amplification. They are not from a peer-reviewed
source; revisit if §12.2 integration drift exceeds the budget at first-real-run measurement.

The chosen transform is recorded in `golden_params.json` as `htsat_input_normalization: "none" | "global_mean_std"` plus the corresponding `mean` / `std` constants.
Rust's mel.rs applies it (or doesn't) accordingly.

### 3.3 Model SHA256 acquisition

Before §3.1 / §3.2, the maintainer downloads the three model artifacts from a pinned Hugging Face revision
(commit hash recorded in `tests/fixtures/MODELS.md`) and computes:

```
shasum -a 256 audio_model_quantized.onnx text_model_quantized.onnx tokenizer.json
```

The SHA256s, the HF revision hash, and the URL are recorded in `tests/fixtures/MODELS.md` and reproduced in
the README.

### 3.4 Spec-update commit sequence

§3.1–§3.3 produce an explicit multi-commit sequence:

0. **Models commit:** `tests/fixtures/MODELS.md` (SHA256s + HF revision pin + URLs).
1. **Scripts commit:** `regen_golden.py` and `inspect_onnx.py` source.
2. **Generated-fixtures commit:** `golden_params.json`, `golden_onnx_io.json`, `golden_*.npy`,
   `tests/fixtures/README.md`.
3. **Spec-update commit:** §8.1 mel parameter table, §8.2 / §9.2 tensor names and shapes, §7.4
   attention-mask / position_ids description, §11.4 warmup string, §12.2 tolerance table — all replaced
   with values consistent with the generated fixtures.
4. **Rust src/ skeleton commit:** module structure, public types with `unimplemented!()` bodies, error
   variants, options, constants from §3.4-3 backfilled. The crate compiles end-to-end but no method
   produces real output.
5. **`tests/clap_integration.rs` and `benches/` commits:** these reference public types from `src/`, so
   the skeleton from step 4 must exist first.
6. **Real `src/` implementation commits** replacing the `unimplemented!()` bodies — mel extractor first
   (verified by §8.1.1 / §8.1.2 unit tests against the librosa fixtures), then audio encoder, then text
   encoder, then `Clap` and the zero-shot helper.

## 4. Crate layout

```
textclap/
├── Cargo.toml                       # see §5 for [lints.rust], docs.rs metadata, include shape
├── build.rs                         # emits cargo:rustc-cfg=tarpaulin when CARGO_FEATURE_TARPAULIN /
│                                    # CARGO_TARPAULIN / CARGO_CFG_TARPAULIN is set; copied verbatim
│                                    # from the sibling crates
├── README.md
├── CHANGELOG.md
├── LICENSE-MIT / LICENSE-APACHE / COPYRIGHT
├── src/
│   ├── lib.rs                       # crate-level docs + module decls + the re-exports below
│   ├── error.rs                     # Error enum (thiserror)
│   ├── options.rs                   # Options, ChunkingOptions; re-exports GraphOptimizationLevel from ort
│   ├── mel.rs                       # MelExtractor: STFT → mel filterbank → log-mel
│   ├── audio.rs                     # AudioEncoder + AUDIO_INPUT_NAME / AUDIO_OUTPUT_NAME consts (§8.2)
│   ├── text.rs                      # TextEncoder + TEXT_INPUT_IDS_NAME / TEXT_ATTENTION_MASK_NAME /
│   │                                # (optional) TEXT_POSITION_IDS_NAME / TEXT_OUTPUT_NAME consts (§9.2)
│   └── clap.rs                      # Clap, Embedding, LabeledScore, LabeledScoreOwned
├── tests/
│   ├── clap_integration.rs          # gated on TEXTCLAP_MODELS_DIR env var
│   └── fixtures/                    # see §3 for full content
├── benches/
│   ├── bench_mel.rs
│   ├── bench_audio_encode.rs
│   └── bench_text_encode.rs
├── examples/
│   ├── audio_window_to_clap.rs      # decoder → 48 kHz → 10 s window → AudioEncoder::embed (indexing)
│   └── index_and_search.rs          # sequential demo: index one window, then run a query
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

`GraphOptimizationLevel` is re-exported by `options.rs` from `ort`, then re-re-exported by `lib.rs` — keeps
the public path stable across `ort` version bumps.

## 5. Dependencies

### Default

| Crate         | Version          | Purpose                                          |
|---------------|------------------|--------------------------------------------------|
| `ort`         | `2.0.0-rc.12`    | ONNX Runtime Rust bindings (matches sibling caret-pin) |
| `rustfft`     | `^6`             | Real-input STFT for mel extraction               |
| `tokenizers`  | `^0.20`          | HF tokenizer.json loader (RoBERTa BPE)           |
| `thiserror`   | `^2`             | Error derives                                    |

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

`examples/` are marked `publish = false`. `silero` / `mediatime` are not direct dev-deps — the kept
example shows only the indexing path; STT / VAD integration is out of scope.

### Cargo.toml shape (matching siblings)

- **`include = [...]`** — whitelist excluding `tests/fixtures/` (~MBs of `.npy` and the WAV) from the
  published crate.
- **`[lints.rust]`** block: `rust_2018_idioms`, `single_use_lifetimes`,
  `unexpected_cfgs = { level = "warn", check-cfg = ['cfg(tarpaulin)', 'cfg(all_tests)'] }`.
- **`[package.metadata.docs.rs]`**: `all-features = true` and `rustdoc-args = ["--cfg", "docsrs"]`. The
  `all-features = true` choice matches soundevents; silero omits it. Acceptable divergence from silero;
  documented here so reviewers don't expect parity on that one line.

### Excluded (deliberate)

- No `tokio`, no async — synchronous library.
- No `download` feature — no network.
- No model bundling.
- **No `ndarray`** — the mel filterbank multiply is small enough to write by hand; ONNX outputs are read
  via `try_extract_tensor::<f32>()` which yields `(Shape, &[f32])` (where `Shape: AsRef<[i64]>`) directly.
- No `tracing` — observability is a §14 follow-up.
- No `num_cpus` — `Options` does not expose thread counts.

## 6. Toolchain & metadata

- **Rust edition:** 2024
- **MSRV:** 1.85
- **License:** MIT OR Apache-2.0
- **Crate-level lints:** `#![deny(missing_docs)]`, `#![forbid(unsafe_code)]`
- **Initial version:** `0.1.0`

## 7. Public API

All public structs use private fields and accessor methods. Builder-style `with_*` returns `Self` by value;
`set_*` mirrors take `&mut self` and return `&mut Self`; both are `pub const fn`. Getters are `pub const fn`
and `#[cfg_attr(not(tarpaulin), inline(always))]` (matches silero/soundevents pattern). Field-less unit
enums and explicitly-fielded structs are public-as-data.

### 7.1 Top-level types

```rust
pub struct Clap          { /* AudioEncoder + TextEncoder */ }
pub struct AudioEncoder  { /* ort::Session + MelExtractor + encoder-owned scratch +
                             audio_output_is_unit_norm: bool (set at construction from
                             golden_onnx_io.json; selects centroid vs spherical-mean
                             aggregation in embed_chunked and the trust-vs-renormalize
                             path in embed/embed_batch). */ }
pub struct TextEncoder   { /* ort::Session + Tokenizer + cached pad_id + encoder-owned scratch +
                             text_output_is_unit_norm: bool (same role on the text side). */ }

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
121 MB int8 text model on disk, plus ORT working buffers). README recommends sequential worker
construction at startup to avoid transient 2× peak memory during ORT weight reformatting.

For asymmetric deployments (§1.1): an indexing-worker process can construct only `AudioEncoder` (saves
~120 MB resident); a query process can construct only `TextEncoder`. The README quick-start shows both
patterns explicitly.

**Drop / shutdown.** Encoders implement `Drop` safely. They are `!Sync` and owned by exactly one thread;
on drop the ORT session releases its weights and the scratch `Vec`s deallocate.

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
    /// to size encoder-owned scratch buffers for batch size 1.
    pub fn warmup(&mut self) -> Result<()>;

    // Single ~10 s clip (LAION-reference compatible — for live indexing or per-clip query embedding,
    // see §1.2 for use-case mapping).
    pub fn classify<'a>(&mut self, samples: &[f32], labels: &'a [&str], k: usize)
        -> Result<Vec<LabeledScore<'a>>>;
    pub fn classify_all<'a>(&mut self, samples: &[f32], labels: &'a [&str])
        -> Result<Vec<LabeledScore<'a>>>;

    /// Long clip via textclap-specific chunking (NOT LAION-reference compatible
    /// — see §7.3 embed_chunked docs and §8.2). For offline single-clip
    /// summarization (§1.2 use case 1).
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
    ///
    /// This is the recommended entry point for live indexing (§1.1).
    pub fn embed(&mut self, samples: &[f32]) -> Result<Embedding>;

    /// Batch of clips of any lengths in 0 < len ≤ 480_000. Each clip is
    /// repeat-padded to 10 s independently — no equal-length requirement.
    /// Returns one Embedding per input clip.
    ///
    /// Empty *slice* returns Ok(Vec::new()). Any clip with len == 0 returns
    /// Error::EmptyAudio with its index. Any non-finite sample returns
    /// Error::NonFiniteAudio with both clip_index and sample_index.
    ///
    /// **Use case (§1.2):** offline backfill or re-indexing N pre-collected
    /// 10 s windows. Live indexing uses embed, since one window is ready
    /// every 10 s of wall-clock.
    ///
    /// **Performance.** Compute scales with N × full-window regardless of
    /// input length — 8 clips of 0.3 s cost the same ONNX time as 8 clips
    /// of 10 s. For N short clips wanting N independent embeddings, this
    /// is the right call. (Concatenating short clips into embed_chunked
    /// produces ONE aggregated embedding, not N — different use case.)
    pub fn embed_batch(&mut self, clips: &[&[f32]]) -> Result<Vec<Embedding>>;

    /// Arbitrary-length input via textclap's chunking convention; produces
    /// ONE embedding for the whole input.
    ///
    /// **Use case (§1.2):** offline single-clip summarization of long-form
    /// audio when the caller wants one embedding for the whole clip.
    /// Live indexing (§1.1) uses fixed 10 s windows + embed instead.
    ///
    /// **For inputs ≤ window_samples** the chunking path produces a
    /// single chunk and the result is bit-identical to embed(). Prefer
    /// embed() in that case — fewer code paths, same answer.
    ///
    /// **WARNING — not LAION-reference compatible.** LAION's reference for
    /// the unfused model uses single-window rand_trunc, not multi-window
    /// aggregation. textclap aggregates by either
    /// (a) centroid-of-un-normalized-projections + L2-normalize, or
    /// (b) spherical-mean (mean of unit vectors) + L2-normalize.
    /// Which one is used depends on whether the ONNX export already
    /// L2-normalizes its output (§3.2 / golden_onnx_io.json determines
    /// this). Embeddings produced via embed_chunked occupy a region of
    /// CLAP-audio space that LAION-reference embeddings do not — querying
    /// a textclap-indexed audio column with embeddings produced through
    /// any other CLAP toolchain will not work.
    pub fn embed_chunked(&mut self, samples: &[f32], opts: &ChunkingOptions)
        -> Result<Embedding>;

    pub fn warmup(&mut self) -> Result<()>;
}
```

**Concurrency model:** `AudioEncoder` is `Send` but **not `Sync`**. Each worker thread owns its own
`AudioEncoder`. Encoder-owned mel feature buffer, FFT scratch, and ONNX input tensor backing are growable
`Vec<f32>`s; sized on the first call (or amortized via `warmup()`); reused thereafter via the
clear→reserve→extend pattern (§7.3.1) — the hot path performs no heap allocation after warmup.

#### 7.3.1 Scratch lifecycle contract (UB prevention)

`#![forbid(unsafe_code)]` blocks `unsafe` blocks in textclap, but **`ort 2.x`'s `TensorRef::from_array_view`
constructs views that the ORT C++ runtime borrows during `session.run()`**. If the underlying scratch
buffer is reallocated after the tensor view is bound and before `session.run()` returns, the C++ side
accesses freed memory — undefined behavior through the FFI boundary.

The implementation adopts silero's production pattern (`silero/src/session.rs`):

1. **`scratch.clear()`** — drops the previous contents but preserves capacity.
2. **`scratch.reserve(required_size)`** — grows capacity if needed; this is the *only* point at which
   reallocation can occur. Nothing borrows the `Vec` yet.
3. **`scratch.extend_from_slice(...)`** *or* **`scratch.resize(required_size, 0.0)`** — fills to the final
   length. Capacity is now committed.
4. **`TensorRef::from_array_view(scratch.as_slice(), ...)`** — borrows `&[f32]` from the `Vec`. From this
   point on, the borrow checker prevents any mutation of the `Vec`; reallocation is structurally
   impossible until the borrow ends.
5. **`session.run()`**.
6. **Tensor views drop** at end of scope. The borrow on `&[f32]` ends; the `Vec` is mutable again.

The borrow checker — not unit tests, not documentation — is the structural protection. After
`session.run()`, every output tensor's shape is validated via a `validate_shape` helper:

```rust
pub(crate) fn validate_shape(
    tensor: &'static str,
    actual: &[i64],
    expected: &[i64],
) -> Result<()>;  // (matches silero's parameter order: actual first, expected second)
```

Mismatch → `Error::UnexpectedTensorShape`. Cost is negligible; catches ORT version skew and model artifact
swaps at the friendly boundary.

### 7.4 `TextEncoder`

```rust
impl TextEncoder {
    pub fn from_files<P: AsRef<Path>>(onnx_path: P, tokenizer_json_path: P, opts: Options) -> Result<Self>;
    pub fn from_memory(onnx_bytes: &[u8], tokenizer_json_bytes: &[u8], opts: Options) -> Result<Self>;

    /// Wraps a pre-built ORT session and Tokenizer. Both are validated:
    /// - session schema vs golden_onnx_io.json (Error::SessionSchema on mismatch);
    /// - tokenizer must NOT have Padding::Fixed (Error::PaddingFixedRejected).
    pub fn from_ort_session(
        session: ort::session::Session, tokenizer: tokenizers::Tokenizer, opts: Options,
    ) -> Result<Self>;

    /// Empty &str returns Error::EmptyInput. Whitespace-only strings are
    /// accepted as-is.
    pub fn embed(&mut self, text: &str) -> Result<Embedding>;

    /// Empty *slice* returns Ok(Vec::new()). Any empty string in the batch
    /// returns Error::EmptyInput with its batch_index.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>>;

    pub fn warmup(&mut self) -> Result<()>;
}
```

**Padding::Fixed handling is asymmetric across constructors:**
- `from_files` / `from_memory`: `Padding::Fixed` in the loaded `tokenizer.json` is **silently rewritten**
  to `BatchLongest`. JSON authors rarely set `Fixed` intentionally; the silent rewrite avoids ~6× perf
  regression on typical short-query inputs without changing semantics.
- `from_ort_session`: `Padding::Fixed` is **rejected** with `Error::PaddingFixedRejected`. The caller
  built this `Tokenizer` deliberately; surface the surprise rather than mutate it.

**Concurrency model:** `Send` but **not `Sync`**. §7.3.1 scratch lifecycle contract applies identically.

**Tokenizer truncation max_length is taken from `tokenizer.json` at construction** — no `max_length` knob
is exposed. The actual value (typically 77 for CLAP, sometimes 512 for RoBERTa default) is recorded in
`golden_params.json`. Long inputs are silently truncated.

**Position-ids:** RoBERTa computes positions as `pad_id + 1 + cumsum(non_pad_mask)`. This is *typically*
inlined into Xenova's ONNX export. If §3.2 finds `position_ids` as an externalized input, the encoder
computes it explicitly using the resolved `pad_id` (§9.1) — *not* the literal `1` — and feeds it as a
third tensor.

### 7.5 `Embedding`

```rust
impl Embedding {
    pub const fn dim(&self) -> usize;            // 512 for 0.1.0; runtime-queryable, future-proof

    // Borrow-only access — supports append_slice into Arrow's MutableBuffer.
    pub fn as_slice(&self) -> &[f32];

    // Owned conversion.
    pub fn to_vec(&self) -> Vec<f32>;

    /// Reconstruct from a stored unit vector. Validates length AND norm
    /// (release-mode check: `(norm² − 1).abs() < 5e-5`).
    ///
    /// **Budget rationale.** The 5e-5 norm² budget absorbs cross-process
    /// Arrow buffer round-tripping ULP drift (the realistic in-scope case);
    /// it is intentionally not as tight as ULP because real Arrow
    /// serializers / Parquet writers / network transports can introduce
    /// small accumulated drift that is still semantically the same vector.
    /// **It is NOT a cross-platform reproducibility check** — cross-platform
    /// comparisons should use is_close_cosine. Truncation is caught earlier
    /// by EmbeddingDimMismatch; fp16 storage round-trip is OUT OF SCOPE
    /// (fp16's ulp(1.0) ≈ 9.77e-4 makes the check fail; users converting
    /// through fp16 should use from_slice_normalizing). Tighten to 1e-6 if
    /// real-world measurement supports it.
    pub fn try_from_unit_slice(s: &[f32]) -> Result<Self>;

    /// Construct from any non-zero slice; always re-normalizes to unit length
    /// (idempotent for input that's already unit-norm). Validates length and
    /// rejects all-zero input via Error::EmbeddingZero.
    pub fn from_slice_normalizing(s: &[f32]) -> Result<Self>;

    // Similarity (== for unit vectors, modulo fp32 ULP).
    pub fn dot(&self, other: &Embedding) -> f32;
    pub fn cosine(&self, other: &Embedding) -> f32;

    /// Approximate equality test — raw float drift, max-abs metric.
    /// Use for tests checking implementation determinism.
    /// Recommended tolerance values: §12.2 reference table.
    pub fn is_close(&self, other: &Embedding, tol: f32) -> bool;

    /// Approximate equality test — semantic (cosine) metric.
    /// Returns true if `1 − self.cosine(other) < tol`.
    ///
    /// **Implementation.** Computed as `0.5 · ‖self − other‖₂² < tol` for
    /// numerical stability — the algebraically-equivalent `1 − dot(a,b)`
    /// suffers catastrophic cancellation in fp32 for very-close vectors,
    /// which matters for the text row of §12.2 (tol ~5e-8). The identity
    /// `1 − cos(θ) = 0.5 · ‖a − b‖²` holds because the Embedding invariant
    /// guarantees both operands are unit-norm to fp32 ULP — if a future
    /// variant of Embedding ever stores non-unit-norm vectors, this
    /// implementation must change to a guarded `1 − dot / (‖a‖·‖b‖)`.
    ///
    /// **Geometric meaning.** For unit vectors of dim d with max-abs drift ε,
    /// `1 − cos(θ) ≤ d · ε² / 2` (worst case). The closed-form rotation
    /// bound is `θ ≤ 2 · arcsin(ε · √d / 2)`; for small ε this linearizes
    /// to `θ ≈ ε · √d`.
    ///
    /// Recommended tolerance values: §12.2 reference table.
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

**No public method exposes a fixed-size array.** Internal storage is `[f32; 512]` for 0.1.0; the API is
dimension-agnostic at signature level. **There is no `pub const DIM`** — code calls `dim()` or
`as_slice().len()`.

**Invariant:** every `Embedding` returned by this crate is L2-normalized to unit length within fp32 ULP.

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

    pub const fn with_graph_optimization_level(self, level: GraphOptimizationLevel) -> Self;
    pub const fn set_graph_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self;
    pub const fn graph_optimization_level(&self) -> GraphOptimizationLevel;
}
```

`GraphOptimizationLevel` is re-exported from `ort` via `options.rs`. **There is no `with_intra_threads`
knob.** Sibling convention: deployment-specific runtime policy is configured one layer up via
`from_ort_session`. Callers needing thread tuning, EP selection (CUDA, CoreML), or other ORT runtime
knobs build their own `ort::Session` directly.

### 7.8 `ChunkingOptions`

```rust
impl ChunkingOptions {
    pub fn new() -> Self;        // window=480_000, hop=480_000, batch_size=8

    pub const fn with_window_samples(self, n: usize) -> Self;
    pub const fn set_window_samples(&mut self, n: usize) -> &mut Self;
    pub const fn window_samples(&self) -> usize;

    pub const fn with_hop_samples(self, n: usize) -> Self;
    pub const fn set_hop_samples(&mut self, n: usize) -> &mut Self;
    pub const fn hop_samples(&self) -> usize;

    pub const fn with_batch_size(self, n: usize) -> Self;
    pub const fn set_batch_size(&mut self, n: usize) -> &mut Self;
    pub const fn batch_size(&self) -> usize;
}
```

Aggregation strategy is fixed in 0.1.0 (centroid or spherical-mean, chosen at construction per §3.2).

Validation runs at use, not at build: `embed_chunked` returns
`Error::ChunkingConfig { window_samples, hop_samples, batch_size }` if any of:
- `window_samples == 0`
- `hop_samples == 0`
- `batch_size == 0`
- `hop_samples > window_samples` (rejected because gapped chunking is rarely intentional and complicates
  the trailing-skip rule). **Deliberate divergence from soundevents'
  `validate_chunking` (`lib.rs:939–949`)**, which only rejects zero values; flagged here so reviewers
  don't expect parity.

Trailing chunks shorter than `window_samples / 4` are skipped to avoid dragging the centroid into noise
from the trailing repeat-pad.

## 8. Audio inference pipeline

### 8.1 Mel-spectrogram extractor (`src/mel.rs`)

`MelExtractor` is `pub(crate)`. Public-to-the-crate API:

```rust
impl MelExtractor {
    pub(crate) fn new() -> Self;

    /// Compute mel features and write into `out`. Caller must size `out` to
    /// exactly `64 * T` (T from golden_params.json). Asserts on length
    /// mismatch.
    pub(crate) fn extract_into(&mut self, samples: &[f32], out: &mut [f32]) -> Result<()>;
}
```

The `&mut [f32]` interface (rather than `&mut Vec<f32>`) is what enables the §7.3.1 lifecycle pattern:
`AudioEncoder` resizes its scratch `Vec` once per call, then hands the extractor sub-slices for each
batch row.

Parameters (subject to §3.1 verification — values in this table are *expected*; recorded values in
`golden_params.json` are authoritative; only the truncation row is intentionally chosen differently):

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

The ONNX input tensor shape `[N, 1, 64, T]` is built as a *view* over the mel feature scratch — the channel
dim is added at tensor-construction time with no data movement, and the underlying `Vec<f32>` length
stays `N · 64 · T`.

#### 8.1.1 Filterbank-correctness unit test

`mel.rs` ships unit tests that compare filter rows 0, 10, and 32 against pre-computed reference rows
(`librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64, fmin=50, fmax=14000, htk=False, norm='slaney')`)
committed as `tests/fixtures/filterbank_row_0.npy` etc. Tolerance: `max_abs_diff < 1e-6`. Row 10 lands
near the 1 kHz Slaney inflection — necessary to discriminate Slaney from HTK construction.

#### 8.1.2 Power-to-dB single-application test

Separate unit test asserts that `MelExtractor::extract_into` differs visibly from a hand-written
"apply power_to_dB twice" reference. Confirms the floor is applied exactly once after the filterbank.

### 8.2 `AudioEncoder` orchestration

> §3.2 backfill: audio-input tensor name and shape, plus `audio_output_is_unit_norm`, are confirmed by
> `inspect_onnx.py` before §3.4 step 3 lands.

```rust
// Filled from golden_onnx_io.json by §3.4 step 3. Module-private (matches silero
// session.rs:12 convention — wider visibility is unnecessary; these constants
// are referenced only inside audio.rs).
const AUDIO_INPUT_NAME:  &str = "input_features";
const AUDIO_OUTPUT_NAME: &str = "audio_embeds";   // example; verify via §3.2
```

`T` (the mel time dimension, recorded in `golden_params.json` per §3.1) is a `pub(crate) const T_FRAMES`
defined in `mel.rs` and re-used here. It is a compile-time constant after §3.4 step 3 backfills the value;
before backfill the §4 skeleton uses a placeholder.

**Internal helper.** A single `pub(crate)` method runs the full forward pipeline (mel → ONNX) for any
non-empty batch and writes raw model outputs into a caller-provided buffer. **Contract: prior contents of
`out` are dropped — the function clears and resizes the buffer to `clips.len()` before writing.**

```rust
/// Compute raw projections (un-normalized if §3.2 said the ONNX output is
/// not internally L2-normalized; unit-norm if it is). Caller is responsible
/// for any subsequent L2 normalization (skipped if the encoder's
/// audio_output_is_unit_norm flag is true).
///
/// `out` is cleared on entry and resized to clips.len() before writing.
pub(crate) fn embed_projections_batched(
    &mut self,
    clips: &[&[f32]],         // 1..=N clips, each 1..=480_000 samples
    out: &mut Vec<[f32; 512]>,
) -> Result<()>;
```

Single-clip operations call this with `clips = &[samples]`. One path; no perf-profile ambiguity.

§7.3.1 scratch-lifecycle contract applies to every `session.run()` here. Implementation pattern, using
ORT 2.0.0-rc.12's actual API surface (verified against silero `src/session.rs:178-187` — single tuple
form for `TensorRef::from_array_view`, and `try_extract_tensor` not `try_extract_raw_tensor`):

```rust
use ort::value::TensorRef;

let n = clips.len();
let row_len = 64 * T_FRAMES;                                       // compile-time const after §3.4
let total = n * row_len;

self.mel_scratch.clear();
self.mel_scratch.resize(total, 0.0);                               // grow + zero-fill in one step;
                                                                   // reserve is redundant after resize
for (i, clip) in clips.iter().enumerate() {
    self.mel.extract_into(
        clip,
        &mut self.mel_scratch[i * row_len .. (i + 1) * row_len],   // &mut [f32]
    )?;
}

let input = TensorRef::from_array_view((
    [n, 1usize, 64, T_FRAMES],                                     // shape: tuple of usize, NOT &[i64]
    self.mel_scratch.as_slice(),                                   // data: &[f32]
))?;

let outputs = self.session.run(ort::inputs![AUDIO_INPUT_NAME => input])?;
let (shape, data) = outputs[AUDIO_OUTPUT_NAME].try_extract_tensor::<f32>()?;
validate_shape("audio_output", shape.as_ref(), &[n as i64, 512])?; // shape.as_ref() yields &[i64]

out.clear();
out.reserve(n);
for i in 0..n {
    let mut row = [0.0f32; 512];
    row.copy_from_slice(&data[i * 512 .. (i + 1) * 512]);
    out.push(row);
}
// tensor views drop here at end of scope; mel_scratch becomes mutable again
```

**`embed(samples)`:**
1. `samples.is_empty()` → `Error::EmptyAudio { clip_index: None }`.
2. `samples.len() > 480_000` → `Error::AudioTooLong { got, max: 480_000 }`.
3. **Finiteness scan:** SIMD pass over `samples` for the first non-finite value. On hit:
   `Error::NonFiniteAudio { clip_index: None, sample_index }`.
4. Call `embed_projections_batched(&[samples], &mut self.proj_scratch)`.
5. Take `self.proj_scratch[0]`; L2-normalize → `Embedding` (or skip the normalize and construct via the
   trusted path if §3.2 says outputs are already unit-norm; debug_assert unit-norm).

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
6. **Aggregate** — selected at construction by §3.2:
   - **Centroid path** (ONNX outputs un-normalized): component-wise mean of raw projections, then
     L2-normalize → `Embedding`.
   - **Spherical-mean path** (ONNX outputs already unit-norm): component-wise mean of unit vectors, then
     L2-normalize → `Embedding`.

   Single-chunk case skips aggregation.

**Allocation budget per call (after warmup):**
- `embed`: only the output `Embedding`.
- `embed_batch(N)`: `Vec<Embedding>` of N entries; mel scratch grows once per new max size.
- `embed_chunked(L, batch=B)`: per-call `Vec<[f32; 512]>` of `ceil(L / hop)` entries.

`warmup()` runs a single `embed` (480 000 samples of silence) which sizes steady-state scratch and triggers
ORT operator specialization.

## 9. Text inference pipeline

### 9.1 Tokenizer

Loaded once at construction. textclap inspects the tokenizer to cache:

- **`pad_id: i64`** — resolved as:
  ```
  pad_id = tokenizer.get_padding().map(|p| p.pad_id)
       .or_else(|| tokenizer.token_to_id("<pad>"))
       .ok_or(Error::NoPadToken)?
  ```
  No literal-1 fallback. Hardcoding `1` is correct only for RoBERTa.

- **`max_length: usize`** — from the tokenizer's truncation configuration.

**Padding-mode handling depends on construction path** (§7.4):
- `from_files` / `from_memory`: `Padding::Fixed` is silently rewritten to `BatchLongest`. `Padding::None` /
  no padding configured: textclap calls `with_padding(...)` with the resolved `pad_id`.
- `from_ort_session`: `Padding::Fixed` → `Error::PaddingFixedRejected`. Other padding modes are accepted
  as-is.

### 9.2 `TextEncoder` orchestration

> §3.2 backfill: tensor names and dtypes (`input_ids`, `attention_mask`, possibly `position_ids`)
> confirmed by `inspect_onnx.py` before §3.4 step 3.

```rust
// Filled from golden_onnx_io.json by §3.4 step 3. Module-private (matches silero convention).
const TEXT_INPUT_IDS_NAME:      &str = "input_ids";
const TEXT_ATTENTION_MASK_NAME: &str = "attention_mask";
const TEXT_POSITION_IDS_NAME:   &str = "position_ids";   // referenced only if §3.2 found this input
const TEXT_OUTPUT_NAME:         &str = "text_embeds";    // example; verify via §3.2
```

§7.3.1 scratch-lifecycle contract applies. ORT output extraction uses `try_extract_tensor::<f32>()` and
`TensorRef::from_array_view((shape_tuple, data))` exactly as in §8.2 (no ndarray, no `try_extract_raw_tensor`).

**`embed(text)`:**
1. `text.is_empty()` → `Error::EmptyInput { batch_index: None }`.
2. `tokenizer.encode(text, add_special_tokens=true)` → `Encoding`.
3. `ids: Vec<i64>` — clear → reserve(T) → extend_from_slice from cast u32 ids. Same for `mask: Vec<i64>`.
4. If §3.2 says `position_ids` is externalized: clear/reserve/extend `pos: Vec<i64>` computed from `mask`
   and the resolved `pad_id`.
5. Bind tensor views via `TensorRef::from_array_view(([1usize, t], ids.as_slice()))?` for each input
   tensor; run; validate output shape `[1, 512]` via the §7.3.1 helper; copy out; drop views.
6. L2-normalize → `Embedding` (or skip if `text_output_is_unit_norm` is true).

**`embed_batch(texts)`:**
1. Empty slice → `Ok(Vec::new())`.
2. For each `texts[i]`: empty → `Error::EmptyInput { batch_index: Some(i) }`.
3. `tokenizer.encode_batch(texts)` → all encodings already padded to `T_max` (BatchLongest applied per §9.1).
4. Resize encoder-owned ids/mask scratch via clear/reserve/resize to `[N × T_max]`; copy in-place.
5. Bind tensor views, run, validate output shape `[N, 512]`, copy out, drop views.
6. Row-by-row L2-normalize → `Vec<Embedding>`.

## 10. Error type

Single `thiserror` enum, exposed at crate root. **No `#[non_exhaustive]`** — sibling convention treats new
variants as minor-version breaks.

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// ONNX model load from a file path (matches silero LoadModel shape).
    #[error("failed to load ONNX model from {path}: {source}")]
    OnnxLoadFromFile {
        path: PathBuf,
        #[source] source: ort::Error,
    },

    /// ONNX model load from caller-supplied bytes (no path available).
    #[error("failed to load ONNX model from memory: {0}")]
    OnnxLoadFromMemory(#[source] ort::Error),

    /// Tokenizer load from a file path (matches OnnxLoadFromFile / silero LoadModel pattern).
    #[error("failed to load tokenizer from {path}: {source}")]
    TokenizerLoadFromFile {
        path: PathBuf,
        #[source] source: tokenizers::Error,
    },

    /// Tokenizer load from caller-supplied bytes (no path available).
    #[error("failed to load tokenizer from memory: {0}")]
    TokenizerLoadFromMemory(#[source] tokenizers::Error),

    /// tokenizer.json had no padding configuration AND no <pad> token.
    /// Either the JSON must declare padding, or it must contain a <pad> token
    /// for textclap to install BatchLongest padding around.
    #[error("tokenizer has no pad token (configure padding in tokenizer.json or include a <pad> token)")]
    NoPadToken,

    /// from_ort_session received a Tokenizer with Padding::Fixed.
    /// from_files / from_memory silently rewrite Fixed → BatchLongest, but
    /// from_ort_session treats a deliberately-built Fixed tokenizer as a
    /// caller bug rather than mutating it.
    #[error("from_ort_session rejected Padding::Fixed (use BatchLongest or pre-pad upstream)")]
    PaddingFixedRejected,

    /// ONNX session schema does not match what the encoder expects (input or
    /// output name / dtype mismatch). Detail string is constructed from
    /// golden_onnx_io.json + the actual session inputs/outputs.
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

    /// Tokenization failed at runtime (rare; mostly malformed Unicode).
    #[error("tokenization failed: {0}")]
    Tokenize(#[source] tokenizers::Error),

    #[error("input text is empty (batch index: {batch_index:?})")]
    EmptyInput { batch_index: Option<usize> },

    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch { expected: usize, got: usize },

    /// Slice passed to from_slice_normalizing was all zeros (or numerically
    /// indistinguishable from zero).
    #[error("embedding is the zero vector")]
    EmbeddingZero,

    #[error("embedding norm out of tolerance: |norm² − 1| = {norm_sq_deviation:.3e}")]
    EmbeddingNotUnitNorm { norm_sq_deviation: f32 },

    /// Carries the actual shape and the expected shape (matches silero's
    /// UnexpectedOutputShape pattern but adds `expected` for richer
    /// diagnostics — deliberate divergence, flagged here).
    #[error("unexpected tensor shape for {tensor}: actual {actual:?}, expected {expected:?}")]
    UnexpectedTensorShape { tensor: &'static str, actual: Vec<i64>, expected: Vec<i64> },

    #[error("ONNX runtime error: {0}")]
    Onnx(#[from] ort::Error),
}
```

`Tokenize`, `TokenizerLoadFromFile`, and `TokenizerLoadFromMemory` carry concrete `tokenizers::Error` (not
boxed dyn), matching silero's sibling pattern. The path-carrying / memory-carrying split mirrors
`OnnxLoadFromFile` / `OnnxLoadFromMemory` for symmetry. Configuration-time errors (`NoPadToken`,
`PaddingFixedRejected`) are separate top-level variants, matching sibling structure (silero's
`InvalidChunkLength`, soundevents' `MissingRatedEventIndex`).

`Tokenize` is for runtime tokenization failures during `embed` / `embed_batch` calls. `embed_batch`
returns a single `Tokenize` error from the underlying `tokenizer.encode_batch` — per-text indexing into
the failure is not surfaced because the upstream API doesn't expose it. Callers needing per-text
diagnostics can call `embed` per item.

## 11. Engineering robustness

### 11.1 ort version coupling

`ort = "2.0.0-rc.12"` (caret) matches silero/soundevents byte-identically. Bumping requires a coordinated
change across the trio.

### 11.2 Model file integrity

The README publishes SHA256s (also in `tests/fixtures/MODELS.md` per §3.3). Mismatched files produce
undefined results — typically `Error::SessionSchema` or `Error::UnexpectedTensorShape` (caught by §7.3.1
shape validation) or, worse, silent embedding drift.

### 11.3 Quantization variant compatibility

textclap 0.1.0 is verified against the **INT8-quantized** export. Tolerances against the **Python int8
reference**:

| Variant  | Audio embedding tolerance vs Python int8 reference | Notes              |
|----------|----------------------------------------------------|--------------------|
| int8     | < 5e-4 (verified target — see §12.2)               | This release       |
| fp16     | likely < 5e-3                                      | Not verified       |
| fp32     | likely < 1e-2 (across-quantization)                | Not verified       |

The fp32-vs-int8 column entry is intentionally looser than int8-vs-int8 — the comparison is *across*
quantization regimes. fp32-vs-fp32 (same-precision goldens regenerated against the fp32 export) would be
tighter.

### 11.4 Cold-start latency

`Clap::warmup` runs a single dummy forward through each encoder — 480 000 samples of silence for audio,
the `warmup_text` string from `golden_params.json` for text. The warmup string is sized at maintenance
time (§3.1) to land at ≥80 BPE tokens, so typical user search queries (5–15 tokens) and edge-case longer
queries up through CLAP's `max_length` (~77) won't reallocate token scratch on the first real call. The
candidate is the pangram repeated until it crosses the threshold; the maintainer measures the actual
post-tokenizer count and records the chosen string as `warmup_text` in `golden_params.json`.

Sizes scratch for batch size 1; the first batched call (e.g. an offline backfill via `embed_batch(N=32)`)
still grows scratch once. Magnitude: audio mel scratch grows from ~250 KB (N=1) to ~8 MB (N=32 batch),
one-time, on the first big call. **Wall-clock impact:** typically 5–50 ms of one-time latency on the
first batched call after warmup, accounting for the scratch growth itself plus ORT operator
re-specialization for the new tensor shape. Amortized to zero across subsequent batches at the same N.
`warmup_for_batch(audio_n, text_n)` is a §14 follow-up if this latency stutter ever becomes
load-bearing.

### 11.5 Test determinism and platform variance

Tests construct `ort::Session` instances directly with `intra_op_threads=1` (sibling convention places
this configuration outside the crate's API; see `from_ort_session`). With single-threaded ORT on the same
process and thread, reduce-order non-determinism is eliminated; same-input → same-output is bit-exact.

**Cross-platform variance is intrinsic and non-trivial.** ORT's CPU EP differs across OSes (MLAS on
Linux/Windows, Accelerate on macOS); FMA fusion and vectorization differ between x86 and ARM, and between
AVX-512 and AVX2. Embedding values differ at the ULP level across runners. The §12.2 tolerance table's
"cross-platform reproducibility" row absorbs this.

**Bench warmup.** `benches/` Criterion harnesses each call `warmup()` in their setup closure before the
`iter` loop.

### 11.6 Model attribution and license compliance

The crate ships with no model files. Downstream users redistributing model files take on the upstream
attribution responsibilities:
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
  - `extract_into` panics on out-len mismatch (length contract).
- **`audio.rs`:**
  - `embed(&[])` → `EmptyAudio { clip_index: None }`.
  - `embed(&[0.0; 480_001])` → `AudioTooLong`.
  - `embed(&[0.0; 480_000])` succeeds.
  - `embed(&[f32::NAN, ...])` → `NonFiniteAudio { clip_index: None, sample_index: 0 }`. Same for ±Inf.
  - `embed_batch` with **uneven-length** clips succeeds (auto-pad).
  - `embed_batch` with one empty clip → `EmptyAudio { clip_index: Some(i) }`.
  - `embed_batch` with one non-finite clip → `NonFiniteAudio { clip_index: Some(i), sample_index }`.
  - Empty batch slice → empty `Vec`.
  - Chunked windowing offsets and chunk counts, including trailing-chunk-skip rule.
  - `ChunkingOptions { hop > window }` → `ChunkingConfig` carrying all three values.
  - Scratch lifecycle stress test: small batch then large batch in sequence; results match
    independently-computed singles within §12.2 tolerance. Liveness check; the borrow-checker
    enforcement of §7.3.1 is the structural guarantee.
- **`text.rs`:**
  - `EmptyInput` for empty `&str` and empty string at index `i` in batch.
  - Empty batch slice → empty `Vec`.
  - `from_files` with `Padding::Fixed` JSON: tokenizer is rewritten to `BatchLongest`.
  - `from_ort_session` with caller-supplied tokenizer using `Padding::Fixed` → `PaddingFixedRejected`.
  - Tokenizer with no pad config and no `<pad>` token → `NoPadToken`.
  - `from_ort_session` with mismatched session schema → `SessionSchema`.
- **`clap.rs`:**
  - `classify(&samples, &[], k)` → `Ok(vec![])`.
  - `classify(&samples, &labels, 0)` → `Ok(vec![])`.
  - `classify(&samples, &labels, 1000)` → returns all `labels.len()` entries.
  - `classify` returns top-k descending; `classify_all` returns all labels.
  - Stable ordering on tied scores.
- **`options.rs`:**
  - Builder methods round-trip through accessors for both `with_*` and `set_*`.
  - `Options::default() == Options::new()`.
- **`Embedding`:**
  - `from_slice_normalizing` always produces unit-norm output for non-zero input.
  - `from_slice_normalizing` rejects all-zero input → `EmbeddingZero`.
  - `try_from_unit_slice` rejects wrong lengths (`EmbeddingDimMismatch`) and non-unit-norm input
    (`EmbeddingNotUnitNorm`) at the 5e-5 norm² budget.
  - `dot ≈ cosine` for unit inputs (within fp32 ULP).
  - `is_close(&self, &self, 0.0)` returns true.
  - `is_close_cosine(&self, &self, 0.0)` returns true.
  - **Cancellation safety:** Construct `a = from_slice_normalizing(&[1.0; 512])` and
    `b = from_slice_normalizing(&[(1.0 + 1e-9); 512])` (both end up unit-norm by construction). Assert
    `a.is_close_cosine(&b, 1e-12)` returns true. The squared-distance implementation handles fp32
    cancellation that the algebraically-equivalent `1 − dot(a, b)` would mask.
  - `to_vec` and `as_slice` are byte-equal.
  - **No `pub const DIM`** — compile-time test that `Embedding::DIM` doesn't resolve.
  - **Custom `Debug` output** does not contain 512 floats.
  - **No `PartialEq` derived** — compile-time test.

### 12.2 Integration test (`tests/clap_integration.rs`)

Gated on `TEXTCLAP_MODELS_DIR` env var (skip with `eprintln!` if unset).

Tests construct `ort::Session` instances with `intra_op_threads=1` (sibling-convention path: configure
outside textclap, pass via `from_ort_session`).

**Tolerance reference table.** For unit vectors of dim 512, the worst-case relationship between max-abs
drift `ε` and cosine drift `1 − cos` is `1 − cos ≤ 512 · ε² / 2 = 256 · ε²`. The cosine column below is
sized at ~1.5–2× the worst-case bound for headroom. Use these values when calling
`is_close` / `is_close_cosine` in tests:

| Comparison                                      | `is_close` (max-abs) | `is_close_cosine` (1−cos) | Origin of cosine value          |
|-------------------------------------------------|----------------------|----------------------------|---------------------------------|
| Rust audio embedding vs golden (int8 vs int8)   | 5e-4                 | 1e-4                       | bound 6.4e-5 + ~1.5× headroom    |
| Rust text embedding vs golden (int8 vs int8)    | 1e-5                 | 5e-8                       | bound 2.6e-8 + ~2× headroom      |
| Cross-quantization audio (fp32 vs int8 ref)     | 1e-2                 | 5e-2                       | bound 2.6e-2 + ~2× headroom      |
| Cross-platform reproducibility (CPU EP variance)| 1e-5 (preliminary)   | 5e-8 (preliminary)         | inherited from text-row math; int8-on-MLAS-vs-Accelerate variance has not been measured — widen if real-run measurement supports it |

Cross-platform reproducibility refers to comparing embeddings produced on different OSes / CPU families
(e.g. Linux MLAS vs macOS Accelerate, x86 vs ARM). On the *same* OS / hardware / thread / process,
embeddings are bit-exact and could be compared at `tol = 0`.

Assertions (use `is_close` form):

| Check                                                          | Tolerance (`max_abs_diff`)         |
|----------------------------------------------------------------|-------------------------------------|
| Rust mel features vs `golden_mel.npy`                          | < 1e-4                              |
| Rust audio embedding vs `golden_audio_emb.npy`                 | < 5e-4 (post-L2)                    |
| Rust text embeddings vs `golden_text_embs.npy`                 | < 1e-5                              |
| `classify_all` discrimination check (see below)                | structural                          |

**Tolerance origin (audio).** Mel drift up to 1e-4 propagates through HTSAT (~14 transformer blocks) with
typical 5–50× contractive amplification. L2 normalization tightens this. **5e-4 is the opening tolerance**;
calibrated downward only after real-run measurement supports it. Per-OS budget tables may be needed.

**Tolerance origin (text).** Integer token ids → no upstream drift; 1e-5 catches RoBERTa wiring bugs.

**`try_from_unit_slice`'s norm² budget (5e-5)** is intentionally tighter than the audio-embedding budget —
goldens are L2-normalized in the same float order Rust uses, and stored vectors that haven't crossed
quantization or platform boundaries should round-trip exactly.

**Discrimination check:** `classify_all` is run with the labels
`["a dog barking", "rain", "music", "silence", "door creaking"]`. The test asserts:
1. `"a dog barking"` ranks in the top 2.
2. `score("a dog barking") - score("music") > 0.05` (irrelevant baseline).

### 12.3 Doctests

Every public function on `Clap`, `AudioEncoder`, `TextEncoder`, `Embedding` ships a runnable rustdoc
example. `Embedding` examples are runnable; encoder examples use `# no_run`.

### 12.4 Benches (`benches/`)

Three Criterion benchmarks. Each `setup` closure calls `warmup()` before `iter`:
- `bench_mel.rs` — `MelExtractor::extract_into` on a 10 s buffer.
- `bench_audio_encode.rs` — full encode for batch sizes 1, 4, 8.
- `bench_text_encode.rs` — text encode for batch sizes 1, 8, 32.

### 12.5 CI

- **rustfmt** (Linux)
- **clippy** (Linux/macOS/Windows × default features × all features)
- **build + test** matrix (Linux/macOS/Windows × stable Rust)
- **doctest** (Linux, all features)
- **coverage** (tarpaulin → codecov, Linux)
- **integration job** (Linux only) — fetches model files into a runner cache, sets `TEXTCLAP_MODELS_DIR`,
  runs `cargo test --test clap_integration`.

## 13. Migration from current template

textclap is currently the bare `al8n/template-rs` scaffold.

### Replace
- `Cargo.toml` (identity, deps, dev-deps, features, MSRV, version 0.1.0, caret-pinned `ort`, `examples`
  marked `publish = false`, `[lints.rust]` block, `[package.metadata.docs.rs]`, `include = [...]`
  whitelist excluding `tests/fixtures/`).
- `README.md` — purpose, install, quick-start showing **both deployment patterns**:

  ```rust
  // Indexing-only worker (saves ~120 MB resident vs Clap):
  fn run_indexer() -> Result<(), Box<dyn std::error::Error>> {
      let mut audio = AudioEncoder::from_file("audio_model_quantized.onnx", Options::new())?;
      audio.warmup()?;
      loop {
          let pcm = decoder.next_10s_at_48khz()?;            // caller-supplied; arbitrary error
          let emb = audio.embed(&pcm)?;                       // textclap::Error
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
  lancedb errors without writing `From` impls — idiomatic for examples and binaries; in library code
  callers might layer their own concrete error type.

  Plus model-acquisition note pointing to HuggingFace **with SHA256s and HF revision pin**, the warning
  that `tokenizer.json` must come from the same Xenova export, model-attribution-on-downstream section
  (§11.6), ort-coupling note, license, the lancedb integration snippet below, **the §1.1
  indexing-vs-query pipeline diagram**, §1.2 use-cases-beyond-live mapping, §1.3 domain-of-training and
  short-clip caveats, thread-tuning note pointing users to `from_ort_session`.
- `src/lib.rs` (keep crate-level lints; replace body with module decls and the explicit re-exports
  listed in §4).
- `tests/foo.rs`, `benches/foo.rs`, `examples/foo.rs` → delete; replace per §4.
- `CHANGELOG.md` → reset to Keep-a-Changelog stub starting at `[0.1.0]`.

### Keep
- `build.rs` — copied verbatim from sibling crates.
- License files (update copyright holder/year).
- `.github/workflows/` skeleton, with deletions per §12.5.

### Add
- `src/error.rs`, `src/options.rs`, `src/mel.rs`, `src/audio.rs`, `src/text.rs`, `src/clap.rs`.
- `tests/fixtures/` contents per §3 / §12.2.
- `examples/audio_window_to_clap.rs` — the §1.1 indexing path: native-rate frames → `rubato` resample to
  48 kHz → buffer 10 s → `AudioEncoder::embed` → 512-dim `Embedding` → push to a stubbed lancedb writer.
- `examples/index_and_search.rs` — sequential demo: embed one fixed window first, then run a single
  query against a stubbed table. Indexing is continuous and query is on-demand in real deployments;
  this example is explicitly a sequential pedagogical run, not a co-running daemon.
- This spec under `docs/superpowers/specs/`.

### lancedb integration snippet (for README)

```rust
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use textclap::Embedding;

// ── Indexing side: every 10 s of buffered 48 kHz mono audio ──
let embedding: Embedding = audio.embed(&pcm_48khz_mono_10s)?;
let dim = embedding.as_slice().len() as i32;          // dimension-agnostic
let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
builder.values().append_slice(embedding.as_slice()); // copies into Arrow's MutableBuffer
builder.append(true);
// (build a record batch with this column + ts_start + ts_end + metadata → table.add(...))

// ── Query side: when a user submits a text search ──
// The text encoder runs ONCE per query, not per indexed item.
let query: Embedding = text.embed("dog barking near a door")?;
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
- A second chunking-aggregation strategy (max, attention pooling, mean-of-logits) if a real CLAP use case
  demonstrates value. Adding it brings back the `Aggregation` enum + `ChunkingOptions::with_aggregation`.
- A `pad_mode: silence` option in `ChunkingOptions` to replace repeat-pad with zero-pad for short clips
  in the chunking path (addresses §1.3 periodicity-artifact concern). A per-call override on `embed` is a
  separate follow-up — not bundled here so the option's scope stays clear.
- An optional **single-window LAION-reference mode** for offline clip embedding — single-window
  rand_trunc with a caller-provided RNG seed for cross-tool retrieval interop. Today's workaround:
  callers do `embed(&samples[..480_000.min(len)])` themselves.
- A doctest on `Embedding::cosine` showing the lancedb round-trip specifically.
- `tracing` feature for service-tier observability.
- `try_reserve_exact` on scratch resizes to surface OOM as `Error::ScratchAlloc` instead of panic.
- `Options::with_truncation_warn_threshold(usize)` to log when text inputs hit the silent truncation cap.
- **Pre-allocation of scratch to a fixed `MAX_BATCH × 64 × T` at construction**, eliminating the
  resize-during-inference class structurally instead of relying on the §7.3.1 borrow-checker pattern.
  Trade: API rigidity. Adopt if profiling or fuzz-style stress testing surfaces resize-related issues.
- **In-flight cancellation.** ORT 2.x exposes `RunOptions::new()?` and `Arc<RunOptions>::terminate()`.
  Implementing this would require threading an `Arc<RunOptions>` through every `embed*` call and exposing
  a `CancelHandle` type. Deferred — feature decision, not infeasibility.
- An AddressSanitizer CI job — Miri can't cross the FFI boundary, but ASan does. Only worth adding if the
  §7.3.1 contract ever needs empirical re-validation beyond what the borrow checker enforces statically.
