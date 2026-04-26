# textclap — CLAP Inference Library Design

**Status:** Approved (revision 17, Phase A backfill applied — fixture-derived values land in §8.1 / §8.2 / §9.2 / §11.4)
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
3. Runs the extractor; saves resulting `[T, 64]` mel features to `golden_mel.npy` (time-major
   layout: one row per frame, 64 mel values per row).
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

- **Audio input shape:** `[batch, 1, T, 64]` *vs* `[batch, T, 64]` (channel dim present?).
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
import librosa
import onnxruntime as ort
from transformers import ClapFeatureExtractor, ClapModel

# Setup — assumes regen_golden.py / inspect_onnx.py share a common preamble.
extractor    = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
pt_model     = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
audio_session = ort.InferenceSession("audio_model_quantized.onnx")
audio, _     = librosa.load("tests/fixtures/sample.wav", sr=48000, mono=True)

# Names recorded by the static-inspection pass earlier in this script.
onnx_io = json.load(open("tests/fixtures/golden_onnx_io.json"))
AUDIO_INPUT_NAME  = onnx_io["audio_input_name"]    # e.g. "input_features"
AUDIO_OUTPUT_NAME = onnx_io["audio_output_name"]   # e.g. "audio_embeds"

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

# torch.no_grad() is required: ClapModel parameters have requires_grad=True even in
# eval() mode, so the output tensor inherits this and .numpy() raises without it.
with torch.no_grad():
    pt_emb = pt_model.get_audio_features(**features)    # [1, 512] fp32
    pt_emb = F.normalize(pt_emb, dim=-1)                # robust to any batch size

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
3. **Spec-update commit.** Every TBD value in this spec is replaced with the resolved fixture-derived
   value. Complete checklist (deletions of "TBD" markers and "expected" qualifiers expected throughout):
   - §8.1 mel parameter table — `T` (frame count), `top_db`, HTSAT-input-norm action (`none` /
     `global_mean_std` plus mean/std). If the action is `global_mean_std`, the mean/std values land as
     `pub(crate) const HTSAT_INPUT_MEAN: f32` / `HTSAT_INPUT_STD: f32` in `mel.rs` (or are inlined into
     the mel pipeline depending on the §3.2 decision). If `none`, no consts are added.
   - §8.2 — `AUDIO_INPUT_NAME`, `AUDIO_OUTPUT_NAME`, audio input shape (3-D vs 4-D channel dim).
   - §9.2 — `TEXT_INPUT_IDS_NAME`, `TEXT_ATTENTION_MASK_NAME`, optional `TEXT_POSITION_IDS_NAME`,
     `TEXT_OUTPUT_NAME`, plus the §7.4 attention-mask / position_ids description (inlined vs
     externalized).
   - §7.5 — `AUDIO_OUTPUT_IS_UNIT_NORM` and `TEXT_OUTPUT_IS_UNIT_NORM` const values (the booleans
     determined by §3.2's static graph inspection of the ONNX-output L2-normalize pattern).
   - §11.4 — `warmup_text` literal string and the script-measured BPE token count.
   - §12.2 tolerance table — adjusted if real-run measurement diverges from the §12.2 derived bounds.
   - §3.4 step 4 readiness check: confirm `ort::session::builder::GraphOptimizationLevel: Copy` at the
     pinned RC (referenced by §7.7's `pub const fn with_graph_optimization_level`).
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
│   ├── mel.rs                       # MelExtractor + T_FRAMES const (§8.2)
│   ├── audio.rs                     # AudioEncoder + AUDIO_INPUT_NAME / AUDIO_OUTPUT_NAME /
│   │                                # AUDIO_OUTPUT_IS_UNIT_NORM consts (§8.2)
│   ├── text.rs                      # TextEncoder + TEXT_INPUT_IDS_NAME / TEXT_OUTPUT_NAME /
│   │                                # TEXT_OUTPUT_IS_UNIT_NORM consts (§9.2; attention_mask and
│   │                                # position_ids are inlined into the ONNX graph, not externalized)
│   └── clap.rs                      # Clap, Embedding, LabeledScore, LabeledScoreOwned + NORM_BUDGET (§7.5)
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
  `unexpected_cfgs = { level = "warn", check-cfg = ['cfg(all_tests)', 'cfg(tarpaulin)'] }` (order
  matches silero/soundevents).
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
pub struct AudioEncoder  { /* ort::Session + MelExtractor + encoder-owned scratch.
                             audio_output_is_unit_norm is a module-private compile-time const
                             (AUDIO_OUTPUT_IS_UNIT_NORM in audio.rs), backfilled by §3.4 step 4 from
                             golden_onnx_io.json's audio_output_is_unit_norm flag. It is queried
                             at every embed* call and selects the centroid vs spherical-mean
                             aggregation in embed_chunked and the trust-vs-renormalize path in
                             embed/embed_batch. Because it's a const, the unused branch is
                             dead-code-eliminated by the optimizer. */ }
pub struct TextEncoder   { /* ort::Session + Tokenizer + cached pad_id + encoder-owned scratch.
                             text_output_is_unit_norm is the analogous TEXT_OUTPUT_IS_UNIT_NORM
                             const in text.rs. */ }

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

    /// Wraps a pre-built ORT session.
    ///
    /// **Two distinct purposes:**
    /// (a) **Thread tuning / EP selection (supported).** Supply a Session
    ///     built with custom `intra_op_threads`, `inter_op_threads`, or a
    ///     different execution provider (CUDA, CoreML). The wrapped
    ///     session must still match the ONNX schema in
    ///     `golden_onnx_io.json` — name and dtype are checked at
    ///     construction; mismatches return `Error::SessionSchema`.
    /// (b) **Model-variant swapping (NOT directly supported).** Supplying
    ///     a different ONNX export (e.g., fp16 instead of int8) requires
    ///     regenerating the `AUDIO_OUTPUT_IS_UNIT_NORM` const via §3.2
    ///     and recompiling textclap. The schema check is necessary but
    ///     NOT sufficient: a wrongly-baked const causes silent corruption
    ///     of stored embeddings (the §8.2 trust-path release-mode guard
    ///     catches this at write-time, returning `EmbeddingNotUnitNorm`).
    ///
    /// `AUDIO_OUTPUT_IS_UNIT_NORM` is a compile-time const baked in during
    /// §3.4 step 4. All three AudioEncoder constructors share the same const.
    /// Regardless of which constructor is used, the resulting `AudioEncoder`
    /// is `Send` but `!Sync` — thread-per-core deployment applies uniformly.
    ///
    /// **Detection asymmetry.** The two const-mismatch failure modes are
    /// not symmetric:
    /// - **const says `false` but ONNX is unit-norm.** Encoder runs the
    ///   L2-normalize step on already-unit-norm vectors (idempotent identity
    ///   to fp32 ULP). Harmless — wastes ~200 ns per call but produces
    ///   correct output. No corruption.
    /// - **const says `true` but ONNX is not unit-norm.** Encoder takes the
    ///   trust path; the release-mode `NORM_BUDGET` guard rejects the output
    ///   with `EmbeddingNotUnitNorm`. Loud, at write-time, before the
    ///   invalid vector reaches lancedb.
    /// The asymmetry is by design: false-says-true (the dangerous direction)
    /// always errors loudly; false-says-false (the safe direction) wastes
    /// CPU but never corrupts data.
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
    /// this). Embeddings produced via embed_chunked therefore live in a
    /// region of CLAP-audio space slightly displaced from LAION-reference
    /// embeddings.
    ///
    /// The interop rule is "self-consistent within textclap": if you index
    /// audio with textclap (single-window embed *or* embed_chunked) and
    /// query with textclap (TextEncoder::embed), cosine search works. If
    /// you index with textclap but query with another CLAP tool's text
    /// encoder — or vice versa — scores can be systematically degraded
    /// because the two sides used different conventions to land in the
    /// shared 512-dim space. **Don't mix toolchains across the index/query
    /// boundary**: index columns produced by textclap should be queried
    /// with textclap.
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

**The Xenova `clap-htsat-unfused` export takes only `input_ids` as an external input** — RoBERTa's
attention masking and position-id derivation are inlined into the graph (using `pad_id=1` internally).
The tokenizer still needs `pad_id` for `BatchLongest` padding so `encode_batch` produces uniform
sequence lengths, but no `attention_mask` or `position_ids` tensor is sent to the model. If a future
export externalizes either, regenerate `golden_onnx_io.json` and update §9.2 accordingly.

### 7.5 `Embedding`

```rust
impl Embedding {
    pub const fn dim(&self) -> usize;            // 512 for 0.1.0; runtime-queryable, future-proof

    // Borrow-only access — supports append_slice into Arrow's MutableBuffer.
    pub fn as_slice(&self) -> &[f32];

    // Owned conversion.
    pub fn to_vec(&self) -> Vec<f32>;

    /// Reconstruct from a stored unit vector. Validates length AND norm
    /// (release-mode check: `(norm² − 1).abs() ≤ NORM_BUDGET`, where
    /// `pub(crate) const NORM_BUDGET: f32 = 1e-4` is defined in `clap.rs`
    /// alongside `Embedding` and referenced from `audio.rs` / `text.rs`.
    /// Same const-at-module-top convention silero uses for its tensor
    /// names (silero `session.rs:12` keeps them module-private; NORM_BUDGET
    /// needs `pub(crate)` for cross-module use, but the structural pattern
    /// is identical).
    ///
    /// **Budget rationale.** Arrow IPC and Parquet for f32 are byte-exact —
    /// no quantization or recomputation in transit. The realistic
    /// in-scope drift source is **summation-order divergence**: NumPy's
    /// `np.linalg.norm` calls BLAS `snrm2` (which may use pairwise or
    /// scaled-sum algorithms), while Rust's `‖x‖` is typically a sequential
    /// fma loop over 512 components. Both are "correct" L2 normalizations
    /// to fp32, but they can disagree by up to ~512·ulp(1) ≈ 6.1e-5 in the
    /// worst case. **The budget ships at 1e-4** — comfortably above the
    /// worst-case bound, with margin for the BLAS-implementation variation
    /// a maintainer might hit when regenerating goldens on a different
    /// runtime (glibc snrm2 vs MKL vs OpenBLAS vs Apple Accelerate). 1e-4
    /// still rejects truncation (10× wrong norm² is ~1.0 deviation),
    /// byte corruption (drift ≫ 1e-4), and fp16 storage round-trip
    /// (~2e-3) by orders of magnitude — the actual bug classes the budget
    /// is meant to catch. Tighten to 5e-5 in a future minor release if
    /// real-run measurement supports it; rolling back is a low-stakes
    /// spec edit, while rolling forward from a 5e-5 production failure
    /// would be a higher-stakes patch.
    ///
    /// This is NOT a cross-platform reproducibility check (use
    /// is_close_cosine for that). Truncation is caught earlier by
    /// EmbeddingDimMismatch; fp16 storage round-trip is OUT OF SCOPE
    /// (fp16's ulp(1.0) ≈ 9.77e-4 makes the check fail; users converting
    /// through fp16 should use from_slice_normalizing).
    pub fn try_from_unit_slice(s: &[f32]) -> Result<Self>;

    /// Construct from any non-zero slice; always re-normalizes to unit length
    /// (idempotent for input that's already unit-norm). Validates length,
    /// rejects all-zero input via Error::EmbeddingZero, and rejects any
    /// non-finite component (NaN, ±Inf) via Error::NonFiniteEmbedding.
    /// The audio path catches non-finite samples earlier via
    /// Error::NonFiniteAudio; this is the safety net for direct-construction
    /// paths (e.g. read-back from a corrupt lancedb cell).
    ///
    /// **Cost.** ~100 ns over 512 components (finiteness scan + L2 norm).
    /// For bulk hot-path import where upstream guarantees finiteness and
    /// unit-norm, prefer `try_from_unit_slice` (skips the renormalization
    /// pass; just validates length and norm).
    pub fn from_slice_normalizing(s: &[f32]) -> Result<Self>;

    // Similarity (== for unit vectors, modulo fp32 ULP).
    pub fn dot(&self, other: &Embedding) -> f32;
    pub fn cosine(&self, other: &Embedding) -> f32;

    /// Approximate equality test — raw float drift, max-abs metric.
    /// Returns true if `(self − other).max_abs() ≤ tol` (note the
    /// inclusive ≤ — this is a deliberate choice so that
    /// `a.is_close(&a, 0.0)` always returns true, which matches how
    /// users write self-equality tests).
    /// Use for tests checking implementation determinism.
    /// Recommended tolerance values: §12.2 reference table.
    pub fn is_close(&self, other: &Embedding, tol: f32) -> bool;

    /// Approximate equality test — semantic (cosine) metric.
    /// Returns true if `1 − self.cosine(other) ≤ tol` (inclusive ≤
    /// for the same self-equality reason).
    ///
    /// **Implementation.** Computed as `0.5 · ‖self − other‖₂² ≤ tol` for
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

**Naming.** `Options` follows soundevents' unqualified naming over silero's `SessionOptions`. Silero
needs the `Session` qualifier to disambiguate from `SpeechOptions` (its segmentation tuning struct).
textclap has no second options type, so unqualified `Options` is unambiguous and shorter at call sites.

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
    /// exactly `T * 64` (T from golden_params.json; time-major layout: one row
    /// per frame, 64 mel values per row). Asserts on length mismatch.
    pub(crate) fn extract_into(&mut self, samples: &[f32], out: &mut [f32]) -> Result<()>;
}
```

The `&mut [f32]` interface (rather than `&mut Vec<f32>`) is what enables the §7.3.1 lifecycle pattern:
`AudioEncoder` resizes its scratch `Vec` once per call, then hands the extractor sub-slices for each
batch row.

Parameters (recorded values from `golden_params.json` per §3.1; only the truncation row is intentionally
chosen differently from HF's `rand_trunc`):

| Parameter            | Value                                             |
|----------------------|---------------------------------------------------|
| Sample rate          | 48 000 Hz                                         |
| Target samples       | 480 000 (10 s)                                    |
| `n_fft`              | 1024                                              |
| Hop length           | 480                                               |
| Window               | **Hann, periodic, length 1024**                   |
| Frame count `T`      | **1001** (HF extractor uses center=True padding)  |
| Mel bins             | 64                                                |
| Mel scale            | **Slaney**                                        |
| Filterbank norm      | **Slaney**                                        |
| Frequency range      | 50 – 14 000 Hz                                    |
| Power spectrum       | `|X|²`                                            |
| Mel→dB transform     | **`10 · log10(max(amin, x))` with `amin = 1e-10`, `ref = 1.0`, `top_db = null` (no clipping); applied exactly once after the mel filterbank** |
| Padding mode         | repeatpad                                         |
| Truncation mode      | head (deterministic; intentionally differs from HF rand_trunc) |
| HTSAT input norm     | **none** (functional verification at maintenance time chose 'none' with drift 1.10e-2 in yellow zone — see `golden_params.json["htsat_norm_drift"]`) |

State allocated once in `new()`, owned by the `MelExtractor`:
- Hann window (`Vec<f32>`, len 1024, periodic convention).
- Mel filterbank (`Vec<f32>`, len 64 × 513).
- `RealFftPlanner<f32>` instance.

The ONNX input tensor shape `[N, 1, T, 64]` is built as a *view* over the mel feature scratch — the channel
dim is added at tensor-construction time with no data movement, and the underlying `Vec<f32>` length
stays `N · T · 64`.

**Layout note.** The HF `ClapFeatureExtractor` produces time-major `[T, 64]` layout. mel.rs writes one
row per frame (64 mel values per row); the ONNX audio model's `[batch, 1, T, 64]` input matches this
directly. Output buffer shape matches `T` from `golden_params.json` — only the axis order is
time-major (frames are the outer dimension, mel bins inner).

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
// Backfilled per §3.4: step 3 lands these names in this spec; step 4 lands
// them as Rust constants in audio.rs. Module-private (matches silero
// session.rs:12) — referenced only inside audio.rs.
const AUDIO_INPUT_NAME:  &str = "input_features";
const AUDIO_OUTPUT_NAME: &str = "audio_embeds";
```

`T` (the mel time dimension, recorded in `golden_params.json` per §3.1) is a `pub(crate) const T_FRAMES`
defined in `mel.rs` and re-used here. It is a compile-time constant after §3.4 step 3 backfills the value;
before backfill the §4 skeleton uses a placeholder.

**Internal helper.** A single `pub(crate)` method runs the full forward pipeline (mel → ONNX) for any
non-empty batch and writes raw model outputs into a caller-provided buffer. **Contract: prior contents of
`out` are dropped — the function clears and resizes the buffer to `clips.len()` before writing.**

```rust
/// Compute the audio model's raw projection outputs. These are
/// un-normalized 512-dim vectors if AUDIO_OUTPUT_IS_UNIT_NORM is false,
/// or already-unit-norm vectors if it is true. Callers (the public
/// embed*/embed_batch/embed_chunked methods) handle any subsequent
/// L2 normalization or release-mode unit-norm guard themselves —
/// the trust-path branch lives in the public methods, not here.
///
/// `out` is cleared on entry; capacity is reserved for clips.len() entries
/// and one row is pushed per clip. Prior contents are dropped — this is the
/// *clear-and-fill* contract, not *append*.
///
/// The chunked path's per-call accumulator (the Vec<[f32; 512]> that
/// gathers all chunks before aggregation) is a *separate* Vec from the
/// encoder-owned proj_scratch the helper writes into; the chunked path
/// copies from proj_scratch into the accumulator after each batch group.
/// The accumulator is freshly heap-allocated on each embed_chunked call:
/// up to `ceil(L / hop) · 2 KB` total (per the §8.2 allocation budget).
/// At the default hop = window = 480_000 samples, 60 s of audio → 6 chunks
/// → ~12 KB. Acceptable because embed_chunked is the offline single-clip
/// summarization path (§1.2), not the live indexing hot path — the live
/// path uses embed and the encoder-owned proj_scratch directly.
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
let row_len = T_FRAMES * 64;                                       // compile-time const after §3.4
let total = n * row_len;

self.mel_scratch.clear();
self.mel_scratch.resize(total, 0.0);                               // grow + zero-fill.
// The zero-fill writes ~8 MB on the first N=32 batch (amortized to zero on
// subsequent same-N batches; the buffer is reused). It is deliberately wasted
// work — extract_into overwrites every cell immediately. Avoiding it would need
// Vec::set_len, which is forbidden by #![forbid(unsafe_code)]. The slice-indexing
// pattern below requires the Vec to be at full length, so resize is the only
// safe path. The cost is the price of forbid(unsafe_code).
for (i, clip) in clips.iter().enumerate() {
    self.mel.extract_into(
        clip,
        &mut self.mel_scratch[i * row_len .. (i + 1) * row_len],   // &mut [f32]
    )?;
}

let input = TensorRef::from_array_view((
    [n, 1usize, T_FRAMES, 64],                                     // shape: tuple of usize, NOT &[i64]
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
5. Take `self.proj_scratch[0]`. If the `AUDIO_OUTPUT_IS_UNIT_NORM` const (§7.1, backfilled by §3.4
   step 4 from `golden_onnx_io.json`) is `true`, construct via the trusted path **with a release-mode
   unit-norm guard**: compute `(‖x‖² − 1).abs()` and return `Error::EmbeddingNotUnitNorm` if it
   exceeds `NORM_BUDGET` (the same 1e-4 budget `try_from_unit_slice` uses; defined once as
   `pub(crate) const NORM_BUDGET: f32 = 1e-4` in `clap.rs` alongside `Embedding`, and referenced
   from all five call sites — `try_from_unit_slice`, audio `embed`/`embed_batch` trust paths,
   text `embed`/`embed_batch` trust paths). The guard is ~512 fma + 1 sub + 1 cmp ≈ 200 ns,
   negligible against ~50 ms ONNX
   inference. It catches the silent-corruption failure mode where a wrongly-baked const (e.g. user
   supplied a different ONNX without rerunning §3.2) would otherwise ship invalid Embeddings to
   lancedb that user `try_from_unit_slice` would later reject on every read-back. Otherwise
   (`AUDIO_OUTPUT_IS_UNIT_NORM == false`) L2-normalize and wrap as `Embedding`.

   **Why 1e-4 and not tighter (e.g. 5e-5).** The realistic failure mode is "wrongly-baked const" —
   a user supplied a different ONNX export through `from_ort_session` without rerunning §3.2. In
   that case the model output is not unit-norm at all (norm² is ≫ 1 if the L2-normalize op was
   absent, or some arbitrary value if a different model entirely). 1e-4 catches all such cases by
   many orders of magnitude; a tighter 5e-5 guard wouldn't catch a different bug class, just risk
   false-positives against legitimate cross-summation-order outputs. Naming consistency with
   `try_from_unit_slice` (shared `NORM_BUDGET`) wins over speculative tightening; §14 tracks the
   future-tightening task once telemetry confirms safety.

**`embed_batch(clips)`:**
1. Empty slice → `Ok(Vec::new())`.
2. For each clip `i`: empty → `Error::EmptyAudio { clip_index: Some(i) }`; too-long → `AudioTooLong`;
   finiteness scan → `Error::NonFiniteAudio { clip_index: Some(i), sample_index }`.
3. Call `embed_projections_batched(clips, &mut self.proj_scratch)`.
4. Row-by-row construct `Embedding`s — same branch as single-clip `embed`: if `AUDIO_OUTPUT_IS_UNIT_NORM`
   is `true`, use the trusted path with the release-mode unit-norm guard described below; otherwise
   L2-normalize each row.

**`embed_chunked(samples, opts)`:**
1. `samples.is_empty()` → `Error::EmptyAudio { clip_index: None }`.
2. Validate `opts` per §7.8 → otherwise `Error::ChunkingConfig { window_samples, hop_samples, batch_size }`.
3. Finiteness scan over `samples`.
4. Compute chunk offsets `0, hop, 2·hop, …` while `offset < samples.len()`. Trailing chunks shorter than
   `window_samples / 4` are skipped (§7.8) unless the input itself is shorter than `window/4`.
5. For each group of `batch_size` chunks: call `embed_projections_batched(group, &mut tmp_proj)`, append
   raw outputs to a per-call `Vec<[f32; 512]>`.
6. **Aggregate** — branch selected by `AUDIO_OUTPUT_IS_UNIT_NORM` (compile-time const, dead-branch
   eliminated):
   - `false` → **Centroid path**: component-wise mean of raw projections, then L2-normalize → `Embedding`.
   - `true` → **Spherical-mean path**: component-wise mean of unit vectors, then L2-normalize → `Embedding`.

   Single-chunk case skips aggregation regardless of branch.

   Both branches end with the same finalization: a local L2-normalize ensures unit-norm by construction,
   and the produced `Embedding` is wrapped via the same trust-path / non-trust-path branch that single-clip
   `embed` uses (step 5). This is **not a sixth NORM_BUDGET site** — `embed_chunked` reuses the audio
   `embed` finalization and inherits site 1's guard, which catches any rounding pathology consistent
   with the other five sites.

**Allocation budget per call (after warmup):**
- `embed`: only the output `Embedding`.
- `embed_batch(N)`: `Vec<Embedding>` of N entries; mel scratch grows once per new max size.
- `embed_chunked(L, batch=B)`: per-call `Vec<[f32; 512]>` of *up to* `ceil(L / hop)` entries (trailing
  partial chunks shorter than `window/4` are skipped per §7.8).

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

> §3.2 backfill: tensor names and dtypes (`input_ids` only — `attention_mask` and `position_ids`
> are inlined into the graph, not externalized) confirmed by `inspect_onnx.py` before §3.4 step 3.

```rust
// Backfilled per §3.4: step 3 lands these names in this spec; step 4 lands
// them as Rust constants in text.rs. Module-private (matches silero convention).
// The Xenova clap-htsat-unfused export inlines attention_mask and position_ids
// derivation into the graph; only input_ids is externalized. See §7.4.
const TEXT_INPUT_IDS_NAME: &str = "input_ids";
const TEXT_OUTPUT_NAME:    &str = "text_embeds";
```

§7.3.1 scratch-lifecycle contract applies. ORT output extraction uses `try_extract_tensor::<f32>()` and
`TensorRef::from_array_view((shape_tuple, data))` exactly as in §8.2 (no ndarray, no `try_extract_raw_tensor`).

**`embed(text)`:**
1. `text.is_empty()` → `Error::EmptyInput { batch_index: None }`.
2. `tokenizer.encode(text, add_special_tokens=true)` → `Encoding`.
3. `ids: Vec<i64>` — clear → reserve(T) → extend_from_slice from cast u32 ids.
4. Bind a single tensor view via `TensorRef::from_array_view(([1usize, t], ids.as_slice()))?`; run with
   `ort::inputs![TEXT_INPUT_IDS_NAME => input_ids_view]`; validate output shape `[1, 512]` via the §7.3.1
   helper; copy out; drop view.
5. If `TEXT_OUTPUT_IS_UNIT_NORM` (§7.1, compile-time const) is `true`, construct via the trusted path
   with the release-mode unit-norm guard (`NORM_BUDGET = 1e-4`; same rationale as the audio-side guard
   at §8.2 step 5); otherwise L2-normalize → `Embedding`.

**`embed_batch(texts)`:**
1. Empty slice → `Ok(Vec::new())`.
2. For each `texts[i]`: empty → `Error::EmptyInput { batch_index: Some(i) }`.
3. `tokenizer.encode_batch(texts)` → all encodings already padded to `T_max` (BatchLongest applied per §9.1).
4. Resize encoder-owned ids scratch via clear/reserve/resize to `[N × T_max]`; copy in-place.
5. Bind a single tensor view, run with `ort::inputs![TEXT_INPUT_IDS_NAME => input_ids_view]`, validate
   output shape `[N, 512]`, copy out, drop view.
6. Row-by-row construct `Embedding`s — same branch as single-clip `embed`: if `TEXT_OUTPUT_IS_UNIT_NORM`
   is `true`, use the trusted path with the release-mode unit-norm guard (`NORM_BUDGET`, same rationale
   as the audio side); otherwise L2-normalize each row.

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

    /// Slice passed to from_slice_normalizing contained a non-finite
    /// component (NaN, +Inf, or -Inf). The audio path catches this earlier
    /// via NonFiniteAudio; this variant is the user-facing safety net for
    /// direct construction (e.g. read-back from a corrupt lancedb cell).
    /// Field name `component_index` matches the qualified-name convention
    /// used by sibling variants (`clip_index`, `batch_index`, `sample_index`)
    /// — minor stylistic divergence from soundevents' bare `index`. The
    /// error message uses bare "index" to avoid the awkward variant-noun-
    /// field triple repetition.
    #[error("embedding contains non-finite component at index {component_index}")]
    NonFiniteEmbedding { component_index: usize },

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

**Note on `Onnx` (#[from]) vs `OnnxLoadFromFile` / `OnnxLoadFromMemory`:** load-time errors are mapped
explicitly via `.map_err(|e| Error::OnnxLoadFromFile { path, source: e })?` so the path is captured.
Runtime ORT errors during `session.run()` fall through `#[from] ort::Error` to the catch-all `Onnx`
variant — no path context is meaningful at that point.

**Deliberate divergence from silero `UnexpectedOutputShape`.** silero's variant carries only
`{ tensor: &'static str, shape: Vec<i64> }` — `expected` is implicit from the tensor name and the
calling site. textclap's `UnexpectedTensorShape` adds an explicit `expected: Vec<i64>` field. textclap
has more diverse tensor shapes across audio (4-D), text (2-D), and outputs (2-D) than silero's narrower
audio-only graph, so carrying `expected` makes the error self-describing without requiring callers to
reverse the calling site. Acceptable structural addition; no API risk.

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
the `warmup_text` string from `golden_params.json` for text. The warmup string is sized **automatically by
the §3.1 prerequisite script** to land at ≥80 BPE tokens, so typical user search queries (5–15 tokens) and
edge-case longer queries up through CLAP's `max_length` (~77) won't reallocate token scratch on the first
real call. The script measures the actual post-tokenizer count via the `tokenizers` Python binding and
records the chosen string (and its measured token count) as `warmup_text` in `golden_params.json`. The
algorithm is deterministic: smallest integer `k` such that `pangram * k` yields ≥80 BPE tokens. No hand
iteration; reproducible across maintainers.

**Recorded `warmup_text`** (from `golden_params.json`; 84 BPE tokens — 9 repetitions of
`"the quick brown fox jumps over the lazy dog "` including the trailing space):

```
the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog 
```

**`warmup_text` drift caveat.** The script measures token count using its generation-time `tokenizers`
Python package against the bundled `tokenizer.json`. If a downstream user supplies a non-default
tokenizer.json or upgrades the Rust `tokenizers` crate beyond the pinned major, the runtime BPE count
may differ from the recorded value. The ≥80 target is robust to small drift but degrades under large
drift; warmup remains correct (still triggers ORT specialization) but token scratch may grow once on
the first real call beyond the recorded threshold.

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
  - `from_slice_normalizing` always produces unit-norm output for non-zero finite input.
  - `from_slice_normalizing` rejects all-zero input → `EmbeddingZero`.
  - `from_slice_normalizing` rejects any non-finite component (NaN, +Inf, -Inf in any of the 512
    positions) → `NonFiniteEmbedding`.
  - **Trust-path release guard:** when `AUDIO_OUTPUT_IS_UNIT_NORM` is `true`, the audio encoder rejects
    non-unit-norm projections with `EmbeddingNotUnitNorm` (~512 fma + 1 sub + 1 cmp at ~200 ns; uses
    the shared `NORM_BUDGET = 1e-4`). Test exercises this by supplying a known non-unit-norm tensor
    through a mocked session and asserting the error is raised.
  - `try_from_unit_slice` rejects wrong lengths (`EmbeddingDimMismatch`) and non-unit-norm input
    (`EmbeddingNotUnitNorm`) at the shared `NORM_BUDGET = 1e-4`.
  - `dot ≈ cosine` for unit inputs (within fp32 ULP).
  - `is_close(&self, &self, 0.0)` returns true.
  - `is_close_cosine(&self, &self, 0.0)` returns true.
  - **Cancellation safety.** A uniform scalar perturbation cancels in normalization (both vectors
    project to the same unit vector and the test is tautological). The test must use a *non-uniform*
    perturbation that survives normalization, with ε small enough to actually trigger fp32 cancellation
    in the naive `1 − dot(a, b)` path:

    ```rust
    let mut x = [0.0_f32; 512]; x[0] = 1.0;
    let mut y = [0.0_f32; 512]; y[0] = 1.0; y[1] = 1.0e-4;
    let a = Embedding::from_slice_normalizing(&x)?;
    let b = Embedding::from_slice_normalizing(&y)?;

    // For ε = 1e-4: ‖y‖² = 1 + 1e-8. fp32 ulp(1) ≈ 1.19e-7, so 1e-8 < ulp/2
    // → ‖y‖² rounds to exactly 1.0, normalization is the identity, b = y.
    // Then dot(a, b) = 1·1 + 0·ε = 1.0 exactly; naive 1 − dot = 0.
    // Safe 0.5 · ‖a − b‖² = ε² / 2 = 5e-9 (representable in fp32).
    //
    // Pick a tolerance below the true 5e-9: the safe impl correctly returns false;
    // the naive impl wrongly returns true (0 < 1e-12). The assertion fails iff
    // the naive impl regressed in.
    assert!(!a.is_close_cosine(&b, 1.0e-12));

    // Sanity: 1.0 − dot(a, b) computed in fp32 is *exactly* 0.0 here because
    // dot(a, b) = 1·1 + 0·1e-4 = 1.0 mathematically. The naive form returning
    // 0 is the bug: it masks the true cosine distance 0.5·‖a − b‖² = 5e-9
    // that the safe form recovers. (This assertion documents the cancellation;
    // the discrimination is in the assert!(!...) above.)
    let naive: f32 = 1.0 - a.dot(&b);
    assert_eq!(naive, 0.0_f32);
    ```

    A regression to the naive implementation makes the `assert!(!...)` fail. ε = 1e-4 (not 1e-9 or 1e-3)
    is the right perturbation. **For ε = 1e-9:** the safe value `0.5·‖a − b‖² = 5e-19` is itself below
    `tol = 1e-12`, so safe and naive both return `true` — no discrimination. **For ε = 1e-3:**
    `‖y‖² = 1 + 1e-6 ≈ 8 ulps` is fp32-representable, normalization genuinely changes the vectors, and
    naive ≈ safe ≈ 5e-7 — also no discrimination.

    **Test scope.** The test verifies cancellation safety only — any implementation that produces
    `5e-9 ± a few fp32 ULP` for these inputs satisfies it. The squared-distance form is the canonical
    fix; alternative correct implementations (e.g. an extended-precision intermediate) would also pass.
  - `to_vec` and `as_slice` are byte-equal.
  - **No `pub const DIM`** — `# compile_fail` doctest on `Embedding`'s rustdoc:
    ```rust,compile_fail
    let _ = textclap::Embedding::DIM;     // textclap intentionally exposes no DIM const
    ```
    Avoids pulling in `trybuild` as a dev-dep.
  - **Custom `Debug` output** does not contain 512 floats — assert via `format!("{:?}", emb)`
    contains `"head:"` and not `"0.0,"` repeated 512 times.
  - **No `PartialEq` derived** — same `# compile_fail` doctest pattern:
    ```rust,compile_fail
    let _ = a == b;                       // Embedding does not implement PartialEq
    ```

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
| Rust audio embedding vs golden (int8 vs int8)   | 5e-4                 | 1e-4 *                     | bound 6.4e-5 + ~1.5× headroom    |
| Rust text embedding vs golden (int8 vs int8)    | 1e-5                 | 5e-8                       | bound 2.6e-8 + ~2× headroom      |
| Cross-quantization audio (fp32 vs int8 ref)     | 1e-2                 | 5e-2                       | bound 2.6e-2 + ~2× headroom      |
| Cross-platform reproducibility (CPU EP variance)| 1e-5 (preliminary)   | 5e-8 (preliminary)         | inherited from text-row math; int8-on-MLAS-vs-Accelerate variance has not been measured — widen if real-run measurement supports it |

\* The audio-row cosine value `1e-4` is numerically equal to `NORM_BUDGET` but they measure different
quantities: the cosine column tests semantic equality `1 − cos(a, b)` between two embeddings via
`is_close_cosine`, while `NORM_BUDGET` is the per-vector unit-norm tolerance `(‖x‖² − 1).abs()` checked
inside `try_from_unit_slice` and the §8.2/§9.2 trust-path guards. The numerical coincidence is from
independent derivations.

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

**`try_from_unit_slice`'s norm² budget (`NORM_BUDGET = 1e-4`)** is comfortably above the worst-case
~6.1e-5 BLAS-vs-sequential-sum drift derived in §7.5 — goldens are L2-normalized in the same float
order Rust uses, but a maintainer regenerating goldens on a different BLAS runtime (MKL vs OpenBLAS
vs Apple Accelerate) could plausibly approach the worst case. 1e-4 absorbs that with margin.

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

The audio/text batched benches use `criterion::BenchmarkGroup` with one entry per batch size; each entry
grows scratch on its first iteration (since `warmup()` only sizes for `N=1`). Criterion's per-group warmup
phase absorbs this — the reported median reflects steady-state cost, not first-call growth.

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
- **Pre-allocation of scratch to a fixed `MAX_BATCH × T × 64` at construction**, eliminating the
  resize-during-inference class structurally instead of relying on the §7.3.1 borrow-checker pattern.
  Trade: API rigidity. Adopt if profiling or fuzz-style stress testing surfaces resize-related issues.
- **In-flight cancellation.** ORT 2.x exposes `RunOptions::new()?` and `Arc<RunOptions>::terminate()`.
  Implementing this would require threading an `Arc<RunOptions>` through every `embed*` call and exposing
  a `CancelHandle` type. Deferred — feature decision, not infeasibility.
- An AddressSanitizer CI job — Miri can't cross the FFI boundary, but ASan does. Only worth adding if the
  §7.3.1 contract ever needs empirical re-validation beyond what the borrow checker enforces statically.
- **fp16 storage round-trip support.** `try_from_unit_slice` is currently OUT OF SCOPE for fp16 storage
  (fp16's ulp(1.0) ≈ 9.77e-4 exceeds the 1e-4 norm budget); users storing embeddings in fp16 columns must
  use `from_slice_normalizing` on read-back. A future variant `try_from_unit_slice_fp16` (with a relaxed
  ~3e-2 budget — derived from worst-case fp16 round-trip drift `Δ(‖x‖²) ≈ 2·δ·√dim = 2·5e-4·√512 ≈ 2.3e-2`,
  rounded up with margin; revisit when the dimension changes) would let users opt into "I know it came
  through fp16, don't normalize again" semantics.
- **Tighten `NORM_BUDGET` from 1e-4 to 5e-5** once first-run BLAS-variation telemetry confirms safety.
  The §7.5 rationale explains the asymmetric-rollback argument for shipping at 1e-4: rolling back to
  5e-5 if measurement supports it is a low-stakes spec edit, while rolling forward from a 5e-5
  production failure would be a higher-stakes patch. **Realistic target depends on telemetry**: the
  worst-case bound is ~6.1e-5 (§7.5 BLAS-vs-sequential-sum derivation), so 5e-5 is achievable only if
  observed drift is ≪ worst-case. If telemetry shows actual drift hitting ≥ 5e-5, the realistic
  tightening target is ~7e-5 (just above worst-case with margin) rather than 5e-5. Tracking this as
  a registered task rather than an inline comment.
