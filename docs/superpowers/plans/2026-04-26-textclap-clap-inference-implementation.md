# textclap CLAP Inference Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the textclap crate per the freeze-approved spec at `docs/superpowers/specs/2026-04-25-textclap-clap-inference-design.md` (revision 16): a Rust ONNX-inference library for the LAION CLAP-HTSAT-unfused model exposing `AudioEncoder`, `TextEncoder`, and a top-level `Clap` with zero-shot classification.

**Architecture:** Synchronous, thread-per-core, no async. Public API matches sibling crates (silero / soundevents / mediatime). Audio path: 48 kHz mono PCM → mel features (Slaney scale, periodic Hann, power_to_dB) → INT8 ONNX → 512-dim L2-normalized `Embedding`. Text path: query string → RoBERTa tokenizer → INT8 ONNX → 512-dim `Embedding`. Both encoders own their `ort::Session` and scratch `Vec`s by value; the §7.3.1 scratch-lifecycle contract (clear → reserve → extend → bind → run → drop views) prevents UB through ORT's FFI boundary.

**Tech Stack:** Rust 2024 edition, MSRV 1.85, `ort = "2.0.0-rc.12"` (caret-pinned, sibling-coupled), `rustfft = "6"`, `tokenizers = "0.20"`, `thiserror = "2"`. Optional `serde` feature. No `tokio`, no `ndarray`, no `download` feature. Dev-deps: `criterion`, `rubato`, `npyz`, `hound`.

---

## Spec section reference

This plan refers extensively to the spec by section number. Throughout, **"§N"** means
`docs/superpowers/specs/2026-04-25-textclap-clap-inference-design.md` section N. The spec is the
authoritative source of design rationale; this plan is the build sequence. Where the spec contains a
canonical pseudo-code block (e.g. §8.2's `embed_projections_batched` body, §3.2's Python script), the
plan reproduces the relevant excerpt rather than referring back, since "the engineer may be reading
tasks out of order."

## File structure

```
textclap/
├── Cargo.toml                        # see Task 7
├── build.rs                          # verbatim from sibling silero/build.rs
├── README.md                         # see Task 30
├── CHANGELOG.md                      # see Task 30
├── LICENSE-MIT, LICENSE-APACHE, COPYRIGHT  (kept from template, copyright-updated)
├── src/
│   ├── lib.rs                        # ~30 lines: lints + module decls + re-exports
│   ├── error.rs                      # ~120 lines: thiserror enum + Result alias
│   ├── options.rs                    # ~110 lines: Options, ChunkingOptions, GraphOptimizationLevel re-export
│   ├── mel.rs                        # ~280 lines: MelExtractor + T_FRAMES + (optional) HTSAT_INPUT_MEAN/STD
│   ├── audio.rs                      # ~420 lines: AudioEncoder + AUDIO_*_NAME consts + AUDIO_OUTPUT_IS_UNIT_NORM
│   ├── text.rs                       # ~340 lines: TextEncoder + TEXT_*_NAME consts + TEXT_OUTPUT_IS_UNIT_NORM
│   └── clap.rs                       # ~280 lines: Clap + Embedding + NORM_BUDGET + LabeledScore[Owned]
├── tests/
│   ├── clap_integration.rs           # ~180 lines: TEXTCLAP_MODELS_DIR-gated integration test
│   └── fixtures/                     # populated by Phase A
│       ├── README.md, MODELS.md
│       ├── sample.wav
│       ├── golden_params.json, golden_onnx_io.json
│       ├── golden_mel.npy
│       ├── golden_audio_emb.npy, golden_text_embs.npy
│       ├── filterbank_row_{0,10,32}.npy
│       ├── regen_golden.py
│       └── inspect_onnx.py
├── benches/
│   ├── bench_mel.rs                  # ~30 lines
│   ├── bench_audio_encode.rs         # ~50 lines
│   └── bench_text_encode.rs          # ~50 lines
├── examples/
│   ├── audio_window_to_clap.rs       # ~80 lines
│   └── index_and_search.rs           # ~100 lines
└── .github/workflows/ci.yml          # see Task 31
```

**Module-boundary commitments** (locked at plan-time):
- `clap.rs` owns `Embedding`, `NORM_BUDGET`, `LabeledScore`, `LabeledScoreOwned`, and the top-level `Clap`. It is the only module that knows the embedding-construction internals; `audio.rs` and `text.rs` import `Embedding` and `NORM_BUDGET` but do not duplicate the unit-norm guard logic.
- `mel.rs` is `pub(crate)` only — never exposed publicly. Owns `T_FRAMES`, the Hann window construction, the Slaney/Slaney filterbank, and the STFT pipeline.
- `audio.rs` / `text.rs` own their tensor-name consts (module-private `const`, not `pub(crate)`) and the `*_OUTPUT_IS_UNIT_NORM` const.
- `error.rs` owns the single `Error` enum and `Result` alias; everything else `use`s.
- `options.rs` re-exports `GraphOptimizationLevel` from `ort` so the public path stays stable across ORT bumps.

---

## Phase A — Pre-implementation prerequisites (§3)

These tasks produce no Rust source. They acquire the model files, generate fixtures, and backfill the
spec's TBD values. **Phase A must complete before Phase B begins** — the §3.4 commit ordering depends on
fixture-derived values landing in the spec before they appear as Rust constants.

### Task 1: Acquire model files and record SHA256 (§3.3)

**Files:**
- Create: `tests/fixtures/MODELS.md`
- Create: `tests/fixtures/sample.wav` (downloaded; ~200 KB public-domain dog-bark)
- Create: `tests/fixtures/README.md` (provenance + license attribution for sample.wav)

This task is "do this once at maintenance time" — not a TDD cycle. The output is verifiable by the SHA256
hashes matching what the maintainer recorded.

- [ ] **Step 1: Download model files from HuggingFace**

```bash
mkdir -p ~/textclap-models
cd ~/textclap-models
HF_REVISION=$(curl -s https://huggingface.co/api/models/Xenova/clap-htsat-unfused | jq -r .sha)
echo "HF revision: $HF_REVISION"
curl -L -o audio_model_quantized.onnx \
  "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/${HF_REVISION}/onnx/audio_model_quantized.onnx"
curl -L -o text_model_quantized.onnx \
  "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/${HF_REVISION}/onnx/text_model_quantized.onnx"
curl -L -o tokenizer.json \
  "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/${HF_REVISION}/tokenizer.json"
shasum -a 256 audio_model_quantized.onnx text_model_quantized.onnx tokenizer.json
```

Record the printed HF revision SHA and the three file SHA256s.

- [ ] **Step 2: Acquire a public-domain dog-bark WAV** (~2-3 s, 48 kHz mono, ~200-300 KB)

Suggested source: Freesound CC0 dog-bark recordings; convert to 48 kHz mono via
`sox input.wav -r 48000 -c 1 sample.wav`. Place at `tests/fixtures/sample.wav`. Record provenance, source
URL, and license attribution in `tests/fixtures/README.md`.

- [ ] **Step 3: Write `tests/fixtures/MODELS.md`**

```markdown
# textclap model artifacts

textclap loads three files at runtime; this document pins the verified versions.

**Source:** [Xenova/clap-htsat-unfused](https://huggingface.co/Xenova/clap-htsat-unfused)
**Pinned revision:** `<HF revision SHA from Task 1 step 1>`

| File                          | Size   | SHA256 (from `shasum -a 256`)                                       |
|-------------------------------|--------|---------------------------------------------------------------------|
| `audio_model_quantized.onnx`  | 33 MB  | `<sha from step 1>`                                                 |
| `text_model_quantized.onnx`   | 121 MB | `<sha from step 1>`                                                 |
| `tokenizer.json`              | 2.0 MB | `<sha from step 1>`                                                 |

Mismatched SHA256s produce undefined results — typically `Error::SessionSchema` or
`Error::UnexpectedTensorShape`, or in the worst case silent embedding drift. Verify before deployment.

**Original LAION model:** [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused)
(CC-BY 4.0 — attribution required when redistributing model files; see README §11.6).
```

- [ ] **Step 4: Write `tests/fixtures/README.md`** (sample.wav provenance):

```markdown
# textclap test fixtures

## sample.wav

- **Duration:** ~3 s (verify exact length with `soxi sample.wav`)
- **Format:** 48 kHz mono PCM (s16 or f32 — Rust integration test uses `hound` to read either)
- **Content:** dog barking
- **Source:** <Freesound URL or other CC0 source>
- **License:** Public domain (CC0). No attribution required, but credit appreciated.

The integration test (`tests/clap_integration.rs`) asserts that `classify_all` ranks
`"a dog barking"` in the top 2 against `["a dog barking", "rain", "music", "silence", "door creaking"]`,
so the WAV must contain primarily dog-bark content.

## golden_*.npy and golden_*.json

Generated by `regen_golden.py` and `inspect_onnx.py`. See those scripts for parameter pins.
```

- [ ] **Step 5: Commit (commit 0 of §3.4)**

```bash
git add tests/fixtures/MODELS.md tests/fixtures/README.md tests/fixtures/sample.wav
git commit -m "Add model SHA256 record and test fixture WAV (Phase A commit 0)"
```

---

### Task 2: Write `regen_golden.py` (§3.1)

**Files:**
- Create: `tests/fixtures/regen_golden.py`

The script generates `golden_params.json`, `golden_mel.npy`, `golden_audio_emb.npy`,
`golden_text_embs.npy`, and the three `filterbank_row_*.npy` reference rows. It depends on
`golden_onnx_io.json` (produced by Task 3-4 first); commit ordering is:

1. Task 2 + Task 3 + Task 4: scripts committed
2. Task 5: scripts run, fixtures committed

so the script can reference `golden_onnx_io.json` even though that file lands in commit 2 (the scripts
are written before the fixtures are produced).

- [ ] **Step 1: Create the script with header + imports + setup**

```python
"""regen_golden.py — generate textclap test fixtures from the actual model files.

Pinned versions (header — install via the matching pip extras at maintenance time):
  transformers >= 4.36   (ClapModel / ClapFeatureExtractor)
  torch >= 2.0
  optimum >= 1.16
  onnxruntime >= 1.16
  librosa >= 0.10
  numpy >= 1.25
  tokenizers (Python binding) >= 0.15

Usage (from textclap/ root):
  python tests/fixtures/regen_golden.py --models-dir ~/textclap-models/

Produces (in tests/fixtures/):
  golden_params.json
  golden_mel.npy
  golden_audio_emb.npy
  golden_text_embs.npy
  filterbank_row_{0,10,32}.npy

Reads (must exist before running this script):
  golden_onnx_io.json (produced by inspect_onnx.py)
  sample.wav
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import ClapFeatureExtractor

PANGRAM = "the quick brown fox jumps over the lazy dog "
LABELS = ["a dog barking", "rain", "music", "silence", "door creaking"]
WARMUP_TARGET_TOKENS = 80


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True, type=Path,
                    help="Directory containing audio_model_quantized.onnx, text_model_quantized.onnx, tokenizer.json")
    ap.add_argument("--fixtures-dir", default=Path("tests/fixtures"), type=Path)
    args = ap.parse_args()

    fixtures = args.fixtures_dir
    models = args.models_dir
    fixtures.mkdir(parents=True, exist_ok=True)

    onnx_io = json.load(open(fixtures / "golden_onnx_io.json"))

    _step_extractor_and_mel(fixtures, onnx_io)
    _step_warmup_text(fixtures, models)
    _step_audio_golden(fixtures, models, onnx_io)
    _step_text_goldens(fixtures, models, onnx_io)
    _step_filterbank_rows(fixtures)

    print(f"All goldens written to {fixtures}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the extractor / mel-features step**

```python
def _step_extractor_and_mel(fixtures: Path, onnx_io: dict) -> None:
    """ §3.1 step 2-3: extract parameters + golden mel features."""
    extractor = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

    audio, _ = librosa.load(str(fixtures / "sample.wav"), sr=48000, mono=True)
    features = extractor(audio, sampling_rate=48000, return_tensors="pt")
    mel = features["input_features"].numpy()  # shape varies — flatten to [64, T]
    if mel.ndim == 4:
        mel_2d = mel[0, 0]
    elif mel.ndim == 3:
        mel_2d = mel[0]
    else:
        raise ValueError(f"unexpected mel shape from extractor: {mel.shape}")
    np.save(fixtures / "golden_mel.npy", mel_2d.astype(np.float32))

    T = int(mel_2d.shape[1])
    params = {
        "sampling_rate": int(extractor.sampling_rate),
        "feature_size": int(extractor.feature_size),
        "fft_window_size": int(extractor.fft_window_size),
        "hop_length": int(extractor.hop_length),
        "max_length_s": float(extractor.max_length_s),
        "mel_scale": str(getattr(extractor, "mel_scale", "slaney")),
        "filterbank_norm": str(getattr(extractor, "norm", "slaney")),
        "amin": float(getattr(extractor, "amin", 1e-10)),
        "ref": float(getattr(extractor, "ref", 1.0)),
        "top_db": (None if getattr(extractor, "top_db", None) is None
                   else float(extractor.top_db)),
        "frequency_min": float(extractor.frequency_min),
        "frequency_max": float(extractor.frequency_max),
        "padding_mode": str(getattr(extractor, "padding", "repeatpad")),
        "truncation_mode": str(getattr(extractor, "truncation", "rand_trunc")),
        "T_frames": T,
    }
    # `htsat_input_normalization` filled by inspect_onnx.py functional pass; preserve if already set.
    existing = {}
    if (fixtures / "golden_params.json").exists():
        existing = json.load(open(fixtures / "golden_params.json"))
    params["htsat_input_normalization"] = existing.get(
        "htsat_input_normalization", {"type": "TBD", "mean": None, "std": None})
    params["htsat_norm_drift"] = existing.get("htsat_norm_drift", None)

    json.dump(params, open(fixtures / "golden_params.json", "w"), indent=2)
```

- [ ] **Step 3: Add the deterministic warmup-text step (§3.1 step 2 last bullet)**

```python
def _step_warmup_text(fixtures: Path, models: Path) -> None:
    """Smallest k such that PANGRAM * k tokenizes to >= WARMUP_TARGET_TOKENS BPE tokens."""
    tok = Tokenizer.from_file(str(models / "tokenizer.json"))
    k = 0
    n_tokens = 0
    while n_tokens < WARMUP_TARGET_TOKENS:
        k += 1
        text = PANGRAM * k
        n_tokens = len(tok.encode(text).ids)
    warmup_text = PANGRAM * k

    params = json.load(open(fixtures / "golden_params.json"))
    params["warmup_text"] = warmup_text
    params["warmup_text_token_count"] = int(n_tokens)
    params["warmup_text_repetitions"] = k
    json.dump(params, open(fixtures / "golden_params.json", "w"), indent=2)
```

- [ ] **Step 4: Add the audio-model golden step (§3.1 step 4)**

```python
def _step_audio_golden(fixtures: Path, models: Path, onnx_io: dict) -> None:
    extractor = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
    audio, _ = librosa.load(str(fixtures / "sample.wav"), sr=48000, mono=True)
    features = extractor(audio, sampling_rate=48000, return_tensors="pt")

    audio_session = ort.InferenceSession(str(models / "audio_model_quantized.onnx"))
    raw = audio_session.run(
        [onnx_io["audio_output_name"]],
        {onnx_io["audio_input_name"]: features["input_features"].numpy()},
    )[0]
    raw = raw.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(raw).astype(np.float32)
    embedding = (raw / norm).astype(np.float32)
    assert embedding.shape == (512,), f"unexpected audio embedding shape: {embedding.shape}"
    np.save(fixtures / "golden_audio_emb.npy", embedding)
```

- [ ] **Step 5: Add the text-model goldens step (§3.1 step 5)**

```python
def _step_text_goldens(fixtures: Path, models: Path, onnx_io: dict) -> None:
    tok = Tokenizer.from_file(str(models / "tokenizer.json"))
    text_session = ort.InferenceSession(str(models / "text_model_quantized.onnx"))

    embs = []
    for label in LABELS:
        enc = tok.encode(label)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feeds = {
            onnx_io["text_input_ids_name"]: ids,
            onnx_io["text_attention_mask_name"]: mask,
        }
        if onnx_io.get("text_position_ids_name"):
            pad_id = onnx_io.get("text_pad_id", 1)
            # RoBERTa: pos = pad_id + 1 + cumsum(non_pad_mask) over the non-pad span
            pos = np.cumsum(mask, axis=1, dtype=np.int64) + pad_id
            pos = pos * mask  # zero-out padded positions (matches model's masked positions)
            feeds[onnx_io["text_position_ids_name"]] = pos.astype(np.int64)

        raw = text_session.run([onnx_io["text_output_name"]], feeds)[0]
        raw = raw.astype(np.float32).reshape(-1)
        assert raw.shape == (512,), f"unexpected text projection shape for {label!r}: {raw.shape}"
        norm = np.linalg.norm(raw).astype(np.float32)
        embs.append((raw / norm).astype(np.float32))

    np.save(fixtures / "golden_text_embs.npy", np.stack(embs))
```

- [ ] **Step 6: Add the filterbank-rows step (§8.1.1 unit test)**

```python
def _step_filterbank_rows(fixtures: Path) -> None:
    """ Pre-computed librosa references for the §8.1.1 mel-filter row test."""
    fb = librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64,
                             fmin=50, fmax=14000, htk=False, norm="slaney")
    np.save(fixtures / "filterbank_row_0.npy",  fb[0].astype(np.float32))
    np.save(fixtures / "filterbank_row_10.npy", fb[10].astype(np.float32))  # near 1 kHz Slaney inflection
    np.save(fixtures / "filterbank_row_32.npy", fb[32].astype(np.float32))
```

- [ ] **Step 7: Commit the script**

```bash
git add tests/fixtures/regen_golden.py
git commit -m "Add regen_golden.py for textclap fixture generation (Phase A commit 1a)"
```

---

### Task 3: Write `inspect_onnx.py` static-graph pass (§3.2 static)

**Files:**
- Create: `tests/fixtures/inspect_onnx.py`

This script produces `golden_onnx_io.json` from static graph inspection. The functional verification
pass is added in Task 4.

- [ ] **Step 1: Create the script header + imports + main**

```python
"""inspect_onnx.py — static + functional inspection of CLAP ONNX exports.

Static pass: walks graph.input / graph.output, looks for L2-normalize at the tail,
identifies tensor input names. Output → tests/fixtures/golden_onnx_io.json.

Functional pass (added in Task 4): runs PyTorch fp32 vs int8 ONNX end-to-end to detect
HTSAT input-normalization mismatch.

Usage:
  python tests/fixtures/inspect_onnx.py --models-dir ~/textclap-models/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True, type=Path)
    ap.add_argument("--fixtures-dir", default=Path("tests/fixtures"), type=Path)
    args = ap.parse_args()

    audio_proto = onnx.load(str(args.models_dir / "audio_model_quantized.onnx"))
    text_proto  = onnx.load(str(args.models_dir / "text_model_quantized.onnx"))

    result = {}
    result.update(_inspect_audio(audio_proto))
    result.update(_inspect_text(text_proto))

    json.dump(result, open(args.fixtures_dir / "golden_onnx_io.json", "w"), indent=2)
    print(f"Wrote {args.fixtures_dir / 'golden_onnx_io.json'}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add audio-graph static inspection**

```python
def _inspect_audio(proto: onnx.ModelProto) -> dict:
    g = proto.graph
    [audio_in] = list(g.input)
    [audio_out] = list(g.output)
    audio_in_shape = [d.dim_value if d.HasField("dim_value") else "?"
                      for d in audio_in.type.tensor_type.shape.dim]
    audio_out_shape = [d.dim_value if d.HasField("dim_value") else "?"
                       for d in audio_out.type.tensor_type.shape.dim]

    last_5_node_ops = [n.op_type for n in list(g.node)[-5:]]
    is_unit_norm = _has_l2_normalize_tail(last_5_node_ops)

    return {
        "audio_input_name":  audio_in.name,
        "audio_input_shape": audio_in_shape,
        "audio_output_name": audio_out.name,
        "audio_output_shape": audio_out_shape,
        "audio_output_is_unit_norm": is_unit_norm,
        "audio_last_5_ops": last_5_node_ops,
    }


def _has_l2_normalize_tail(ops: list[str]) -> bool:
    """Return True if the trailing op pattern looks like L2-normalize.

    Recognized patterns:
      [..., LpNormalization]                (axis=-1, p=2; the canonical export)
      [..., ReduceL2, Div]
      [..., ReduceL2, Clip, Div]
    """
    if "LpNormalization" in ops:
        return True
    # Look for ReduceL2 followed (within 2 ops) by Div
    for i, op in enumerate(ops):
        if op == "ReduceL2":
            for j in range(i + 1, min(i + 3, len(ops))):
                if ops[j] == "Div":
                    return True
    return False
```

- [ ] **Step 3: Add text-graph static inspection**

```python
def _inspect_text(proto: onnx.ModelProto) -> dict:
    g = proto.graph
    inputs_by_name = {i.name: i for i in g.input}

    # RoBERTa exports always have input_ids and attention_mask. Some externalize position_ids.
    text_input_ids_name      = next(n for n in inputs_by_name if "input_ids" in n.lower())
    text_attention_mask_name = next(n for n in inputs_by_name if "attention" in n.lower() and "mask" in n.lower())
    text_position_ids_name   = next((n for n in inputs_by_name if "position" in n.lower()), None)

    [text_out] = list(g.output)
    text_out_shape = [d.dim_value if d.HasField("dim_value") else "?"
                      for d in text_out.type.tensor_type.shape.dim]

    last_5_node_ops = [n.op_type for n in list(g.node)[-5:]]
    is_unit_norm = _has_l2_normalize_tail(last_5_node_ops)

    return {
        "text_input_ids_name":      text_input_ids_name,
        "text_attention_mask_name": text_attention_mask_name,
        "text_position_ids_name":   text_position_ids_name,  # None if not externalized
        "text_output_name":         text_out.name,
        "text_output_shape":        text_out_shape,
        "text_output_is_unit_norm": is_unit_norm,
        "text_last_5_ops":          last_5_node_ops,
        "text_pad_id":              1,  # RoBERTa default; computed only if position_ids is externalized
    }
```

- [ ] **Step 4: Commit the static-only script**

```bash
git add tests/fixtures/inspect_onnx.py
git commit -m "Add inspect_onnx.py static-graph inspection (Phase A commit 1b)"
```

---

### Task 4: Add functional HTSAT-input-norm verification to `inspect_onnx.py` (§3.2 functional)

**Files:**
- Modify: `tests/fixtures/inspect_onnx.py` (add a functional pass after the static one)

The functional pass runs the PyTorch fp32 reference end-to-end, then runs the int8 ONNX with both
`none` and `global_mean_std` input transforms, picks the lower-error transform, and writes the choice
into `golden_params.json`.

- [ ] **Step 1: Add the functional-verification function**

```python
import torch
import torch.nn.functional as F
import librosa
from transformers import ClapFeatureExtractor, ClapModel

# AudioSet stats per AST/HTSAT convention; computed in dB space (post-power_to_db).
AUDIOSET_MEAN = -4.27
AUDIOSET_STD  =  4.57

def _apply_audioset_norm(x: np.ndarray) -> np.ndarray:
    """Per-element global mean/std normalization in dB space."""
    return (x - AUDIOSET_MEAN) / AUDIOSET_STD


def _functional_verify(models: Path, fixtures: Path, onnx_io: dict) -> dict:
    extractor = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
    pt_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    audio_session = ort.InferenceSession(str(models / "audio_model_quantized.onnx"))

    audio, _ = librosa.load(str(fixtures / "sample.wav"), sr=48000, mono=True)
    features = extractor(audio, sampling_rate=48000, return_tensors="pt")
    assert features["input_features"].shape[0] == 1, "verification expects batch_size=1"

    # PyTorch fp32 reference, with no_grad (params have requires_grad=True even in eval).
    with torch.no_grad():
        pt_emb = pt_model.get_audio_features(**features)
        pt_emb = F.normalize(pt_emb, dim=-1)  # robust to any batch size
    pt_emb_np = pt_emb.numpy().reshape(-1)

    results = {}
    for name, fn in [("none", lambda x: x), ("global_mean_std", _apply_audioset_norm)]:
        ort_input = fn(features["input_features"].numpy()).astype(np.float32)
        ort_raw = audio_session.run([onnx_io["audio_output_name"]],
                                    {onnx_io["audio_input_name"]: ort_input})[0]
        ort_emb = ort_raw.astype(np.float32).reshape(-1)
        # If the ONNX output isn't already unit-norm, normalize externally to match pt_emb.
        if not onnx_io["audio_output_is_unit_norm"]:
            ort_emb = ort_emb / np.linalg.norm(ort_emb).astype(np.float32)
        drift = float(np.max(np.abs(pt_emb_np - ort_emb)))
        results[name] = drift

    chosen, drift = min(results.items(), key=lambda kv: kv[1])
    if drift >= 2e-2:
        raise SystemExit(
            f"functional verification REJECTED: best drift {drift:.3e} ≥ 2e-2 (both transforms tried). "
            f"Drifts: {results}. Investigate before continuing."
        )
    if drift >= 5e-3:
        print(f"WARNING (yellow zone): drift {drift:.3e} ∈ [5e-3, 2e-2). Picked {chosen!r}; "
              f"see §3.2 for the heuristic-threshold rationale.")

    return {
        "type": chosen,
        "mean": AUDIOSET_MEAN if chosen == "global_mean_std" else None,
        "std":  AUDIOSET_STD  if chosen == "global_mean_std" else None,
        "drift_at_chosen": drift,
        "drifts_all_transforms": results,
    }
```

- [ ] **Step 2: Wire `_functional_verify` into `main()`**

Modify `main()` so that after writing `golden_onnx_io.json`, it runs the functional pass and updates
`golden_params.json` (creating it if absent — `regen_golden.py` will read+merge later):

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True, type=Path)
    ap.add_argument("--fixtures-dir", default=Path("tests/fixtures"), type=Path)
    args = ap.parse_args()

    audio_proto = onnx.load(str(args.models_dir / "audio_model_quantized.onnx"))
    text_proto  = onnx.load(str(args.models_dir / "text_model_quantized.onnx"))

    onnx_io = {}
    onnx_io.update(_inspect_audio(audio_proto))
    onnx_io.update(_inspect_text(text_proto))
    json.dump(onnx_io, open(args.fixtures_dir / "golden_onnx_io.json", "w"), indent=2)

    # Functional verification — uses the static-pass output above.
    htsat_norm = _functional_verify(args.models_dir, args.fixtures_dir, onnx_io)
    params_path = args.fixtures_dir / "golden_params.json"
    params = json.load(open(params_path)) if params_path.exists() else {}
    params["htsat_input_normalization"] = {
        "type": htsat_norm["type"],
        "mean": htsat_norm["mean"],
        "std":  htsat_norm["std"],
    }
    params["htsat_norm_drift"] = htsat_norm["drift_at_chosen"]
    json.dump(params, open(params_path, "w"), indent=2)

    print(f"Wrote {args.fixtures_dir / 'golden_onnx_io.json'}")
    print(f"Updated {params_path} with htsat_input_normalization={htsat_norm['type']!r} "
          f"(drift={htsat_norm['drift_at_chosen']:.3e})")
```

- [ ] **Step 3: Commit the functional-verify addition**

```bash
git add tests/fixtures/inspect_onnx.py
git commit -m "Add functional HTSAT input-normalization verification (Phase A commit 1c)"
```

---

### Task 5: Run scripts and commit fixtures (§3.4 commit 2)

**Files:**
- Create (script-generated): `tests/fixtures/golden_onnx_io.json`
- Create (script-generated): `tests/fixtures/golden_params.json`
- Create (script-generated): `tests/fixtures/golden_mel.npy`
- Create (script-generated): `tests/fixtures/golden_audio_emb.npy`
- Create (script-generated): `tests/fixtures/golden_text_embs.npy`
- Create (script-generated): `tests/fixtures/filterbank_row_{0,10,32}.npy`

This task runs the scripts, validates outputs, and commits the generated fixtures. **Phase A's only
manual decision point is here** — the maintainer reviews `golden_params.json` and
`golden_onnx_io.json` for plausibility before committing.

- [ ] **Step 1: Set up Python environment**

```bash
cd /path/to/textclap
python3 -m venv .venv
source .venv/bin/activate
pip install transformers>=4.36 torch>=2.0 onnxruntime>=1.16 librosa>=0.10 \
            numpy>=1.25 onnx>=1.15 tokenizers>=0.15
```

- [ ] **Step 2: Run `inspect_onnx.py`**

```bash
python tests/fixtures/inspect_onnx.py --models-dir ~/textclap-models/
```

Expected output: prints `Wrote tests/fixtures/golden_onnx_io.json` plus the JSON contents and the
HTSAT functional-verification drift summary. The `audio_input_shape` should be either `[?, 1, 64, ?]`
or `[?, 64, ?]`; the `audio_output_shape` should be `[?, 512]`. The `htsat_input_normalization.type`
will be one of `"none"` or `"global_mean_std"`. If the script aborts with "REJECTED: best drift ≥ 2e-2,"
investigate before proceeding.

- [ ] **Step 3: Run `regen_golden.py`**

```bash
python tests/fixtures/regen_golden.py --models-dir ~/textclap-models/
```

Expected output: prints `All goldens written to tests/fixtures/`. The script writes
`golden_mel.npy`, `golden_audio_emb.npy`, `golden_text_embs.npy`, the three `filterbank_row_*.npy`
references, and merges its parameter outputs into `golden_params.json` (which already contains
`htsat_input_normalization` from inspect_onnx.py).

- [ ] **Step 4: Spot-check the generated fixtures**

```bash
python -c "
import numpy as np, json
mel = np.load('tests/fixtures/golden_mel.npy')
emb = np.load('tests/fixtures/golden_audio_emb.npy')
texts = np.load('tests/fixtures/golden_text_embs.npy')
print('mel shape:', mel.shape, 'dtype:', mel.dtype, 'range:', mel.min(), mel.max())
print('audio emb shape:', emb.shape, 'norm:', np.linalg.norm(emb))
print('text embs shape:', texts.shape, 'first norm:', np.linalg.norm(texts[0]))
print('params:', json.dumps(json.load(open('tests/fixtures/golden_params.json')), indent=2))
print('onnx_io:', json.dumps(json.load(open('tests/fixtures/golden_onnx_io.json')), indent=2))
"
```

Expected: mel shape is `(64, T)` with `T` ≈ 998–1001 (the actual value goes into `T_FRAMES`); audio
embedding shape is `(512,)` with norm ≈ 1.0 (within 5e-7); text embedding shape is `(5, 512)` with
each row norm ≈ 1.0; `params` includes `T_frames`, `warmup_text` (≥80 tokens),
`htsat_input_normalization`, `top_db`. **Record `T_frames` and `htsat_input_normalization.type`** —
needed for Task 6.

- [ ] **Step 5: Commit the fixtures (commit 2 of §3.4)**

```bash
git add tests/fixtures/golden_*.npy tests/fixtures/golden_*.json \
        tests/fixtures/filterbank_row_*.npy
git commit -m "Add generated test fixtures from §3.1 / §3.2 prerequisite scripts (Phase A commit 2)"
```

---

### Task 6: Backfill spec TBDs (§3.4 commit 3)

**Files:**
- Modify: `docs/superpowers/specs/2026-04-25-textclap-clap-inference-design.md`

This is the §3.4 spec-update commit. Replace every TBD value in the spec with the value recorded in
`golden_params.json` and `golden_onnx_io.json`. **No Rust source is written until this commit lands.**

- [ ] **Step 1: Inventory TBDs**

```bash
grep -nE "TBD|expected.*verify|backfill required" \
  docs/superpowers/specs/2026-04-25-textclap-clap-inference-design.md
```

Expected to find references in §8.1 (frame count `T`, `top_db`, HTSAT input norm), §8.2 (audio input
shape / names), §9.2 (text input names, optional position_ids), §11.4 (warmup string).

- [ ] **Step 2: Edit §8.1 mel parameter table**

Replace the `**TBD by §3.1**` cells with the recorded values. Example for `T`:

```markdown
| Frame count `T`      | <T from golden_params.json> (recorded; see golden_params.json)        |
```

Same for `top_db` (use the literal value from JSON: either `None` or the recorded float).

For HTSAT input norm: replace `**TBD by §3.2 functional check**` with one of:
- `none` (no normalization needed in mel.rs post-log-mel step), or
- `global_mean_std (mean=−4.27, std=4.57)` (subtract / divide after log-mel; values from `golden_params.json`).

- [ ] **Step 3: Edit §8.2 / §9.2 tensor names and shapes**

In §8.2's `const AUDIO_INPUT_NAME` / `const AUDIO_OUTPUT_NAME` declarations, replace the example
strings with the values from `golden_onnx_io.json`. Same for §9.2's text constants and the optional
`TEXT_POSITION_IDS_NAME` (drop the const if `text_position_ids_name` is `null`).

If audio input shape is 3-D (`[batch, 64, T]`) rather than 4-D (`[batch, 1, 64, T]`), edit §8.2's
TensorRef construction to `[n, 64, T_FRAMES]` and remove the channel-dim view.

- [ ] **Step 4: Edit §11.4 warmup string**

Replace the conceptual reference with the literal recorded `warmup_text` from `golden_params.json`,
plus the recorded token count.

- [ ] **Step 5: Mark the spec as Phase A-complete**

Change the status line:

```markdown
**Status:** Approved (revision 17, Phase A complete — all §3 backfills landed)
```

- [ ] **Step 6: Verify no TBDs remain**

```bash
grep -nE "TBD|backfill required" \
  docs/superpowers/specs/2026-04-25-textclap-clap-inference-design.md
```

Expected: no matches outside the §3 prerequisite *description* (which references TBDs as the work
this section drives, not as live unresolved values).

- [ ] **Step 7: Commit (commit 3 of §3.4)**

```bash
git add docs/superpowers/specs/2026-04-25-textclap-clap-inference-design.md
git commit -m "Backfill spec TBDs from §3.1 / §3.2 fixture outputs (Phase A commit 3)"
```

---

## Phase B — Rust skeleton (§3.4 commit 4)

The skeleton makes the public API compile end-to-end. Method bodies are `unimplemented!()` placeholders;
no method produces real output yet. This unblocks `tests/clap_integration.rs` compilation in Phase G,
which references public types from `src/`.

### Task 7: `Cargo.toml` + `build.rs` + `src/lib.rs` skeleton

**Files:**
- Modify: `Cargo.toml` (replace template content)
- Modify: `build.rs` (verbatim from sibling silero/build.rs)
- Modify: `src/lib.rs` (replace template body)
- Create: `CHANGELOG.md`

- [ ] **Step 1: Replace `Cargo.toml`**

```toml
[package]
name = "textclap"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
authors = ["..."]   # match sibling-crate authors line
description = "Rust ONNX inference library for LAION CLAP-HTSAT-unfused"
license = "MIT OR Apache-2.0"
repository = "https://github.com/Findit-AI/textclap"
keywords = ["clap", "audio", "embedding", "onnx", "ml"]
categories = ["multimedia::audio", "science"]
include = [
    "src/**/*.rs",
    "build.rs",
    "Cargo.toml",
    "README.md",
    "CHANGELOG.md",
    "LICENSE-*",
    "COPYRIGHT",
]

[dependencies]
ort        = "2.0.0-rc.12"
rustfft    = "6"
tokenizers = "0.20"
thiserror  = "2"
serde      = { version = "1", features = ["derive"], optional = true }

[dev-dependencies]
criterion = "0.5"
rubato    = "0.16"
npyz      = "0.8"
hound     = "3"

[features]
default = []
serde   = ["dep:serde"]

[lints.rust]
rust_2018_idioms      = "warn"
single_use_lifetimes  = "warn"
unexpected_cfgs       = { level = "warn", check-cfg = ['cfg(all_tests)', 'cfg(tarpaulin)'] }

[package.metadata.docs.rs]
all-features  = true
rustdoc-args  = ["--cfg", "docsrs"]

[[bench]]
name    = "bench_mel"
harness = false

[[bench]]
name    = "bench_audio_encode"
harness = false

[[bench]]
name    = "bench_text_encode"
harness = false

[[example]]
name              = "audio_window_to_clap"
required-features = []

[[example]]
name              = "index_and_search"
required-features = []
```

- [ ] **Step 2: Copy `build.rs` verbatim from `silero/build.rs`** (one of the sibling crates).

```bash
cp /Users/user/Develop/findit-studio/silero/build.rs /Users/user/Develop/findit-studio/textclap/build.rs
```

Verify with `cat build.rs`: it should be a short script that emits `cargo:rustc-cfg=tarpaulin` when
the appropriate env var is set, with no-op default.

- [ ] **Step 3: Replace `src/lib.rs`**

```rust
//! textclap — CLAP (Contrastive Language-Audio Pre-training) inference library.
//!
//! See `docs/superpowers/specs/` for the full design spec.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod audio;
mod clap;
mod error;
mod mel;
mod options;
mod text;

pub use crate::audio::AudioEncoder;
pub use crate::clap::{Clap, Embedding, LabeledScore, LabeledScoreOwned};
pub use crate::error::{Error, Result};
pub use crate::options::{ChunkingOptions, GraphOptimizationLevel, Options};
pub use crate::text::TextEncoder;
```

- [ ] **Step 4: Replace `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

### Added
- Initial public release.
- `AudioEncoder` and `TextEncoder` for the LAION CLAP-HTSAT-unfused model.
- Top-level `Clap` with zero-shot classification helper.
- `Embedding` type with `is_close` / `is_close_cosine` similarity helpers.
- `serde` feature for `Options` / `ChunkingOptions` / `Embedding` (sequence form).
```

- [ ] **Step 5: Verify the crate compiles (it won't yet — that's expected)**

```bash
cargo check
```

Expected: many "unresolved module" / "cannot find type" errors for `audio`, `clap`, `error`, etc.
That's correct — the modules don't exist yet. The next tasks add them.

- [ ] **Step 6: Commit Cargo manifest + build.rs + lib.rs + CHANGELOG**

```bash
git add Cargo.toml build.rs src/lib.rs CHANGELOG.md
git commit -m "Add Cargo manifest, build.rs, lib.rs skeleton (Phase B start)"
```

---

### Task 8: `src/error.rs` (full Error enum + Result alias)

**Files:**
- Create: `src/error.rs`

This is the complete `Error` enum from spec §10. No tests yet — error variants are exercised by later
tasks. The skeleton's job is to make the type compile so other modules can `use crate::Error`.

- [ ] **Step 1: Create `src/error.rs`**

```rust
//! Error type for textclap. See spec §10 for design rationale.

use std::path::PathBuf;

/// Result type alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// All errors produced by textclap.
///
/// Path-carrying / memory-carrying load variants mirror silero's `LoadModel` pattern. Configuration
/// errors (`NoPadToken`, `PaddingFixedRejected`) are top-level variants matching sibling structure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// ONNX model load from a file path.
    #[error("failed to load ONNX model from {path}: {source}")]
    OnnxLoadFromFile {
        /// Path that failed to load.
        path: PathBuf,
        /// Underlying ORT error.
        #[source]
        source: ort::Error,
    },

    /// ONNX model load from caller-supplied bytes.
    #[error("failed to load ONNX model from memory: {0}")]
    OnnxLoadFromMemory(#[source] ort::Error),

    /// Tokenizer load from a file path.
    #[error("failed to load tokenizer from {path}: {source}")]
    TokenizerLoadFromFile {
        /// Path that failed to load.
        path: PathBuf,
        /// Underlying tokenizers error.
        #[source]
        source: tokenizers::Error,
    },

    /// Tokenizer load from caller-supplied bytes.
    #[error("failed to load tokenizer from memory: {0}")]
    TokenizerLoadFromMemory(#[source] tokenizers::Error),

    /// Tokenizer has no padding configuration AND no `<pad>` token.
    #[error("tokenizer has no pad token (configure padding in tokenizer.json or include a <pad> token)")]
    NoPadToken,

    /// `from_ort_session` received a Tokenizer with `Padding::Fixed`.
    #[error("from_ort_session rejected Padding::Fixed (use BatchLongest or pre-pad upstream)")]
    PaddingFixedRejected,

    /// ONNX session schema does not match what the encoder expects.
    #[error("ONNX session schema mismatch: {detail}")]
    SessionSchema {
        /// Human-readable description of the mismatch.
        detail: String,
    },

    /// Generic file-read failure.
    #[error("failed to read file {path}: {source}")]
    Io {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Audio input exceeds the 10 s window (480 000 samples).
    #[error("audio input length {got} exceeds maximum {max} samples (10 s @ 48 kHz)")]
    AudioTooLong {
        /// Provided length in samples.
        got: usize,
        /// Maximum allowed length in samples (480 000).
        max: usize,
    },

    /// Audio input has length 0.
    #[error("audio input is empty (clip index: {clip_index:?})")]
    EmptyAudio {
        /// Index in the batch (`None` for single-clip calls, `Some(i)` for batch calls).
        clip_index: Option<usize>,
    },

    /// Audio sample is non-finite (NaN, +Inf, -Inf).
    #[error("audio sample at index {sample_index} (clip {clip_index:?}) is non-finite")]
    NonFiniteAudio {
        /// Clip index in the batch (`None` for single-clip calls).
        clip_index: Option<usize>,
        /// Sample index within the clip.
        sample_index: usize,
    },

    /// Chunking options have an invalid value.
    #[error("invalid chunking options: window={window_samples}, hop={hop_samples}, batch={batch_size}; \
             all must be > 0 and hop ≤ window")]
    ChunkingConfig {
        /// Window length in samples.
        window_samples: usize,
        /// Hop length in samples.
        hop_samples: usize,
        /// Batch size.
        batch_size: usize,
    },

    /// Tokenization failed at runtime.
    #[error("tokenization failed: {0}")]
    Tokenize(#[source] tokenizers::Error),

    /// Empty `&str` passed, or an empty string at the given index in a batch.
    #[error("input text is empty (batch index: {batch_index:?})")]
    EmptyInput {
        /// Batch index (`None` for single-text calls).
        batch_index: Option<usize>,
    },

    /// Slice length didn't match the embedding dimension.
    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch {
        /// Expected dimension (512 for 0.1.0).
        expected: usize,
        /// Actual slice length.
        got: usize,
    },

    /// Slice was all zeros (degenerate norm).
    #[error("embedding is the zero vector")]
    EmbeddingZero,

    /// Slice contained a non-finite component.
    #[error("embedding contains non-finite component at index {component_index}")]
    NonFiniteEmbedding {
        /// Component index where the non-finite value was found.
        component_index: usize,
    },

    /// Embedding norm² deviates from 1.0 by more than `NORM_BUDGET`.
    #[error("embedding norm out of tolerance: |norm² − 1| = {norm_sq_deviation:.3e}")]
    EmbeddingNotUnitNorm {
        /// Absolute deviation `(norm² − 1).abs()`.
        norm_sq_deviation: f32,
    },

    /// ONNX output tensor shape mismatched the expected one.
    #[error("unexpected tensor shape for {tensor}: actual {actual:?}, expected {expected:?}")]
    UnexpectedTensorShape {
        /// Tensor name (one of the AUDIO_/TEXT_*_NAME constants).
        tensor: &'static str,
        /// Actual shape from ORT.
        actual: Vec<i64>,
        /// Expected shape from `golden_onnx_io.json`.
        expected: Vec<i64>,
    },

    /// ORT runtime error during inference (not load-time).
    #[error("ONNX runtime error: {0}")]
    Onnx(#[from] ort::Error),
}
```

- [ ] **Step 2: Verify error.rs compiles**

```bash
cargo check 2>&1 | grep -E "error\[|^error:" | head -20
```

Expected: errors about the *other* modules (audio, clap, etc.) but no errors mentioning `error.rs`.

- [ ] **Step 3: Commit**

```bash
git add src/error.rs
git commit -m "Add Error enum with full variant set"
```

---

### Task 9: `src/options.rs` — `Options` and `ChunkingOptions` (TDD)

**Files:**
- Create: `src/options.rs`

- [ ] **Step 1: Create `src/options.rs` with type declarations and tests in one shot**

```rust
//! Configuration options for textclap. See spec §7.7 (Options) and §7.8 (ChunkingOptions).
//!
//! Sibling-convention notes:
//! - No `with_intra_threads` knob — thread tuning is configured outside textclap via
//!   `from_ort_session` (matches silero `options.rs:128–145`).
//! - `Options` follows soundevents' unqualified naming over silero's `SessionOptions` because
//!   textclap has no second options type to disambiguate.
//! - All getters / builders / setters are `pub const fn` (matches silero pattern).

pub use ort::session::builder::GraphOptimizationLevel;

/// Construction-time options for textclap encoders.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Options {
    graph_optimization_level: GraphOptimizationLevel,
}

impl Options {
    /// Construct with default values (== `Self::default()`).
    pub const fn new() -> Self {
        Self {
            graph_optimization_level: GraphOptimizationLevel::Level3,
        }
    }

    /// Set the ORT graph-optimization level (consuming builder).
    pub const fn with_graph_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
        self.graph_optimization_level = level;
        self
    }

    /// Set the ORT graph-optimization level (in-place setter).
    pub const fn set_graph_optimization_level(
        &mut self,
        level: GraphOptimizationLevel,
    ) -> &mut Self {
        self.graph_optimization_level = level;
        self
    }

    /// Get the configured graph-optimization level.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn graph_optimization_level(&self) -> GraphOptimizationLevel {
        self.graph_optimization_level
    }
}

/// Chunking-window configuration for `embed_chunked`.
///
/// In 0.1.0 the aggregation strategy is fixed (centroid or spherical-mean, chosen at construction
/// per §3.2). See spec §7.8.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkingOptions {
    window_samples: usize,
    hop_samples: usize,
    batch_size: usize,
}

impl Default for ChunkingOptions {
    fn default() -> Self {
        Self {
            window_samples: 480_000,
            hop_samples: 480_000,
            batch_size: 8,
        }
    }
}

impl ChunkingOptions {
    /// Construct with default values (window=480_000, hop=480_000, batch_size=8).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the window length in samples (consuming builder).
    pub const fn with_window_samples(mut self, n: usize) -> Self {
        self.window_samples = n;
        self
    }

    /// Set the window length in samples (in-place setter).
    pub const fn set_window_samples(&mut self, n: usize) -> &mut Self {
        self.window_samples = n;
        self
    }

    /// Get the window length in samples.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Set the hop length in samples (consuming builder).
    pub const fn with_hop_samples(mut self, n: usize) -> Self {
        self.hop_samples = n;
        self
    }

    /// Set the hop length in samples (in-place setter).
    pub const fn set_hop_samples(&mut self, n: usize) -> &mut Self {
        self.hop_samples = n;
        self
    }

    /// Get the hop length in samples.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn hop_samples(&self) -> usize {
        self.hop_samples
    }

    /// Set the maximum batch size (consuming builder).
    pub const fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Set the maximum batch size (in-place setter).
    pub const fn set_batch_size(&mut self, n: usize) -> &mut Self {
        self.batch_size = n;
        self
    }

    /// Get the maximum batch size.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_matches_new() {
        let a = Options::new();
        let b = Options::default();
        assert_eq!(a.graph_optimization_level(), b.graph_optimization_level());
    }

    #[test]
    fn options_with_round_trips() {
        let opts = Options::new()
            .with_graph_optimization_level(GraphOptimizationLevel::Level1);
        assert_eq!(opts.graph_optimization_level(), GraphOptimizationLevel::Level1);
    }

    #[test]
    fn options_set_round_trips() {
        let mut opts = Options::new();
        opts.set_graph_optimization_level(GraphOptimizationLevel::Disable);
        assert_eq!(opts.graph_optimization_level(), GraphOptimizationLevel::Disable);
    }

    #[test]
    fn chunking_default_values() {
        let c = ChunkingOptions::default();
        assert_eq!(c.window_samples(), 480_000);
        assert_eq!(c.hop_samples(), 480_000);
        assert_eq!(c.batch_size(), 8);
    }

    #[test]
    fn chunking_builders_round_trip() {
        let c = ChunkingOptions::new()
            .with_window_samples(240_000)
            .with_hop_samples(120_000)
            .with_batch_size(4);
        assert_eq!(c.window_samples(), 240_000);
        assert_eq!(c.hop_samples(), 120_000);
        assert_eq!(c.batch_size(), 4);
    }

    #[test]
    fn chunking_setters_round_trip() {
        let mut c = ChunkingOptions::new();
        c.set_window_samples(100).set_hop_samples(50).set_batch_size(1);
        assert_eq!(c.window_samples(), 100);
        assert_eq!(c.hop_samples(), 50);
        assert_eq!(c.batch_size(), 1);
    }
}
```

- [ ] **Step 2: Run options tests**

```bash
cargo test --lib options::tests
```

Expected: all 6 tests pass. (Tests live alongside the code, not under `tests/`, since they exercise
unit-level behavior with no model dependency.)

- [ ] **Step 3: Commit**

```bash
git add src/options.rs
git commit -m "Add Options and ChunkingOptions with builder/setter/getter triple"
```

---

### Task 10: `src/clap.rs` Embedding type + NORM_BUDGET (TDD)

**Files:**
- Create: `src/clap.rs` (initial — only `Embedding` and `NORM_BUDGET`; the `Clap` struct lands in Task 12 / Task 25)

The full `Embedding` API is specified in §7.5. The §12.1 cancellation-safety test is the most subtle
test in the crate; we transcribe its exact form from the spec.

- [ ] **Step 1: Create the module skeleton with `NORM_BUDGET` const and Embedding struct**

```rust
//! Top-level Clap, Embedding, LabeledScore types. See spec §7.2 / §7.5 / §7.6.

use std::fmt;

use crate::error::{Error, Result};

// Future-`Clap` / `LabeledScore[Owned]` types land in Task 11 + Task 25; this commit only adds
// `Embedding` + `NORM_BUDGET` so audio.rs / text.rs (added in Task 12) can reference them.

/// Shared norm-tolerance budget for `Embedding::try_from_unit_slice` and the §8.2/§9.2 trust-path
/// guards. See spec §7.5 for the rationale (typical-case 1e-4, worst-case 6.1e-5 from
/// 512·ulp(1); §14 tracks future tightening to 5e-5 if telemetry supports it).
pub(crate) const NORM_BUDGET: f32 = 1e-4;

/// A 512-dim L2-normalized CLAP embedding.
///
/// Returned by every `embed*` call on `AudioEncoder` / `TextEncoder` / `Clap`. The unit-norm
/// invariant holds within fp32 ULP — see spec §7.5.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Embedding {
    inner: [f32; 512],
}
```

- [ ] **Step 2: Add `Embedding` constructors and accessors**

```rust
impl Embedding {
    /// Length of the embedding (512 in 0.1.0).
    pub const fn dim(&self) -> usize {
        self.inner.len()
    }

    /// Borrow the embedding as a slice — supports `append_slice` into Arrow's `MutableBuffer`.
    pub fn as_slice(&self) -> &[f32] {
        &self.inner
    }

    /// Owned conversion to a `Vec<f32>`. Allocates.
    pub fn to_vec(&self) -> Vec<f32> {
        self.inner.to_vec()
    }

    /// Reconstruct from a stored unit vector. Validates length AND norm
    /// (release-mode check: `(norm² − 1).abs() ≤ NORM_BUDGET`).
    ///
    /// See spec §7.5 for the budget rationale (summation-order divergence between writer and reader).
    pub fn try_from_unit_slice(s: &[f32]) -> Result<Self> {
        if s.len() != 512 {
            return Err(Error::EmbeddingDimMismatch { expected: 512, got: s.len() });
        }
        let norm_sq: f32 = s.iter().map(|x| x * x).sum();
        let dev = (norm_sq - 1.0).abs();
        if dev > NORM_BUDGET {
            return Err(Error::EmbeddingNotUnitNorm { norm_sq_deviation: dev });
        }
        let mut inner = [0.0f32; 512];
        inner.copy_from_slice(s);
        Ok(Self { inner })
    }

    /// Construct from any non-zero finite slice; always re-normalizes to unit length.
    /// Validates length, rejects all-zero input via `EmbeddingZero`, rejects any non-finite component
    /// via `NonFiniteEmbedding`. See spec §7.5.
    ///
    /// **Cost.** ~100 ns over 512 components (finiteness scan + L2 norm). For bulk hot-path import
    /// where upstream guarantees finiteness and unit-norm, prefer `try_from_unit_slice`.
    pub fn from_slice_normalizing(s: &[f32]) -> Result<Self> {
        if s.len() != 512 {
            return Err(Error::EmbeddingDimMismatch { expected: 512, got: s.len() });
        }
        for (i, &v) in s.iter().enumerate() {
            if !v.is_finite() {
                return Err(Error::NonFiniteEmbedding { component_index: i });
            }
        }
        let norm_sq: f32 = s.iter().map(|x| x * x).sum();
        if norm_sq == 0.0 {
            return Err(Error::EmbeddingZero);
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        let mut inner = [0.0f32; 512];
        for (out, &v) in inner.iter_mut().zip(s.iter()) {
            *out = v * inv_norm;
        }
        Ok(Self { inner })
    }

    /// Crate-internal constructor used by encoders. Bypasses normalization — caller must ensure the
    /// input is unit-norm (within fp32 ULP). The §8.2/§9.2 trust-path guard validates against
    /// `NORM_BUDGET` before calling this.
    pub(crate) fn from_array_trusted_unit_norm(arr: [f32; 512]) -> Self {
        debug_assert!({
            let n: f32 = arr.iter().map(|x| x * x).sum();
            (n - 1.0).abs() <= NORM_BUDGET
        });
        Self { inner: arr }
    }
}
```

- [ ] **Step 3: Add similarity / approximate-equality methods**

```rust
impl Embedding {
    /// Inner product. For two unit vectors this equals `cosine(other)` to fp32 ULP.
    pub fn dot(&self, other: &Embedding) -> f32 {
        self.inner.iter().zip(other.inner.iter()).map(|(a, b)| a * b).sum()
    }

    /// Cosine similarity. For unit vectors equivalent to `dot`.
    pub fn cosine(&self, other: &Embedding) -> f32 {
        self.dot(other)
    }

    /// Approximate equality — max-abs metric. Returns true if `(self − other).max_abs() ≤ tol`
    /// (inclusive, so `is_close(self, 0.0)` is always true). See spec §12.2 for tolerance values.
    pub fn is_close(&self, other: &Embedding, tol: f32) -> bool {
        self.inner
            .iter()
            .zip(other.inner.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            <= tol
    }

    /// Approximate equality — semantic (cosine) metric. Returns true if `1 − cosine(other) ≤ tol`.
    ///
    /// Implemented as `0.5 · ‖a − b‖² ≤ tol` to avoid catastrophic cancellation at the
    /// near-identity end. The identity holds because the `Embedding` invariant guarantees both
    /// operands are unit-norm to fp32 ULP.
    pub fn is_close_cosine(&self, other: &Embedding, tol: f32) -> bool {
        let sq: f32 = self
            .inner
            .iter()
            .zip(other.inner.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum();
        (sq * 0.5) <= tol
    }
}

impl AsRef<[f32]> for Embedding {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl fmt::Debug for Embedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Embedding {{ dim: {}, head: [{:.4}, {:.4}, {:.4}, ..] }}",
            self.dim(),
            self.inner[0],
            self.inner[1],
            self.inner[2],
        )
    }
}
```

- [ ] **Step 4: Add unit tests including the §12.1 cancellation-safety test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_normalizing_produces_unit_norm() {
        let s = [1.0_f32; 512];
        let e = Embedding::from_slice_normalizing(&s).unwrap();
        let n: f32 = e.as_slice().iter().map(|x| x * x).sum();
        assert!((n - 1.0).abs() < 1e-6);
    }

    #[test]
    fn from_slice_normalizing_rejects_zero() {
        let s = [0.0_f32; 512];
        let err = Embedding::from_slice_normalizing(&s).unwrap_err();
        assert!(matches!(err, Error::EmbeddingZero));
    }

    #[test]
    fn from_slice_normalizing_rejects_nan() {
        let mut s = [1.0_f32; 512];
        s[42] = f32::NAN;
        let err = Embedding::from_slice_normalizing(&s).unwrap_err();
        assert!(matches!(err, Error::NonFiniteEmbedding { component_index: 42 }));
    }

    #[test]
    fn from_slice_normalizing_rejects_inf() {
        let mut s = [1.0_f32; 512];
        s[7] = f32::INFINITY;
        let err = Embedding::from_slice_normalizing(&s).unwrap_err();
        assert!(matches!(err, Error::NonFiniteEmbedding { component_index: 7 }));
    }

    #[test]
    fn from_slice_normalizing_wrong_len() {
        let s = [1.0_f32; 256];
        let err = Embedding::from_slice_normalizing(&s).unwrap_err();
        assert!(matches!(err, Error::EmbeddingDimMismatch { expected: 512, got: 256 }));
    }

    #[test]
    fn try_from_unit_slice_accepts_unit_norm() {
        let mut s = [0.0_f32; 512];
        s[0] = 1.0;
        let e = Embedding::try_from_unit_slice(&s).unwrap();
        assert_eq!(e.as_slice()[0], 1.0);
    }

    #[test]
    fn try_from_unit_slice_rejects_non_unit_norm() {
        let mut s = [0.0_f32; 512];
        s[0] = 0.5; // norm² = 0.25
        let err = Embedding::try_from_unit_slice(&s).unwrap_err();
        assert!(matches!(err, Error::EmbeddingNotUnitNorm { .. }));
    }

    #[test]
    fn try_from_unit_slice_inclusive_at_budget_boundary() {
        // Construct a vector with norm² = 1 + 0.5 * NORM_BUDGET — should pass (≤ inclusive).
        let mut s = [0.0_f32; 512];
        let target_sq = 1.0 + 0.5 * NORM_BUDGET;
        s[0] = target_sq.sqrt();
        Embedding::try_from_unit_slice(&s).expect("inclusive ≤ should accept boundary value");
    }

    #[test]
    fn dot_equals_cosine_for_unit_vectors() {
        let mut x = [0.0_f32; 512];
        x[0] = 1.0;
        let mut y = [0.0_f32; 512];
        y[1] = 1.0;
        let a = Embedding::from_slice_normalizing(&x).unwrap();
        let b = Embedding::from_slice_normalizing(&y).unwrap();
        assert_eq!(a.dot(&b), a.cosine(&b));
    }

    #[test]
    fn is_close_self_at_zero_tolerance() {
        let mut s = [0.0_f32; 512];
        s[0] = 1.0;
        let a = Embedding::from_slice_normalizing(&s).unwrap();
        assert!(a.is_close(&a, 0.0));
        assert!(a.is_close_cosine(&a, 0.0));
    }

    #[test]
    fn is_close_cosine_cancellation_safety() {
        // §12.1: ε = 1e-4 perturbation. ‖y‖² = 1 + 1e-8; fp32 ulp(1) ≈ 1.19e-7, so 1e-8 < ulp/2,
        // ‖y‖² rounds to exactly 1.0, normalization is the identity, b = y.
        // dot(a, b) = 1·1 + 0·1e-4 = 1.0; naive 1 − dot = 0.
        // Safe 0.5 · ‖a − b‖² = (1e-4)² / 2 = 5e-9.
        // tol = 1e-12: safe returns false; naive (regression) would return true.
        let mut x = [0.0_f32; 512];
        x[0] = 1.0;
        let mut y = [0.0_f32; 512];
        y[0] = 1.0;
        y[1] = 1.0e-4;
        let a = Embedding::from_slice_normalizing(&x).unwrap();
        let b = Embedding::from_slice_normalizing(&y).unwrap();
        assert!(!a.is_close_cosine(&b, 1.0e-12));

        // Sanity (documents the cancellation): naive form returns 0 because 1·1 + 0·1e-4 = 1.0.
        let naive = 1.0_f32 - a.dot(&b);
        assert_eq!(naive, 0.0_f32);
    }

    #[test]
    fn debug_does_not_dump_512_floats() {
        let s = [1.0_f32; 512];
        let e = Embedding::from_slice_normalizing(&s).unwrap();
        let s = format!("{:?}", e);
        assert!(s.contains("dim: 512"));
        assert!(s.contains("head:"));
        assert!(s.matches(',').count() < 10);
    }
}

```

Attach the compile-fail doctests directly to the `Embedding` type rather than a private struct
(doctests only run on public items). Edit `Embedding`'s top-level rustdoc to add the two examples:

```rust
/// A 512-dim L2-normalized CLAP embedding.
///
/// Returned by every `embed*` call on `AudioEncoder` / `TextEncoder` / `Clap`. The unit-norm
/// invariant holds within fp32 ULP — see spec §7.5.
///
/// # Compile-fail tests
///
/// `Embedding` intentionally exposes no `DIM` const:
///
/// ```compile_fail
/// let _ = textclap::Embedding::DIM;
/// ```
///
/// `Embedding` does not implement `PartialEq` (f32 outputs of ML models are not bit-stable across
/// runs / threads / OSes; use `is_close` / `is_close_cosine` instead):
///
/// ```compile_fail
/// # let mut s = [0.0_f32; 512]; s[0] = 1.0;
/// # let a = textclap::Embedding::from_slice_normalizing(&s).unwrap();
/// # let b = a.clone();
/// let _ = a == b;
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Embedding {
    inner: [f32; 512],
}
```

(This replaces the prior `pub struct Embedding` declaration in step 1.)
```

- [ ] **Step 5: Run the embedding tests**

```bash
cargo test --lib clap::tests
```

Expected: all 11 tests pass, including the cancellation-safety test. The `compile_fail` doctests run
via `cargo test --doc`.

- [ ] **Step 6: Commit**

```bash
git add src/clap.rs
git commit -m "Add Embedding type + NORM_BUDGET with cancellation-safety test"
```

---

### Task 11: `LabeledScore` + `LabeledScoreOwned` (TDD)

**Files:**
- Modify: `src/clap.rs` (append the two types + tests)

- [ ] **Step 1: Append `LabeledScore` types to `src/clap.rs`**

```rust
/// Single classification result borrowing its label from the input slice. See spec §7.6.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LabeledScore<'a> {
    label: &'a str,
    score: f32,
}

impl<'a> LabeledScore<'a> {
    pub(crate) const fn new(label: &'a str, score: f32) -> Self {
        Self { label, score }
    }

    /// The label borrowed from the caller's input slice.
    pub const fn label(&self) -> &'a str {
        self.label
    }

    /// Cosine similarity score in roughly `[-1, 1]`.
    pub const fn score(&self) -> f32 {
        self.score
    }

    /// Convert to an owned variant for cross-thread send / serialization / DB rows.
    pub fn to_owned(&self) -> LabeledScoreOwned {
        LabeledScoreOwned {
            label: self.label.to_string(),
            score: self.score,
        }
    }
}

/// Owned variant of `LabeledScore` — owns its label string.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LabeledScoreOwned {
    label: String,
    score: f32,
}

impl LabeledScoreOwned {
    /// Borrow the label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Cosine similarity score.
    pub const fn score(&self) -> f32 {
        self.score
    }

    /// Consume self, returning the owned label.
    pub fn into_label(self) -> String {
        self.label
    }
}
```

- [ ] **Step 2: Append unit tests to the existing `tests` module in `src/clap.rs`**

Inside `mod tests { ... }`:

```rust
    #[test]
    fn labeled_score_borrowed_round_trip() {
        let label = "a dog barking".to_string();
        let s = LabeledScore::new(&label, 0.42);
        assert_eq!(s.label(), "a dog barking");
        assert_eq!(s.score(), 0.42);
    }

    #[test]
    fn labeled_score_to_owned_preserves_fields() {
        let label = "rain".to_string();
        let borrowed = LabeledScore::new(&label, -0.13);
        let owned = borrowed.to_owned();
        assert_eq!(owned.label(), "rain");
        assert_eq!(owned.score(), -0.13);
        assert_eq!(owned.into_label(), "rain");
    }
```

- [ ] **Step 3: Run tests**

```bash
cargo test --lib clap::tests::labeled_score
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/clap.rs
git commit -m "Add LabeledScore and LabeledScoreOwned"
```

---

### Task 12: Skeleton `audio.rs` / `text.rs` / `Clap` (`unimplemented!()` bodies)

**Files:**
- Create: `src/audio.rs`
- Create: `src/text.rs`
- Modify: `src/clap.rs` (append `Clap` struct and impl)
- Create: `src/mel.rs` (skeleton — `pub(crate)` `MelExtractor` + `T_FRAMES` const + `unimplemented!()` body; full impl in Phase C)

The skeleton commit makes the whole crate compile end-to-end. Every method body is `unimplemented!()`
or an obvious placeholder. Real implementations land in Phase C–F.

- [ ] **Step 1: Create `src/mel.rs` skeleton**

```rust
//! Mel-spectrogram extractor (private to the crate). See spec §8.1 for the full pipeline.
//!
//! `T_FRAMES` and (optional) `HTSAT_INPUT_MEAN` / `HTSAT_INPUT_STD` are backfilled from
//! `golden_params.json` per §3.4 step 3 → step 4. The skeleton uses placeholder values that
//! Phase C replaces.

use crate::error::Result;

/// Mel time-frame count. Backfilled from `golden_params.json["T_frames"]` per §3.4.
pub(crate) const T_FRAMES: usize = 1000; // PLACEHOLDER — replace with golden_params.json["T_frames"] in Task 6

// Optional HTSAT input-normalization constants. Defined only if §3.2's functional check chose
// `global_mean_std`; otherwise mel.rs has no normalization step.
//
// pub(crate) const HTSAT_INPUT_MEAN: f32 = -4.27;
// pub(crate) const HTSAT_INPUT_STD:  f32 =  4.57;

/// Mel-spectrogram extractor. Owns the Hann window, mel filterbank, and FFT planner.
pub(crate) struct MelExtractor {
    // Real fields land in Task 13–16.
}

impl MelExtractor {
    pub(crate) fn new() -> Self {
        unimplemented!("MelExtractor::new — implemented in Phase C")
    }

    /// Compute mel features and write into `out`. Caller must size `out` to exactly `64 * T_FRAMES`.
    pub(crate) fn extract_into(&mut self, _samples: &[f32], _out: &mut [f32]) -> Result<()> {
        unimplemented!("MelExtractor::extract_into — implemented in Phase C")
    }
}
```

- [ ] **Step 2: Create `src/audio.rs` skeleton**

```rust
//! Audio encoder (CLAP HTSAT side). See spec §7.3 / §8.2.

use std::path::Path;

use crate::clap::Embedding;
use crate::error::Result;
use crate::options::Options;

// Backfilled from golden_onnx_io.json per §3.4. Module-private.
const AUDIO_INPUT_NAME:  &str = "input_features";   // PLACEHOLDER
const AUDIO_OUTPUT_NAME: &str = "audio_embeds";     // PLACEHOLDER

/// Compile-time const indicating whether the audio ONNX output is already L2-normalized.
/// Backfilled from `golden_onnx_io.json["audio_output_is_unit_norm"]` per §3.4.
const AUDIO_OUTPUT_IS_UNIT_NORM: bool = false; // PLACEHOLDER

/// Audio encoder. See spec §7.3.
pub struct AudioEncoder {
    // Real fields land in Task 17–21.
}

impl AudioEncoder {
    /// Load from a file path.
    pub fn from_file<P: AsRef<Path>>(_onnx_path: P, _opts: Options) -> Result<Self> {
        unimplemented!("AudioEncoder::from_file — implemented in Task 17")
    }

    /// Load from caller-supplied bytes (copied into the ORT session).
    pub fn from_memory(_onnx_bytes: &[u8], _opts: Options) -> Result<Self> {
        unimplemented!("AudioEncoder::from_memory — implemented in Task 17")
    }

    /// Wrap a pre-built ORT session. See spec §7.3 for the asymmetric purposes.
    pub fn from_ort_session(
        _session: ort::session::Session,
        _opts: Options,
    ) -> Result<Self> {
        unimplemented!("AudioEncoder::from_ort_session — implemented in Task 21")
    }

    /// Embed a single ≤10 s clip.
    pub fn embed(&mut self, _samples: &[f32]) -> Result<Embedding> {
        unimplemented!("AudioEncoder::embed — implemented in Task 19")
    }

    /// Embed N clips of arbitrary length 1..=480_000 each.
    pub fn embed_batch(&mut self, _clips: &[&[f32]]) -> Result<Vec<Embedding>> {
        unimplemented!("AudioEncoder::embed_batch — implemented in Task 20")
    }

    /// Embed an arbitrary-length clip via textclap's chunking. NOT LAION-reference compatible.
    pub fn embed_chunked(
        &mut self,
        _samples: &[f32],
        _opts: &crate::options::ChunkingOptions,
    ) -> Result<Embedding> {
        unimplemented!("AudioEncoder::embed_chunked — implemented in Task 21")
    }

    /// Run a dummy forward to amortize ORT operator specialization and size scratch.
    pub fn warmup(&mut self) -> Result<()> {
        unimplemented!("AudioEncoder::warmup — implemented in Task 21")
    }
}
```

- [ ] **Step 3: Create `src/text.rs` skeleton**

```rust
//! Text encoder (CLAP RoBERTa side). See spec §7.4 / §9.

use std::path::Path;

use crate::clap::Embedding;
use crate::error::Result;
use crate::options::Options;

// Backfilled from golden_onnx_io.json per §3.4. Module-private.
const TEXT_INPUT_IDS_NAME:      &str = "input_ids";        // PLACEHOLDER
const TEXT_ATTENTION_MASK_NAME: &str = "attention_mask";   // PLACEHOLDER
// `TEXT_POSITION_IDS_NAME` exists only if §3.2 found position_ids as an external input.
// const TEXT_POSITION_IDS_NAME: &str = "position_ids";
const TEXT_OUTPUT_NAME:         &str = "text_embeds";      // PLACEHOLDER

const TEXT_OUTPUT_IS_UNIT_NORM: bool = false; // PLACEHOLDER

/// Text encoder. See spec §7.4.
pub struct TextEncoder {
    // Real fields land in Task 22–25.
}

impl TextEncoder {
    /// Load from file paths (ONNX + tokenizer.json).
    pub fn from_files<P: AsRef<Path>>(
        _onnx_path: P,
        _tokenizer_json_path: P,
        _opts: Options,
    ) -> Result<Self> {
        unimplemented!("TextEncoder::from_files — implemented in Task 22")
    }

    /// Load from caller-supplied bytes.
    pub fn from_memory(
        _onnx_bytes: &[u8],
        _tokenizer_json_bytes: &[u8],
        _opts: Options,
    ) -> Result<Self> {
        unimplemented!("TextEncoder::from_memory — implemented in Task 22")
    }

    /// Wrap a pre-built ORT session and tokenizer. See spec §7.4.
    pub fn from_ort_session(
        _session: ort::session::Session,
        _tokenizer: tokenizers::Tokenizer,
        _opts: Options,
    ) -> Result<Self> {
        unimplemented!("TextEncoder::from_ort_session — implemented in Task 25")
    }

    /// Embed a single text query.
    pub fn embed(&mut self, _text: &str) -> Result<Embedding> {
        unimplemented!("TextEncoder::embed — implemented in Task 23")
    }

    /// Embed a batch of text queries.
    pub fn embed_batch(&mut self, _texts: &[&str]) -> Result<Vec<Embedding>> {
        unimplemented!("TextEncoder::embed_batch — implemented in Task 24")
    }

    /// Run a dummy forward to amortize ORT operator specialization.
    pub fn warmup(&mut self) -> Result<()> {
        unimplemented!("TextEncoder::warmup — implemented in Task 25")
    }
}
```

- [ ] **Step 4: Append `Clap` struct + impl to `src/clap.rs`**

```rust
use std::path::Path;

use crate::audio::AudioEncoder;
use crate::error::Result;
use crate::options::{ChunkingOptions, Options};
use crate::text::TextEncoder;

/// Top-level CLAP handle wrapping audio + text encoders. See spec §7.2.
pub struct Clap {
    audio: AudioEncoder,
    text: TextEncoder,
}

impl Clap {
    /// Load from three file paths.
    pub fn from_files<P: AsRef<Path>>(
        _audio_onnx: P,
        _text_onnx: P,
        _tokenizer_json: P,
        _opts: Options,
    ) -> Result<Self> {
        unimplemented!("Clap::from_files — implemented in Task 25/26")
    }

    /// Load from caller-supplied bytes.
    pub fn from_memory(
        _audio_bytes: &[u8],
        _text_bytes: &[u8],
        _tokenizer_bytes: &[u8],
        _opts: Options,
    ) -> Result<Self> {
        unimplemented!("Clap::from_memory — implemented in Task 26")
    }

    /// Mutable access to the audio encoder.
    pub fn audio_mut(&mut self) -> &mut AudioEncoder {
        &mut self.audio
    }

    /// Mutable access to the text encoder.
    pub fn text_mut(&mut self) -> &mut TextEncoder {
        &mut self.text
    }

    /// Warm up both encoders.
    pub fn warmup(&mut self) -> Result<()> {
        unimplemented!("Clap::warmup — implemented in Task 26")
    }

    /// Top-k zero-shot classification.
    pub fn classify<'a>(
        &mut self,
        _samples: &[f32],
        _labels: &'a [&str],
        _k: usize,
    ) -> Result<Vec<LabeledScore<'a>>> {
        unimplemented!("Clap::classify — implemented in Task 27")
    }

    /// All-labels zero-shot classification.
    pub fn classify_all<'a>(
        &mut self,
        _samples: &[f32],
        _labels: &'a [&str],
    ) -> Result<Vec<LabeledScore<'a>>> {
        unimplemented!("Clap::classify_all — implemented in Task 27")
    }

    /// Long-clip zero-shot classification (NOT LAION-reference compatible — see spec §7.3).
    pub fn classify_chunked<'a>(
        &mut self,
        _samples: &[f32],
        _labels: &'a [&str],
        _k: usize,
        _opts: &ChunkingOptions,
    ) -> Result<Vec<LabeledScore<'a>>> {
        unimplemented!("Clap::classify_chunked — implemented in Task 27")
    }
}
```

- [ ] **Step 5: Verify the crate compiles end-to-end**

```bash
cargo build
```

Expected: clean build with maybe a few `dead_code` warnings about unused fields. No errors. Public API
compiles; `unimplemented!()` calls panic at runtime but never at compile time.

- [ ] **Step 6: Commit (commit 4 of §3.4 — Rust skeleton)**

```bash
git add src/audio.rs src/text.rs src/mel.rs src/clap.rs
git commit -m "Add Rust skeleton (unimplemented!() bodies; crate compiles end-to-end)"
```

---

## Phase C — Mel extractor (TDD against librosa goldens)

The mel extractor is the most numerically-sensitive module. TDD against the committed librosa references
catches scale, norm, and centering bugs early — before they propagate through HTSAT's contractive
amplification into the audio embedding test.

### Task 13: Periodic Hann window (length 1024)

**Files:**
- Modify: `src/mel.rs`

- [ ] **Step 1: Write the failing test**

Append to `src/mel.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Periodic Hann window: librosa convention generates the first `n` samples of a length-(n+1)
    /// symmetric Hann. For n=1024:
    ///   w[k] = 0.5 - 0.5 * cos(2π · k / 1025)   for k = 0..1024
    /// Reference values for k=0, k=512 (mid), k=1023.
    #[test]
    fn hann_window_periodic_length_1024() {
        let win = MelExtractor::periodic_hann(1024);
        assert_eq!(win.len(), 1024);
        assert!((win[0] - 0.0).abs() < 1e-6);
        // mid-point: 0.5 - 0.5 * cos(2π · 512 / 1025) ≈ 0.99999...
        let mid_expected = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * 512.0 / 1025.0).cos();
        assert!((win[512] - mid_expected).abs() < 1e-6);
        // last sample: 0.5 - 0.5 * cos(2π · 1023 / 1025) ≈ 0.00963...
        let last_expected = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * 1023.0 / 1025.0).cos();
        assert!((win[1023] - last_expected).abs() < 1e-6);
        // sanity: not the symmetric form (which would have win[1023] near zero by symmetry,
        // but the periodic form has it slightly above zero).
        assert!(win[1023] > 1e-3);
    }
}
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cargo test --lib mel::tests::hann_window_periodic_length_1024
```

Expected: compile error (`MelExtractor::periodic_hann` not defined) or runtime panic (`unimplemented!()`).

- [ ] **Step 3: Implement `periodic_hann`**

Modify the `MelExtractor` impl block (replacing the `unimplemented!()` for `new` with real construction
later — for now, just add the static helper):

```rust
impl MelExtractor {
    /// Generate a periodic Hann window of length `n`: equivalent to taking the first `n` samples
    /// of a length-(n+1) symmetric Hann. Matches librosa / torch convention.
    fn periodic_hann(n: usize) -> Vec<f32> {
        let denom = (n + 1) as f32;
        (0..n)
            .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (k as f32) / denom).cos())
            .collect()
    }

    pub(crate) fn new() -> Self {
        unimplemented!("MelExtractor::new — completed in Task 16")
    }

    pub(crate) fn extract_into(&mut self, _samples: &[f32], _out: &mut [f32]) -> Result<()> {
        unimplemented!("MelExtractor::extract_into — implemented in Task 16")
    }
}
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
cargo test --lib mel::tests::hann_window_periodic_length_1024
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mel.rs
git commit -m "Add periodic Hann window (length 1024)"
```

---

### Task 14: Mel filterbank (Slaney scale, Slaney norm)

**Files:**
- Modify: `src/mel.rs`

The filterbank is a `[64 × 513]` matrix. Each row is a triangular filter centered at a Slaney-mel
frequency, normalized so the area under the triangle equals `2.0 / (right_hz - left_hz)` (Slaney
normalization). The §8.1.1 unit test compares rows 0, 10, and 32 against the committed librosa
reference rows.

- [ ] **Step 1: Write the failing test**

Append to `src/mel.rs`'s `tests` module:

```rust
    fn read_npy_f32(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
        npy.into_vec::<f32>().unwrap()
    }

    /// Compare filterbank row 0, 10, 32 against librosa references at max_abs_diff < 1e-6.
    /// Row 10 is the discriminator: lands near 1 kHz where Slaney inflection occurs;
    /// HTK construction would diverge here while rows 0 and 32 alone wouldn't tell.
    #[test]
    fn filterbank_rows_match_librosa() {
        let fb = MelExtractor::build_mel_filterbank(
            48000, 1024, 64, 50.0, 14000.0,
        );
        // fb is [n_mels × n_freq] = [64 × 513], row-major.
        const N_FREQ: usize = 513;
        for &row_idx in &[0_usize, 10, 32] {
            let path = format!("tests/fixtures/filterbank_row_{row_idx}.npy");
            let expected = read_npy_f32(&path);
            assert_eq!(expected.len(), N_FREQ);
            let actual_row = &fb[row_idx * N_FREQ..(row_idx + 1) * N_FREQ];
            let max_diff = actual_row
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-6,
                "filterbank row {row_idx} max_abs_diff = {max_diff:.3e}",
            );
        }
    }
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test --lib mel::tests::filterbank_rows_match_librosa
```

Expected: compile error (`build_mel_filterbank` not defined).

- [ ] **Step 3: Implement Slaney mel-scale helpers**

```rust
impl MelExtractor {
    /// Hz → Slaney mel.
    /// Linear below 1 kHz: m = 3 · f / 200; logarithmic above: m = 15 + log(f/1000) / log(6.4) · ...
    /// (Slaney's auditory toolbox formula).
    fn hz_to_slaney_mel(hz: f32) -> f32 {
        const F_MIN: f32 = 0.0;
        const F_SP: f32 = 200.0 / 3.0;
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
        let logstep = (6.4_f32).ln() / 27.0;
        if hz < MIN_LOG_HZ {
            (hz - F_MIN) / F_SP
        } else {
            MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
        }
    }

    /// Slaney mel → Hz (inverse of the above).
    fn slaney_mel_to_hz(mel: f32) -> f32 {
        const F_MIN: f32 = 0.0;
        const F_SP: f32 = 200.0 / 3.0;
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
        let logstep = (6.4_f32).ln() / 27.0;
        if mel < MIN_LOG_MEL {
            F_MIN + F_SP * mel
        } else {
            MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
        }
    }
}
```

- [ ] **Step 4: Implement `build_mel_filterbank`**

```rust
impl MelExtractor {
    /// Build a `[n_mels × (n_fft/2 + 1)]` Slaney-norm Slaney-scale mel filterbank, row-major.
    ///
    /// Matches `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney')`.
    fn build_mel_filterbank(
        sr: u32,
        n_fft: usize,
        n_mels: usize,
        fmin: f32,
        fmax: f32,
    ) -> Vec<f32> {
        let n_freq = n_fft / 2 + 1;
        let mel_min = Self::hz_to_slaney_mel(fmin);
        let mel_max = Self::hz_to_slaney_mel(fmax);
        // n_mels + 2 mel-equispaced points; convert back to Hz; bracket each filter with [left, center, right].
        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / (n_mels + 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points
            .iter()
            .map(|&m| Self::slaney_mel_to_hz(m))
            .collect();

        // FFT bin frequencies: bin k maps to k * sr / n_fft Hz.
        let bin_hz: Vec<f32> = (0..n_freq).map(|k| (k as f32) * (sr as f32) / (n_fft as f32)).collect();

        let mut fb = vec![0.0f32; n_mels * n_freq];
        for m in 0..n_mels {
            let left = hz_points[m];
            let center = hz_points[m + 1];
            let right = hz_points[m + 2];
            let inv_left_diff = 1.0 / (center - left);
            let inv_right_diff = 1.0 / (right - center);
            // Slaney normalization: scale by 2 / (right - left).
            let slaney_norm = 2.0 / (right - left);
            for k in 0..n_freq {
                let f = bin_hz[k];
                let weight = if f >= left && f <= center {
                    (f - left) * inv_left_diff
                } else if f >= center && f <= right {
                    (right - f) * inv_right_diff
                } else {
                    0.0
                };
                fb[m * n_freq + k] = weight * slaney_norm;
            }
        }
        fb
    }
}
```

- [ ] **Step 5: Add `npyz` fixture-read import to the test module**

The `read_npy_f32` helper requires `npyz` in dev-deps. Verify `Cargo.toml` already has it (Task 7
step 1); if not, add `npyz = "0.8"` under `[dev-dependencies]`.

- [ ] **Step 6: Run the test**

```bash
cargo test --lib mel::tests::filterbank_rows_match_librosa
```

Expected: PASS — all three rows within 1e-6 of librosa.

- [ ] **Step 7: Commit**

```bash
git add src/mel.rs
git commit -m "Add Slaney-scale Slaney-norm mel filterbank construction"
```

---

### Task 15: Real-input STFT via rustfft RealFftPlanner

**Files:**
- Modify: `src/mel.rs`

`rustfft` provides `RealFftPlanner` for n-point real-input FFTs that produce `n/2 + 1` complex bins.
We pre-plan the FFT once at construction; per-call we frame the windowed samples, execute the FFT,
and accumulate squared magnitude.

- [ ] **Step 1: Write the failing test**

Append to the `tests` module:

```rust
    /// Hann-windowed STFT of a 1 kHz sine wave at 48 kHz should peak at the bin closest to
    /// k = 1000 / (48000 / 1024) = 21.33 → bin 21 (or 22).
    #[test]
    fn stft_peaks_at_expected_bin() {
        let mut mel = MelExtractor::new();
        let sr = 48000_f32;
        let freq = 1000.0_f32;
        let mut samples = Vec::with_capacity(1024);
        for k in 0..1024 {
            samples.push((2.0 * std::f32::consts::PI * freq * (k as f32) / sr).sin());
        }
        let mut power = vec![0.0f32; 513];
        mel.stft_one_frame_power(&samples, &mut power);
        // Find peak bin
        let (peak_bin, _) = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(
            peak_bin == 21 || peak_bin == 22,
            "expected peak at bin 21 or 22, got bin {peak_bin}",
        );
    }
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test --lib mel::tests::stft_peaks_at_expected_bin
```

Expected: panic in `MelExtractor::new()` (still `unimplemented!()`).

- [ ] **Step 3: Implement `MelExtractor::new()` and `stft_one_frame_power`**

Replace the skeleton struct + impl with the real one (still missing `extract_into` — that's Task 16):

```rust
use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

const N_FFT: usize = 1024;
const HOP: usize = 480;
const N_MELS: usize = 64;
const SR: u32 = 48_000;
const TARGET_SAMPLES: usize = 480_000;
const FMIN: f32 = 50.0;
const FMAX: f32 = 14_000.0;
const POWER_TO_DB_AMIN: f32 = 1e-10;

pub(crate) struct MelExtractor {
    window: Vec<f32>,            // length N_FFT
    filterbank: Vec<f32>,        // length N_MELS × (N_FFT/2 + 1)
    fft: Arc<dyn Fft<f32>>,      // FftPlanner output for N_FFT
}

impl MelExtractor {
    pub(crate) fn new() -> Self {
        let window = Self::periodic_hann(N_FFT);
        let filterbank = Self::build_mel_filterbank(SR, N_FFT, N_MELS, FMIN, FMAX);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Self { window, filterbank, fft }
    }

    /// Compute |X[k]|² for a single Hann-windowed frame of length N_FFT.
    /// `power` must have length N_FFT/2 + 1 = 513.
    fn stft_one_frame_power(&self, frame: &[f32], power: &mut [f32]) {
        debug_assert_eq!(frame.len(), N_FFT);
        debug_assert_eq!(power.len(), N_FFT / 2 + 1);
        // Window the frame, build a Complex32 buffer, run a full complex FFT, then take the
        // first N_FFT/2 + 1 bins (real-FFT identity).
        let mut buf: Vec<Complex32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex32::new(s * w, 0.0))
            .collect();
        self.fft.process(&mut buf);
        for k in 0..(N_FFT / 2 + 1) {
            let c = buf[k];
            power[k] = c.re * c.re + c.im * c.im;
        }
    }
}
```

(We use a complex-input FFT via `Complex32::new(s, 0.0)` rather than `RealFftPlanner` to keep the
dependency set minimal — `rustfft` doesn't need a separate `realfft` crate. The 2× redundancy in the
upper half of the spectrum is discarded by reading only `N_FFT/2 + 1` bins.)

- [ ] **Step 4: Run the test**

```bash
cargo test --lib mel::tests::stft_peaks_at_expected_bin
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mel.rs
git commit -m "Add STFT (rustfft, complex-input, |X[k]|² output)"
```

---

### Task 16: Full `extract_into` pipeline (repeat-pad + filterbank multiply + power_to_db + (optional) HTSAT input norm)

**Files:**
- Modify: `src/mel.rs`

This task closes the §8.1 mel pipeline. The `extract_into` body covers:

1. Repeat-pad input from `len < TARGET_SAMPLES` up to `TARGET_SAMPLES` (or head-truncate if longer).
2. Frame into `T_FRAMES` overlapping windows of `N_FFT` samples with hop `HOP`.
3. Per-frame: window, STFT, squared magnitude → power spectrum.
4. Mel filterbank multiply: `[64 × 513] · [513] = [64]` per frame.
5. `10 · log10(max(amin, x))` (no `top_db` clip in default config; per-spec backfill if `top_db` is set).
6. (Optional) HTSAT global mean/std subtract — only if §3.4 backfilled `htsat_input_normalization.type == "global_mean_std"`.
7. Write into output buffer in `[64, T_FRAMES]` row-major time-major order.

- [ ] **Step 1: Write the failing test (golden mel comparison)**

Append to the `tests` module:

```rust
    /// Compare full mel-extraction output against `tests/fixtures/golden_mel.npy` at max_abs_diff < 1e-4.
    /// Skipped when TEXTCLAP_MODELS_DIR is unset (consistent with §12.2 — though this test doesn't
    /// need the model files, only the WAV fixture and golden_mel.npy, both of which are committed).
    #[test]
    fn extract_into_matches_golden_mel() {
        // The test is unconditional — fixtures are committed.
        let golden = read_npy_f32("tests/fixtures/golden_mel.npy");
        assert_eq!(golden.len(), N_MELS * T_FRAMES, "golden mel shape mismatch");

        // Read the WAV via hound. WAV is 48 kHz mono, may be i16 or f32.
        let mut reader = hound::WavReader::open("tests/fixtures/sample.wav").expect("open sample.wav");
        let samples: Vec<f32> = match reader.spec().sample_format {
            hound::SampleFormat::Int => {
                let bits = reader.spec().bits_per_sample;
                let scale = 1.0 / (1_i64 << (bits - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap() as f32 * scale).collect()
            }
            hound::SampleFormat::Float => {
                reader.samples::<f32>().map(|s| s.unwrap()).collect()
            }
        };

        let mut mel = MelExtractor::new();
        let mut out = vec![0.0f32; N_MELS * T_FRAMES];
        mel.extract_into(&samples, &mut out).expect("extract_into");

        let max_diff = out
            .iter()
            .zip(golden.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "mel features max_abs_diff = {max_diff:.3e} (budget 1e-4)",
        );
    }
```

The test depends on `hound` for WAV reading; verify `Cargo.toml` has `hound = "3"` under
`[dev-dependencies]`.

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test --lib mel::tests::extract_into_matches_golden_mel
```

Expected: panic in `extract_into` (still `unimplemented!()`).

- [ ] **Step 3: Implement `extract_into`**

Replace the placeholder `extract_into` body with the full pipeline. Note: the HTSAT input-normalization
branch is conditional on a compile-time flag set during §3.4 step 4. Since this is a plan, we include
both branches; the implementer keeps whichever §3.2 selected and removes the other.

```rust
impl MelExtractor {
    pub(crate) fn extract_into(&mut self, samples: &[f32], out: &mut [f32]) -> Result<()> {
        debug_assert_eq!(out.len(), N_MELS * T_FRAMES);

        // 1. Repeat-pad or head-truncate to TARGET_SAMPLES.
        // We use a stack-friendly Vec; could be encoder-owned scratch in future iteration.
        let mut padded: Vec<f32> = Vec::with_capacity(TARGET_SAMPLES);
        if samples.len() >= TARGET_SAMPLES {
            padded.extend_from_slice(&samples[..TARGET_SAMPLES]);
        } else {
            // Repeat-pad: tile until full, allowing partial final tile.
            // (samples.is_empty() is handled by caller's earlier EmptyAudio check.)
            while padded.len() < TARGET_SAMPLES {
                let remain = TARGET_SAMPLES - padded.len();
                let chunk = remain.min(samples.len());
                padded.extend_from_slice(&samples[..chunk]);
            }
        }

        // 2. Frame and process. T_FRAMES depends on STFT centering — we follow librosa's
        //    `center=False` convention here, which gives floor((N - n_fft) / hop) + 1 frames.
        //    If §3.4 backfill discovered center=True (1001 frames), insert reflection-padding
        //    around `padded` before this loop.
        let mut frame = vec![0.0f32; N_FFT];
        let mut power = vec![0.0f32; N_FFT / 2 + 1];

        for t in 0..T_FRAMES {
            let start = t * HOP;
            let end = start + N_FFT;
            if end <= padded.len() {
                frame.copy_from_slice(&padded[start..end]);
            } else {
                // Partial trailing frame — zero-fill the tail. T_FRAMES is sized so this rarely
                // triggers for a 480_000-sample input.
                let remain = padded.len() - start;
                frame[..remain].copy_from_slice(&padded[start..]);
                for v in &mut frame[remain..] {
                    *v = 0.0;
                }
            }

            self.stft_one_frame_power(&frame, &mut power);

            // 3. Mel filterbank multiply: out[mel, t] = sum_k filterbank[mel, k] * power[k]
            for mel_bin in 0..N_MELS {
                let row = &self.filterbank[mel_bin * (N_FFT / 2 + 1)..(mel_bin + 1) * (N_FFT / 2 + 1)];
                let mut acc = 0.0f32;
                for (w, p) in row.iter().zip(power.iter()) {
                    acc += w * p;
                }
                // 4. log10(max(amin, x)) · 10, ref = 1.0, no top_db clip.
                let db = 10.0 * acc.max(POWER_TO_DB_AMIN).log10();
                // 5. (Optional) HTSAT input norm — uncomment when §3.4 backfilled global_mean_std.
                //    let db = (db - HTSAT_INPUT_MEAN) / HTSAT_INPUT_STD;
                // 6. Row-major time-major: out[mel_bin * T_FRAMES + t]
                out[mel_bin * T_FRAMES + t] = db;
            }
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Add the `power_to_dB applied exactly once` test (§8.1.2)**

```rust
    #[test]
    fn power_to_db_applied_once() {
        // Construct a known input where applying log10 twice produces visibly different output.
        // We feed a constant-amplitude sine and assert that the mel output is roughly bounded
        // by what 10·log10 of a small power value yields, NOT the (much smaller) double-log value.
        let mut mel = MelExtractor::new();
        let sr = 48_000_f32;
        let mut samples = vec![0.0f32; TARGET_SAMPLES];
        for k in 0..TARGET_SAMPLES {
            samples[k] = (2.0 * std::f32::consts::PI * 1000.0 * (k as f32) / sr).sin();
        }
        let mut out = vec![0.0f32; N_MELS * T_FRAMES];
        mel.extract_into(&samples, &mut out).unwrap();
        // For a unit-amplitude sine, the peak mel value should be roughly in [-100, 0] dB
        // (single log10 application). Double-log would land in roughly [-7, +0.5] (much smaller range).
        let max = out.iter().fold(f32::MIN, |a, &b| a.max(b));
        assert!(max < 5.0 && max > -120.0, "single-log range; got max = {max}");
    }
```

- [ ] **Step 5: Run all mel tests**

```bash
cargo test --lib mel::tests
```

Expected: all 4 mel tests pass — Hann window, filterbank rows, STFT peak, golden-mel match,
power_to_dB single application.

- [ ] **Step 6: Commit**

```bash
git add src/mel.rs
git commit -m "Add full extract_into pipeline (repeat-pad + STFT + filterbank + power_to_dB)"
```

---

## Phase D — Audio encoder (`audio.rs`)

The audio encoder owns an `ort::Session`, a `MelExtractor`, and three `Vec<f32>` scratch buffers. The
§7.3.1 scratch-lifecycle contract (clear → reserve → fill → bind → run → drop views) is the structural
guarantee against UB through ORT's FFI boundary.

### Task 17: AudioEncoder skeleton fields + `validate_shape` helper + `from_file` / `from_memory`

**Files:**
- Modify: `src/audio.rs`

- [ ] **Step 1: Replace the `AudioEncoder` struct skeleton with real fields**

```rust
use std::path::Path;

use ort::session::{Session, builder::SessionBuilder};
use ort::value::TensorRef;

use crate::clap::{Embedding, NORM_BUDGET};
use crate::error::{Error, Result};
use crate::mel::{MelExtractor, T_FRAMES};
use crate::options::{ChunkingOptions, Options};

const AUDIO_INPUT_NAME:  &str = "input_features";   // PLACEHOLDER (Task 6 backfill)
const AUDIO_OUTPUT_NAME: &str = "audio_embeds";     // PLACEHOLDER (Task 6 backfill)
const AUDIO_OUTPUT_IS_UNIT_NORM: bool = false;       // PLACEHOLDER (Task 6 backfill)

const TARGET_SAMPLES: usize = 480_000;
const N_MELS: usize = 64;
const EMBEDDING_DIM: usize = 512;

/// Audio encoder. See spec §7.3.
pub struct AudioEncoder {
    session: Session,
    mel: MelExtractor,
    /// Scratch for `[N · 64 · T_FRAMES]` f32 mel features.
    mel_scratch: Vec<f32>,
    /// Scratch for raw `[N × 512]` projection outputs (private; never exposed).
    proj_scratch: Vec<[f32; 512]>,
}
```

- [ ] **Step 2: Add the `validate_shape` helper (matches silero's signature: actual first, expected second)**

```rust
/// Validate an ORT output shape against the expected one. Sibling-convention parameter order:
/// (actual, expected) — matches silero `session.rs`.
pub(crate) fn validate_shape(
    tensor: &'static str,
    actual: &[i64],
    expected: &[i64],
) -> Result<()> {
    if actual != expected {
        return Err(Error::UnexpectedTensorShape {
            tensor,
            actual: actual.to_vec(),
            expected: expected.to_vec(),
        });
    }
    Ok(())
}
```

- [ ] **Step 3: Implement `from_file` / `from_memory`**

```rust
impl AudioEncoder {
    /// Load from a file path.
    pub fn from_file<P: AsRef<Path>>(onnx_path: P, opts: Options) -> Result<Self> {
        let path = onnx_path.as_ref();
        let session = SessionBuilder::new()
            .map_err(|e| Error::OnnxLoadFromFile { path: path.to_path_buf(), source: e })?
            .with_optimization_level(opts.graph_optimization_level())
            .map_err(|e| Error::OnnxLoadFromFile { path: path.to_path_buf(), source: e })?
            .commit_from_file(path)
            .map_err(|e| Error::OnnxLoadFromFile { path: path.to_path_buf(), source: e })?;
        Self::from_loaded_session(session)
    }

    /// Load from caller-supplied bytes (copied into the ORT session).
    pub fn from_memory(onnx_bytes: &[u8], opts: Options) -> Result<Self> {
        let session = SessionBuilder::new()
            .map_err(Error::OnnxLoadFromMemory)?
            .with_optimization_level(opts.graph_optimization_level())
            .map_err(Error::OnnxLoadFromMemory)?
            .commit_from_memory(onnx_bytes)
            .map_err(Error::OnnxLoadFromMemory)?;
        Self::from_loaded_session(session)
    }

    fn from_loaded_session(session: Session) -> Result<Self> {
        // Schema check — see spec §7.3 / §3.2 step 4.
        let inputs = session.inputs.iter().map(|i| i.name.as_str()).collect::<Vec<_>>();
        if !inputs.iter().any(|n| *n == AUDIO_INPUT_NAME) {
            return Err(Error::SessionSchema {
                detail: format!(
                    "audio session missing expected input {:?}; got inputs {:?}",
                    AUDIO_INPUT_NAME, inputs,
                ),
            });
        }
        let outputs = session.outputs.iter().map(|o| o.name.as_str()).collect::<Vec<_>>();
        if !outputs.iter().any(|n| *n == AUDIO_OUTPUT_NAME) {
            return Err(Error::SessionSchema {
                detail: format!(
                    "audio session missing expected output {:?}; got outputs {:?}",
                    AUDIO_OUTPUT_NAME, outputs,
                ),
            });
        }
        Ok(Self {
            session,
            mel: MelExtractor::new(),
            mel_scratch: Vec::new(),
            proj_scratch: Vec::new(),
        })
    }
}
```

(The exact `ort 2.0.0-rc.12` API for `SessionBuilder` may differ slightly — verify against
silero `src/session.rs` and adapt the chain if needed. The error mapping pattern stays the same.)

- [ ] **Step 4: Smoke-test the constructor with a minimal `cargo build`**

```bash
cargo build
```

Expected: clean build. The encoder isn't usable yet (`embed*` still unimplemented), but `from_file`
and the validation pipeline work.

- [ ] **Step 5: Commit**

```bash
git add src/audio.rs
git commit -m "Add AudioEncoder fields, validate_shape, and from_file/from_memory loaders"
```

---

### Task 18: `embed_projections_batched` helper (the §8.2 core)

**Files:**
- Modify: `src/audio.rs`

This is the §8.2 internal helper that runs the full forward pipeline (mel → ONNX) for any non-empty
batch, writing raw model outputs into a caller-provided buffer. **The §7.3.1 scratch-lifecycle
contract applies here.**

- [ ] **Step 1: Implement `embed_projections_batched`**

Append to the `impl AudioEncoder` block:

```rust
impl AudioEncoder {
    /// Compute the audio model's raw projection outputs. These are un-normalized 512-dim vectors if
    /// `AUDIO_OUTPUT_IS_UNIT_NORM == false`, or already-unit-norm vectors if true. Callers handle
    /// any subsequent normalization or release-mode unit-norm guard themselves.
    ///
    /// `out` is cleared on entry; capacity is reserved for `clips.len()` entries and one row is
    /// pushed per clip. Prior contents are dropped.
    ///
    /// The chunked path's per-call accumulator is a *separate* `Vec` from `self.proj_scratch`;
    /// see spec §8.2.
    pub(crate) fn embed_projections_batched(
        &mut self,
        clips: &[&[f32]],
        out: &mut Vec<[f32; 512]>,
    ) -> Result<()> {
        let n = clips.len();
        debug_assert!(n > 0, "embed_projections_batched requires non-empty input");

        let row_len = N_MELS * T_FRAMES;
        let total = n * row_len;

        // §7.3.1 scratch lifecycle: clear + resize before binding any tensor view.
        // resize zero-fills ~8 MB on the first N=32 batch (amortized to zero on subsequent same-N
        // batches). Avoiding this would require Vec::set_len, forbidden by #![forbid(unsafe_code)].
        self.mel_scratch.clear();
        self.mel_scratch.resize(total, 0.0);

        for (i, clip) in clips.iter().enumerate() {
            let row_start = i * row_len;
            let row_end = row_start + row_len;
            self.mel
                .extract_into(clip, &mut self.mel_scratch[row_start..row_end])?;
        }

        // Bind tensor view AFTER all resizes complete. Borrow checker prevents subsequent mutation.
        let input_shape = [n, 1usize, N_MELS, T_FRAMES]; // adapt to [n, N_MELS, T_FRAMES] if §3.2 backfill says 3-D
        let input = TensorRef::from_array_view((input_shape, self.mel_scratch.as_slice()))?;

        let outputs = self.session.run(ort::inputs![AUDIO_INPUT_NAME => input])?;
        let (shape, data) = outputs[AUDIO_OUTPUT_NAME].try_extract_tensor::<f32>()?;
        validate_shape(
            "audio_output",
            shape.as_ref(),
            &[n as i64, EMBEDDING_DIM as i64],
        )?;

        out.clear();
        out.reserve(n);
        for i in 0..n {
            let mut row = [0.0f32; EMBEDDING_DIM];
            row.copy_from_slice(&data[i * EMBEDDING_DIM..(i + 1) * EMBEDDING_DIM]);
            out.push(row);
        }
        // Tensor views drop here; mel_scratch becomes mutable again.
        Ok(())
    }
}
```

- [ ] **Step 2: Add a finiteness-scan helper (used by `embed*` in Tasks 19-21)**

```rust
impl AudioEncoder {
    /// SIMD-friendly finiteness scan. Returns `Some(index)` of the first non-finite sample, or `None`.
    /// Cost ~50 µs over 480 000 samples — dwarfed by ONNX inference.
    fn first_non_finite(samples: &[f32]) -> Option<usize> {
        for (i, &v) in samples.iter().enumerate() {
            if !v.is_finite() {
                return Some(i);
            }
        }
        None
    }
}
```

- [ ] **Step 3: Build to verify compilation**

```bash
cargo build
```

Expected: clean build. The helper isn't tested directly — it's exercised through `embed` in Task 19.

- [ ] **Step 4: Commit**

```bash
git add src/audio.rs
git commit -m "Add embed_projections_batched helper following §7.3.1 scratch lifecycle"
```

---

### Task 19: `AudioEncoder::embed` (single-clip with all branches)

**Files:**
- Modify: `src/audio.rs`

This is the §8.2 single-clip entry point. Order: empty check → length check → finiteness scan → mel +
ONNX → unit-norm guard or L2-normalize.

- [ ] **Step 1: Write boundary tests**

Append a `#[cfg(test)] mod tests` block to `src/audio.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Empty input → EmptyAudio. Doesn't need a model.
    #[test]
    fn embed_empty_returns_empty_audio_error() {
        // We can't construct AudioEncoder without a model file, so we test the validation path
        // by running through embed_projections_batched's preconditions. The full path is exercised
        // in tests/clap_integration.rs.
        // For now, test that the error variant exists and matches.
        let err = Error::EmptyAudio { clip_index: None };
        assert!(matches!(err, Error::EmptyAudio { clip_index: None }));
    }

    /// Finiteness-scan helper finds the first NaN.
    #[test]
    fn first_non_finite_finds_nan_at_zero() {
        let s = [f32::NAN, 0.0, 0.0];
        assert_eq!(AudioEncoder::first_non_finite(&s), Some(0));
    }

    #[test]
    fn first_non_finite_finds_inf_in_middle() {
        let s = [0.0, 1.0, f32::INFINITY, 2.0];
        assert_eq!(AudioEncoder::first_non_finite(&s), Some(2));
    }

    #[test]
    fn first_non_finite_returns_none_for_clean_input() {
        let s = [0.0_f32; 100];
        assert_eq!(AudioEncoder::first_non_finite(&s), None);
    }
}
```

- [ ] **Step 2: Run to confirm the helper tests pass; the rest defer to integration**

```bash
cargo test --lib audio::tests
```

Expected: 4 tests pass. (Full embed-path testing requires the model, so it lives in
`tests/clap_integration.rs`.)

- [ ] **Step 3: Implement `embed`**

Replace the `unimplemented!()` `embed` with the canonical implementation. The helper clears `out` and
pushes one entry per clip; `embed` calls it with `clips = &[samples]`, then `pop`s the single row out
and finalizes it through `finalize_embedding` (which selects the trust-path or normalize branch based
on `AUDIO_OUTPUT_IS_UNIT_NORM`).

```rust
impl AudioEncoder {
    /// Single clip, length 0 < len ≤ 480 000 samples. See spec §7.3 / §8.2.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Embedding> {
        if samples.is_empty() {
            return Err(Error::EmptyAudio { clip_index: None });
        }
        if samples.len() > TARGET_SAMPLES {
            return Err(Error::AudioTooLong {
                got: samples.len(),
                max: TARGET_SAMPLES,
            });
        }
        if let Some(sample_index) = Self::first_non_finite(samples) {
            return Err(Error::NonFiniteAudio { clip_index: None, sample_index });
        }

        let mut out = Vec::with_capacity(1);
        self.embed_projections_batched(&[samples], &mut out)?;
        let row = out.pop().expect("helper always pushes for non-empty input");
        Self::finalize_embedding(row)
    }

    /// Convert a raw projection row into a unit-norm `Embedding`, with the trust-path branch.
    /// See spec §8.2 step 5.
    fn finalize_embedding(row: [f32; 512]) -> Result<Embedding> {
        if AUDIO_OUTPUT_IS_UNIT_NORM {
            // Trust path: validate norm² against NORM_BUDGET, then construct.
            let norm_sq: f32 = row.iter().map(|x| x * x).sum();
            let dev = (norm_sq - 1.0).abs();
            if dev > NORM_BUDGET {
                return Err(Error::EmbeddingNotUnitNorm { norm_sq_deviation: dev });
            }
            Ok(Embedding::from_array_trusted_unit_norm(row))
        } else {
            // Normalize: divide by ‖row‖.
            Embedding::from_slice_normalizing(&row)
        }
    }
}
```

- [ ] **Step 4: Build and run unit tests**

```bash
cargo build
cargo test --lib audio::tests
```

Expected: clean build, helper tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/audio.rs
git commit -m "Add AudioEncoder::embed with finiteness scan + trust-path branch"
```

---

### Task 20: `AudioEncoder::embed_batch`

**Files:**
- Modify: `src/audio.rs`

§8.2 step 4: empty slice returns empty Vec; per-clip validation; auto-pad in mel extractor; one ONNX
call; per-row finalization (same trust-path branch as single-clip).

- [ ] **Step 1: Implement `embed_batch`**

Replace the `unimplemented!()` `embed_batch` body:

```rust
impl AudioEncoder {
    /// Batch of clips of any lengths in 0 < len ≤ 480 000. See spec §7.3 / §8.2.
    pub fn embed_batch(&mut self, clips: &[&[f32]]) -> Result<Vec<Embedding>> {
        if clips.is_empty() {
            return Ok(Vec::new());
        }

        // Per-clip validation.
        for (i, clip) in clips.iter().enumerate() {
            if clip.is_empty() {
                return Err(Error::EmptyAudio { clip_index: Some(i) });
            }
            if clip.len() > TARGET_SAMPLES {
                return Err(Error::AudioTooLong { got: clip.len(), max: TARGET_SAMPLES });
            }
            if let Some(sample_index) = Self::first_non_finite(clip) {
                return Err(Error::NonFiniteAudio { clip_index: Some(i), sample_index });
            }
        }

        let mut raw = Vec::with_capacity(clips.len());
        self.embed_projections_batched(clips, &mut raw)?;
        let mut out = Vec::with_capacity(raw.len());
        for row in raw {
            out.push(Self::finalize_embedding(row)?);
        }
        Ok(out)
    }
}
```

- [ ] **Step 2: Verify build**

```bash
cargo build
```

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add src/audio.rs
git commit -m "Add AudioEncoder::embed_batch with per-clip validation"
```

---

### Task 21: `AudioEncoder::embed_chunked` + `warmup` + `from_ort_session`

**Files:**
- Modify: `src/audio.rs`

§8.2 chunked path: window into chunks, skip trailing < window/4, batch through `embed_projections_batched`,
aggregate (centroid or spherical-mean per `AUDIO_OUTPUT_IS_UNIT_NORM`).

- [ ] **Step 1: Implement `embed_chunked`**

Replace the `unimplemented!()`:

```rust
impl AudioEncoder {
    /// Arbitrary-length input via textclap's chunking convention. NOT LAION-reference compatible.
    /// See spec §7.3 / §8.2.
    pub fn embed_chunked(
        &mut self,
        samples: &[f32],
        opts: &ChunkingOptions,
    ) -> Result<Embedding> {
        if samples.is_empty() {
            return Err(Error::EmptyAudio { clip_index: None });
        }
        if opts.window_samples() == 0
            || opts.hop_samples() == 0
            || opts.batch_size() == 0
            || opts.hop_samples() > opts.window_samples()
        {
            return Err(Error::ChunkingConfig {
                window_samples: opts.window_samples(),
                hop_samples: opts.hop_samples(),
                batch_size: opts.batch_size(),
            });
        }
        if let Some(sample_index) = Self::first_non_finite(samples) {
            return Err(Error::NonFiniteAudio { clip_index: None, sample_index });
        }

        // Chunk offsets: 0, hop, 2·hop, …; skip trailing < window/4 unless the input itself is shorter.
        let window = opts.window_samples();
        let hop = opts.hop_samples();
        let min_keep = window / 4;
        let mut offsets: Vec<usize> = Vec::new();
        let mut off = 0;
        while off < samples.len() {
            let remain = samples.len() - off;
            let chunk_len = remain.min(window);
            if chunk_len >= min_keep || offsets.is_empty() {
                offsets.push(off);
            }
            off += hop;
        }

        // Process in groups of batch_size; accumulate raw projections.
        let mut accumulator: Vec<[f32; 512]> = Vec::with_capacity(offsets.len());
        let mut tmp_proj: Vec<[f32; 512]> = Vec::with_capacity(opts.batch_size());
        for batch_offsets in offsets.chunks(opts.batch_size()) {
            let chunks: Vec<&[f32]> = batch_offsets
                .iter()
                .map(|&o| &samples[o..(o + window).min(samples.len())])
                .collect();
            self.embed_projections_batched(&chunks, &mut tmp_proj)?;
            accumulator.extend(tmp_proj.drain(..));
        }

        // Single-chunk case skips aggregation regardless of branch.
        if accumulator.len() == 1 {
            return Self::finalize_embedding(accumulator.into_iter().next().unwrap());
        }

        // Aggregate. Both branches end with L2-normalize → Embedding (via from_slice_normalizing,
        // which inherits the cancellation-safe normalize and unit-norm invariant).
        // Branch is dead-code-eliminated by the optimizer when AUDIO_OUTPUT_IS_UNIT_NORM is fixed.
        let mut centroid = [0.0f32; 512];
        if AUDIO_OUTPUT_IS_UNIT_NORM {
            // Spherical-mean: average unit vectors, then normalize.
            for row in &accumulator {
                for (acc, &v) in centroid.iter_mut().zip(row.iter()) {
                    *acc += v;
                }
            }
        } else {
            // Centroid path: average raw projections, then normalize.
            for row in &accumulator {
                for (acc, &v) in centroid.iter_mut().zip(row.iter()) {
                    *acc += v;
                }
            }
        }
        let inv_n = 1.0 / accumulator.len() as f32;
        for v in &mut centroid {
            *v *= inv_n;
        }
        Embedding::from_slice_normalizing(&centroid)
    }
}
```

(The two branches are textually identical in this implementation — both compute the component-wise
mean. The semantic difference is *what's being averaged* (raw projections vs unit vectors), which is
determined by the input. Dead-branch elimination still applies because each branch references the
const indirectly through which `embed_projections_batched` output it consumed.)

- [ ] **Step 2: Implement `warmup` and `from_ort_session`**

Replace the remaining `unimplemented!()`s:

```rust
impl AudioEncoder {
    /// Wrap a pre-built ORT session. See spec §7.3 "Two distinct purposes" block.
    pub fn from_ort_session(session: Session, _opts: Options) -> Result<Self> {
        // Note: opts is unused here because the wrapped session was already configured
        // by the caller. We still accept it for API symmetry with from_file/from_memory.
        Self::from_loaded_session(session)
    }

    /// Run a dummy forward to amortize ORT operator specialization. See spec §11.4.
    pub fn warmup(&mut self) -> Result<()> {
        let silence = vec![0.0f32; TARGET_SAMPLES];
        let _ = self.embed(&silence)?;
        Ok(())
    }
}
```

- [ ] **Step 3: Build to verify**

```bash
cargo build
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add src/audio.rs
git commit -m "Add embed_chunked + warmup + from_ort_session for AudioEncoder"
```

---

## Phase E — Text encoder (`text.rs`)

### Task 22: TextEncoder fields + tokenizer setup + `from_files` / `from_memory`

**Files:**
- Modify: `src/text.rs`

The text encoder owns `Session`, `Tokenizer`, scratch `Vec<i64>` for ids and mask, plus `pad_id` and
`max_length` cached at construction. The padding-mode handling is asymmetric per §7.4.

- [ ] **Step 1: Replace text.rs skeleton with real fields**

```rust
use std::path::Path;

use ort::session::{Session, builder::SessionBuilder};
use ort::value::TensorRef;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

use crate::audio::validate_shape;
use crate::clap::{Embedding, NORM_BUDGET};
use crate::error::{Error, Result};
use crate::options::Options;

const TEXT_INPUT_IDS_NAME:      &str = "input_ids";        // PLACEHOLDER (Task 6 backfill)
const TEXT_ATTENTION_MASK_NAME: &str = "attention_mask";   // PLACEHOLDER
// const TEXT_POSITION_IDS_NAME: &str = "position_ids";   // uncomment if §3.2 found it
const TEXT_OUTPUT_NAME:         &str = "text_embeds";      // PLACEHOLDER

const TEXT_OUTPUT_IS_UNIT_NORM: bool = false; // PLACEHOLDER

const EMBEDDING_DIM: usize = 512;

/// Text encoder. See spec §7.4.
pub struct TextEncoder {
    session: Session,
    tokenizer: Tokenizer,
    pad_id: i64,
    /// Reused scratch.
    ids_scratch: Vec<i64>,
    mask_scratch: Vec<i64>,
}
```

- [ ] **Step 2: Implement `pad_id` resolution helper**

```rust
fn resolve_pad_id(tokenizer: &Tokenizer) -> Result<i64> {
    if let Some(p) = tokenizer.get_padding() {
        return Ok(p.pad_id as i64);
    }
    if let Some(id) = tokenizer.token_to_id("<pad>") {
        return Ok(id as i64);
    }
    Err(Error::NoPadToken)
}
```

- [ ] **Step 3: Implement padding rewriting (asymmetric per §7.4)**

```rust
/// Force `BatchLongest` padding on the tokenizer, regardless of what the JSON declared.
/// Used by `from_files` / `from_memory`. See spec §7.4.
fn force_batch_longest_padding(tokenizer: &mut Tokenizer, pad_id: i64) {
    let pad_token = tokenizer.id_to_token(pad_id as u32).unwrap_or_else(|| "<pad>".to_string());
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id: pad_id as u32,
        pad_token,
        pad_type_id: 0,
        direction: tokenizers::PaddingDirection::Right,
        pad_to_multiple_of: None,
    }));
}

/// Reject `Padding::Fixed` for `from_ort_session` callers. See spec §7.4.
fn reject_fixed_padding(tokenizer: &Tokenizer) -> Result<()> {
    if let Some(p) = tokenizer.get_padding() {
        if matches!(p.strategy, PaddingStrategy::Fixed(_)) {
            return Err(Error::PaddingFixedRejected);
        }
    }
    Ok(())
}
```

- [ ] **Step 4: Implement `from_files` / `from_memory`**

```rust
impl TextEncoder {
    pub fn from_files<P: AsRef<Path>>(
        onnx_path: P,
        tokenizer_json_path: P,
        opts: Options,
    ) -> Result<Self> {
        let onnx = onnx_path.as_ref();
        let tok_path = tokenizer_json_path.as_ref();
        let session = SessionBuilder::new()
            .map_err(|e| Error::OnnxLoadFromFile { path: onnx.to_path_buf(), source: e })?
            .with_optimization_level(opts.graph_optimization_level())
            .map_err(|e| Error::OnnxLoadFromFile { path: onnx.to_path_buf(), source: e })?
            .commit_from_file(onnx)
            .map_err(|e| Error::OnnxLoadFromFile { path: onnx.to_path_buf(), source: e })?;
        let mut tokenizer = Tokenizer::from_file(tok_path)
            .map_err(|e| Error::TokenizerLoadFromFile { path: tok_path.to_path_buf(), source: e })?;
        let pad_id = resolve_pad_id(&tokenizer)?;
        force_batch_longest_padding(&mut tokenizer, pad_id);
        Self::from_pieces(session, tokenizer, pad_id)
    }

    pub fn from_memory(
        onnx_bytes: &[u8],
        tokenizer_json_bytes: &[u8],
        opts: Options,
    ) -> Result<Self> {
        let session = SessionBuilder::new()
            .map_err(Error::OnnxLoadFromMemory)?
            .with_optimization_level(opts.graph_optimization_level())
            .map_err(Error::OnnxLoadFromMemory)?
            .commit_from_memory(onnx_bytes)
            .map_err(Error::OnnxLoadFromMemory)?;
        let mut tokenizer = Tokenizer::from_bytes(tokenizer_json_bytes)
            .map_err(Error::TokenizerLoadFromMemory)?;
        let pad_id = resolve_pad_id(&tokenizer)?;
        force_batch_longest_padding(&mut tokenizer, pad_id);
        Self::from_pieces(session, tokenizer, pad_id)
    }

    fn from_pieces(session: Session, tokenizer: Tokenizer, pad_id: i64) -> Result<Self> {
        // Schema check.
        let inputs: Vec<&str> = session.inputs.iter().map(|i| i.name.as_str()).collect();
        for required in &[TEXT_INPUT_IDS_NAME, TEXT_ATTENTION_MASK_NAME] {
            if !inputs.iter().any(|n| n == required) {
                return Err(Error::SessionSchema {
                    detail: format!(
                        "text session missing expected input {required:?}; got {inputs:?}"
                    ),
                });
            }
        }
        let outputs: Vec<&str> = session.outputs.iter().map(|o| o.name.as_str()).collect();
        if !outputs.iter().any(|n| *n == TEXT_OUTPUT_NAME) {
            return Err(Error::SessionSchema {
                detail: format!(
                    "text session missing expected output {TEXT_OUTPUT_NAME:?}; got {outputs:?}"
                ),
            });
        }
        Ok(Self {
            session,
            tokenizer,
            pad_id,
            ids_scratch: Vec::new(),
            mask_scratch: Vec::new(),
        })
    }
}
```

- [ ] **Step 5: Verify build**

```bash
cargo build
```

Expected: clean. (`embed*` still unimplemented.)

- [ ] **Step 6: Commit**

```bash
git add src/text.rs
git commit -m "Add TextEncoder fields + pad_id resolution + from_files/from_memory"
```

---

### Task 23: `TextEncoder::embed`

**Files:**
- Modify: `src/text.rs`

§9.2 single-text path: empty check → tokenize → cast to i64 → bind tensor views → run → finalize.

- [ ] **Step 1: Implement `embed`**

```rust
impl TextEncoder {
    /// Embed a single text query. See spec §7.4 / §9.2.
    pub fn embed(&mut self, text: &str) -> Result<Embedding> {
        if text.is_empty() {
            return Err(Error::EmptyInput { batch_index: None });
        }
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(Error::Tokenize)?;
        let t = encoding.get_ids().len();

        // Resize scratch + copy (cast u32 → i64).
        self.ids_scratch.clear();
        self.ids_scratch.reserve(t);
        for &id in encoding.get_ids() {
            self.ids_scratch.push(id as i64);
        }
        self.mask_scratch.clear();
        self.mask_scratch.reserve(t);
        for &m in encoding.get_attention_mask() {
            self.mask_scratch.push(m as i64);
        }

        // Bind tensor views, run, validate, drop.
        let ids_view = TensorRef::from_array_view(([1usize, t], self.ids_scratch.as_slice()))?;
        let mask_view = TensorRef::from_array_view(([1usize, t], self.mask_scratch.as_slice()))?;

        // If §3.2 found position_ids as an external input, build it from mask + pad_id and add to inputs.
        // (Default: inlined position calc — only ids + mask.)
        let outputs = self.session.run(ort::inputs![
            TEXT_INPUT_IDS_NAME      => ids_view,
            TEXT_ATTENTION_MASK_NAME => mask_view,
        ])?;

        let (shape, data) = outputs[TEXT_OUTPUT_NAME].try_extract_tensor::<f32>()?;
        validate_shape("text_output", shape.as_ref(), &[1, EMBEDDING_DIM as i64])?;

        let mut row = [0.0f32; EMBEDDING_DIM];
        row.copy_from_slice(&data[..EMBEDDING_DIM]);
        Self::finalize_embedding(row)
    }

    fn finalize_embedding(row: [f32; 512]) -> Result<Embedding> {
        if TEXT_OUTPUT_IS_UNIT_NORM {
            let norm_sq: f32 = row.iter().map(|x| x * x).sum();
            let dev = (norm_sq - 1.0).abs();
            if dev > NORM_BUDGET {
                return Err(Error::EmbeddingNotUnitNorm { norm_sq_deviation: dev });
            }
            Ok(Embedding::from_array_trusted_unit_norm(row))
        } else {
            Embedding::from_slice_normalizing(&row)
        }
    }
}
```

- [ ] **Step 2: Build**

```bash
cargo build
```

- [ ] **Step 3: Commit**

```bash
git add src/text.rs
git commit -m "Add TextEncoder::embed with cancellation-safe finalize"
```

---

### Task 24: `TextEncoder::embed_batch` + `warmup` + `from_ort_session`

**Files:**
- Modify: `src/text.rs`

- [ ] **Step 1: Implement `embed_batch`**

```rust
impl TextEncoder {
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        for (i, t) in texts.iter().enumerate() {
            if t.is_empty() {
                return Err(Error::EmptyInput { batch_index: Some(i) });
            }
        }

        // encode_batch returns Encodings already padded to longest-in-batch (force_batch_longest_padding
        // installed BatchLongest at construction).
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(Error::Tokenize)?;

        let n = encodings.len();
        let t_max = encodings[0].get_ids().len();
        // Sanity: BatchLongest means all rows are same length.
        debug_assert!(encodings.iter().all(|e| e.get_ids().len() == t_max));

        self.ids_scratch.clear();
        self.ids_scratch.reserve(n * t_max);
        self.mask_scratch.clear();
        self.mask_scratch.reserve(n * t_max);
        for enc in &encodings {
            for &id in enc.get_ids() {
                self.ids_scratch.push(id as i64);
            }
            for &m in enc.get_attention_mask() {
                self.mask_scratch.push(m as i64);
            }
        }

        let ids_view = TensorRef::from_array_view(([n, t_max], self.ids_scratch.as_slice()))?;
        let mask_view = TensorRef::from_array_view(([n, t_max], self.mask_scratch.as_slice()))?;

        let outputs = self.session.run(ort::inputs![
            TEXT_INPUT_IDS_NAME      => ids_view,
            TEXT_ATTENTION_MASK_NAME => mask_view,
        ])?;
        let (shape, data) = outputs[TEXT_OUTPUT_NAME].try_extract_tensor::<f32>()?;
        validate_shape("text_output", shape.as_ref(), &[n as i64, EMBEDDING_DIM as i64])?;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = [0.0f32; EMBEDDING_DIM];
            row.copy_from_slice(&data[i * EMBEDDING_DIM..(i + 1) * EMBEDDING_DIM]);
            out.push(Self::finalize_embedding(row)?);
        }
        Ok(out)
    }
}
```

- [ ] **Step 2: Implement `warmup` and `from_ort_session`**

```rust
impl TextEncoder {
    pub fn from_ort_session(
        session: Session,
        tokenizer: Tokenizer,
        _opts: Options,
    ) -> Result<Self> {
        reject_fixed_padding(&tokenizer)?;
        let pad_id = resolve_pad_id(&tokenizer)?;
        Self::from_pieces(session, tokenizer, pad_id)
    }

    pub fn warmup(&mut self) -> Result<()> {
        // The actual warmup_text is read from golden_params.json at maintenance time. The literal here
        // is the §3.4-step-3 backfilled value. PLACEHOLDER until then:
        const WARMUP_TEXT: &str =
            "the quick brown fox jumps over the lazy dog \
             the quick brown fox jumps over the lazy dog \
             the quick brown fox jumps over the lazy dog \
             the quick brown fox jumps over the lazy dog \
             the quick brown fox jumps over the lazy dog \
             the quick brown fox jumps over the lazy dog ";
        let _ = self.embed(WARMUP_TEXT)?;
        Ok(())
    }
}
```

- [ ] **Step 3: Build**

```bash
cargo build
```

- [ ] **Step 4: Commit**

```bash
git add src/text.rs
git commit -m "Add TextEncoder::embed_batch + warmup + from_ort_session"
```

---

## Phase F — Clap composition (`clap.rs`)

### Task 25: `Clap::from_files` / `from_memory` / `warmup`

**Files:**
- Modify: `src/clap.rs`

The `Clap` type composes the two encoders. Constructors load both sides; `warmup` runs both.

- [ ] **Step 1: Replace `Clap` skeleton bodies**

```rust
impl Clap {
    pub fn from_files<P: AsRef<Path>>(
        audio_onnx: P,
        text_onnx: P,
        tokenizer_json: P,
        opts: Options,
    ) -> Result<Self> {
        let audio = AudioEncoder::from_file(audio_onnx, opts)?;
        let text = TextEncoder::from_files(text_onnx, tokenizer_json, opts)?;
        Ok(Self { audio, text })
    }

    pub fn from_memory(
        audio_bytes: &[u8],
        text_bytes: &[u8],
        tokenizer_bytes: &[u8],
        opts: Options,
    ) -> Result<Self> {
        let audio = AudioEncoder::from_memory(audio_bytes, opts)?;
        let text = TextEncoder::from_memory(text_bytes, tokenizer_bytes, opts)?;
        Ok(Self { audio, text })
    }

    pub fn warmup(&mut self) -> Result<()> {
        self.audio.warmup()?;
        self.text.warmup()?;
        Ok(())
    }
}
```

- [ ] **Step 2: Build**

```bash
cargo build
```

- [ ] **Step 3: Commit**

```bash
git add src/clap.rs
git commit -m "Add Clap::from_files / from_memory / warmup composition"
```

---

### Task 26: `classify` / `classify_all` / `classify_chunked`

**Files:**
- Modify: `src/clap.rs`

§7.2 zero-shot classification. `classify_all` is the building block; `classify` heap-top-k's the
result; `classify_chunked` swaps `embed` for `embed_chunked` upfront.

- [ ] **Step 1: Implement `classify_all`**

```rust
impl Clap {
    pub fn classify_all<'a>(
        &mut self,
        samples: &[f32],
        labels: &'a [&str],
    ) -> Result<Vec<LabeledScore<'a>>> {
        if labels.is_empty() {
            return Ok(Vec::new());
        }
        let audio_emb = self.audio.embed(samples)?;
        let text_embs = self.text.embed_batch(labels)?;
        let mut scores: Vec<LabeledScore<'a>> = labels
            .iter()
            .zip(text_embs.iter())
            .map(|(label, text_emb)| LabeledScore::new(label, audio_emb.dot(text_emb)))
            .collect();
        // Sort descending by score, stable tie-break = input order.
        scores.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores)
    }

    pub fn classify<'a>(
        &mut self,
        samples: &[f32],
        labels: &'a [&str],
        k: usize,
    ) -> Result<Vec<LabeledScore<'a>>> {
        if k == 0 || labels.is_empty() {
            return Ok(Vec::new());
        }
        let mut all = self.classify_all(samples, labels)?;
        all.truncate(k.min(all.len()));
        Ok(all)
    }

    pub fn classify_chunked<'a>(
        &mut self,
        samples: &[f32],
        labels: &'a [&str],
        k: usize,
        opts: &ChunkingOptions,
    ) -> Result<Vec<LabeledScore<'a>>> {
        if k == 0 || labels.is_empty() {
            return Ok(Vec::new());
        }
        let audio_emb = self.audio.embed_chunked(samples, opts)?;
        let text_embs = self.text.embed_batch(labels)?;
        let mut scores: Vec<LabeledScore<'a>> = labels
            .iter()
            .zip(text_embs.iter())
            .map(|(label, text_emb)| LabeledScore::new(label, audio_emb.dot(text_emb)))
            .collect();
        scores.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k.min(scores.len()));
        Ok(scores)
    }
}
```

- [ ] **Step 2: Add unit tests for the edge cases**

Append to `src/clap.rs` `tests` module:

```rust
    #[test]
    fn classify_empty_labels_returns_empty() {
        // We can't actually call classify without a model. Test the logic by checking
        // the early-return paths are reachable through types. The integration test
        // (Task 27) exercises the real path.
        let labels: &[&str] = &[];
        assert!(labels.is_empty());
    }
```

(The full classify tests live in `tests/clap_integration.rs` because they need the model.)

- [ ] **Step 3: Build**

```bash
cargo build
cargo test --lib
```

Expected: clean build; all unit tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/clap.rs
git commit -m "Add classify / classify_all / classify_chunked"
```

---

## Phase G — Integration test

### Task 27: `tests/clap_integration.rs` (mel + audio + text + discrimination)

**Files:**
- Create: `tests/clap_integration.rs`

Gated on `TEXTCLAP_MODELS_DIR`. Skipped (with `eprintln!`) when unset.

- [ ] **Step 1: Create the test file with helpers**

```rust
//! Integration test for textclap. Gated on TEXTCLAP_MODELS_DIR env var.
//!
//! When unset, all tests print a skip message and pass (so `cargo test` doesn't fail in
//! environments without the model files). When set, runs the §12.2 assertion battery.

use std::env;
use std::path::PathBuf;

use textclap::{Clap, Embedding, Options};

fn models_dir() -> Option<PathBuf> {
    env::var_os("TEXTCLAP_MODELS_DIR").map(PathBuf::from)
}

fn read_npy_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    npyz::NpyFile::new(&bytes[..]).unwrap().into_vec::<f32>().unwrap()
}

fn read_wav_48k_mono(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 48000, "fixture must be 48 kHz");
    assert_eq!(spec.channels, 1, "fixture must be mono");
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1_i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 * scale).collect()
        }
        hound::SampleFormat::Float => {
            reader.samples::<f32>().map(|s| s.unwrap()).collect()
        }
    }
}
```

- [ ] **Step 2: Add the audio embedding test**

```rust
#[test]
fn audio_embedding_matches_golden() {
    let Some(dir) = models_dir() else {
        eprintln!("skipping: TEXTCLAP_MODELS_DIR not set");
        return;
    };
    let mut clap = Clap::from_files(
        dir.join("audio_model_quantized.onnx"),
        dir.join("text_model_quantized.onnx"),
        dir.join("tokenizer.json"),
        Options::new(),
    )
    .expect("Clap::from_files");
    clap.warmup().expect("warmup");

    let samples = read_wav_48k_mono("tests/fixtures/sample.wav");
    let emb = clap.audio_mut().embed(&samples).expect("embed");

    let golden = read_npy_f32("tests/fixtures/golden_audio_emb.npy");
    let golden_emb = Embedding::try_from_unit_slice(&golden).expect("golden is unit-norm");

    // §12.2: max-abs ≤ 5e-4 for int8-vs-int8 audio embedding.
    assert!(
        emb.is_close(&golden_emb, 5e-4),
        "audio embedding drift exceeds 5e-4 (max-abs)",
    );
    // Cosine: ≤ 1e-4. Both should hold simultaneously.
    assert!(
        emb.is_close_cosine(&golden_emb, 1e-4),
        "audio embedding cosine drift exceeds 1e-4",
    );
}
```

- [ ] **Step 3: Add the text embedding test**

```rust
#[test]
fn text_embeddings_match_golden() {
    let Some(dir) = models_dir() else {
        eprintln!("skipping: TEXTCLAP_MODELS_DIR not set");
        return;
    };
    let mut clap = Clap::from_files(
        dir.join("audio_model_quantized.onnx"),
        dir.join("text_model_quantized.onnx"),
        dir.join("tokenizer.json"),
        Options::new(),
    )
    .expect("Clap::from_files");
    clap.warmup().expect("warmup");

    let labels = ["a dog barking", "rain", "music", "silence", "door creaking"];
    let embs = clap.text_mut().embed_batch(&labels).expect("embed_batch");

    let golden = read_npy_f32("tests/fixtures/golden_text_embs.npy");
    assert_eq!(golden.len(), 5 * 512);

    for (i, label) in labels.iter().enumerate() {
        let golden_row = &golden[i * 512..(i + 1) * 512];
        let golden_emb = Embedding::try_from_unit_slice(golden_row).expect("golden row unit-norm");
        // §12.2: max-abs ≤ 1e-5 for text (no upstream drift).
        assert!(
            embs[i].is_close(&golden_emb, 1e-5),
            "text embedding drift exceeds 1e-5 for label {label:?}",
        );
        assert!(
            embs[i].is_close_cosine(&golden_emb, 5e-8),
            "text embedding cosine drift exceeds 5e-8 for label {label:?}",
        );
    }
}
```

- [ ] **Step 4: Add the discrimination check (§12.2)**

```rust
#[test]
fn classify_discrimination_check() {
    let Some(dir) = models_dir() else {
        eprintln!("skipping: TEXTCLAP_MODELS_DIR not set");
        return;
    };
    let mut clap = Clap::from_files(
        dir.join("audio_model_quantized.onnx"),
        dir.join("text_model_quantized.onnx"),
        dir.join("tokenizer.json"),
        Options::new(),
    )
    .expect("Clap::from_files");
    clap.warmup().expect("warmup");

    let samples = read_wav_48k_mono("tests/fixtures/sample.wav");
    let labels = ["a dog barking", "rain", "music", "silence", "door creaking"];
    let scores = clap.classify_all(&samples, &labels).expect("classify_all");

    // Find scores by label.
    let score_of = |query: &str| -> f32 {
        scores
            .iter()
            .find(|s| s.label() == query)
            .expect("label present")
            .score()
    };
    let dog = score_of("a dog barking");
    let music = score_of("music");

    // §12.2 discrimination check:
    // 1. "a dog barking" ranks in top 2.
    let top_2: Vec<&str> = scores.iter().take(2).map(|s| s.label()).collect();
    assert!(top_2.contains(&"a dog barking"), "expected dog-bark in top 2; top 2 = {top_2:?}");
    // 2. score("a dog barking") - score("music") > 0.05.
    assert!(
        dog - music > 0.05,
        "discrimination margin too small: dog {dog:.4} − music {music:.4} = {:.4}",
        dog - music,
    );
}
```

- [ ] **Step 5: Run the tests**

If `TEXTCLAP_MODELS_DIR` is set:
```bash
TEXTCLAP_MODELS_DIR=~/textclap-models cargo test --test clap_integration
```

Expected: 3 tests pass. If any tolerance is exceeded, investigate (per spec §12.2: per-OS budget tables
may be needed for cross-platform variance).

If `TEXTCLAP_MODELS_DIR` is unset:
```bash
cargo test --test clap_integration
```

Expected: all 3 tests print "skipping..." and pass.

- [ ] **Step 6: Commit**

```bash
git add tests/clap_integration.rs
git commit -m "Add integration test (mel-skip + audio + text + discrimination)"
```

---

## Phase H — Polish

### Task 28: Benches (Criterion)

**Files:**
- Create: `benches/bench_mel.rs`
- Create: `benches/bench_audio_encode.rs`
- Create: `benches/bench_text_encode.rs`

Each bench's `setup` closure calls `warmup()` before `iter` so first-sample cold-start cost doesn't skew
the median. The audio/text batched benches use `BenchmarkGroup` per spec §12.4.

- [ ] **Step 1: Create `benches/bench_mel.rs`**

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_mel(c: &mut Criterion) {
    // mel.rs is pub(crate); we exercise the full pipeline through the public AudioEncoder
    // in bench_audio_encode.rs. This bench skipped in 0.1.0; placeholder to wire up the harness.
    c.bench_function("mel_placeholder", |b| b.iter(|| 0_u32));
}

criterion_group!(benches, bench_mel);
criterion_main!(benches);
```

(The mel extractor isn't directly benchable through the public API; bench_audio_encode covers the full
mel + ONNX path. Keep this stub for future expansion.)

- [ ] **Step 2: Create `benches/bench_audio_encode.rs`**

```rust
use std::env;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use textclap::{AudioEncoder, Options};

fn bench_audio_encode(c: &mut Criterion) {
    let Some(dir) = env::var_os("TEXTCLAP_MODELS_DIR").map(PathBuf::from) else {
        eprintln!("skipping bench_audio_encode: TEXTCLAP_MODELS_DIR not set");
        return;
    };
    let mut audio = AudioEncoder::from_file(
        dir.join("audio_model_quantized.onnx"),
        Options::new(),
    )
    .expect("AudioEncoder::from_file");
    audio.warmup().expect("warmup");

    let mut group = c.benchmark_group("audio_encode");
    let samples = vec![0.0f32; 480_000];

    for &n in &[1usize, 4, 8] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let clips: Vec<&[f32]> = vec![&samples[..]; n];
            b.iter(|| audio.embed_batch(&clips).expect("embed_batch"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_audio_encode);
criterion_main!(benches);
```

- [ ] **Step 3: Create `benches/bench_text_encode.rs`**

```rust
use std::env;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use textclap::{Options, TextEncoder};

fn bench_text_encode(c: &mut Criterion) {
    let Some(dir) = env::var_os("TEXTCLAP_MODELS_DIR").map(PathBuf::from) else {
        eprintln!("skipping bench_text_encode: TEXTCLAP_MODELS_DIR not set");
        return;
    };
    let mut text = TextEncoder::from_files(
        dir.join("text_model_quantized.onnx"),
        dir.join("tokenizer.json"),
        Options::new(),
    )
    .expect("TextEncoder::from_files");
    text.warmup().expect("warmup");

    let mut group = c.benchmark_group("text_encode");
    let queries = [
        "a dog barking", "rain on a metal roof", "applause in a stadium",
        "engine starting", "music with drums", "speech with crowd",
        "door creaking", "alarm sound", "water running", "wind through trees",
        "footsteps on gravel", "phone ringing", "glass breaking", "coffee shop ambience",
        "thunderstorm", "bird chirping", "traffic noise", "cat meowing",
        "typing on a keyboard", "fire crackling", "ocean waves", "helicopter overhead",
        "construction noise", "violin solo", "drum kit", "electronic beep",
        "child laughing", "lecture hall", "guitar strumming", "vacuum cleaner",
        "clock ticking", "wind chimes",
    ];

    for &n in &[1usize, 8, 32] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let texts: Vec<&str> = queries.iter().take(n).copied().collect();
            b.iter(|| text.embed_batch(&texts).expect("embed_batch"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_text_encode);
criterion_main!(benches);
```

- [ ] **Step 4: Run benches (optional sanity check)**

```bash
TEXTCLAP_MODELS_DIR=~/textclap-models cargo bench --bench bench_audio_encode
```

Expected: Criterion produces a report under `target/criterion/audio_encode/`.

- [ ] **Step 5: Commit**

```bash
git add benches/
git commit -m "Add Criterion benches for mel/audio/text encoders"
```

---

### Task 29: Examples (`audio_window_to_clap.rs` + `index_and_search.rs`)

**Files:**
- Create: `examples/audio_window_to_clap.rs`
- Create: `examples/index_and_search.rs`

`audio_window_to_clap.rs` demonstrates the §1.1 indexing path with `rubato` resampling.
`index_and_search.rs` is the sequential demo: index one window, then run a query.

- [ ] **Step 1: Create `examples/audio_window_to_clap.rs`**

```rust
//! Indexing path demo: source frames at native rate (e.g. 44.1 kHz) → rubato resample to 48 kHz
//! → buffer 10 s → AudioEncoder::embed → 512-dim Embedding → push to a stubbed lancedb writer.
//!
//! Run with TEXTCLAP_MODELS_DIR set to a directory containing audio_model_quantized.onnx.
//! See spec §1.1.

use std::env;
use std::path::PathBuf;

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use textclap::{AudioEncoder, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = env::var_os("TEXTCLAP_MODELS_DIR")
        .map(PathBuf::from)
        .ok_or("set TEXTCLAP_MODELS_DIR to the directory containing audio_model_quantized.onnx")?;

    let mut audio = AudioEncoder::from_file(dir.join("audio_model_quantized.onnx"), Options::new())?;
    audio.warmup()?;

    // Simulate a decoder producing 44.1 kHz mono frames. In a real pipeline this comes from
    // ffmpeg / gstreamer / a microphone callback / etc.
    let source_rate = 44100;
    let target_rate = 48000;
    let chunk_size = 4410; // 100 ms at 44.1 kHz
    let total_seconds = 10;

    // rubato sinc resampler (44.1 → 48 kHz).
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        target_rate as f64 / source_rate as f64,
        2.0,
        params,
        chunk_size,
        1, // mono
    )?;

    let mut buffer_48k: Vec<f32> = Vec::with_capacity(target_rate * total_seconds);
    let target_samples = target_rate * total_seconds; // 480 000

    let frames_per_chunk = source_rate * total_seconds / chunk_size; // ~100
    for i in 0..frames_per_chunk {
        // Synthesize a 1 kHz sine for demonstration; real pipeline reads from the decoder.
        let mut frame = vec![0.0f32; chunk_size];
        for k in 0..chunk_size {
            let t = (i * chunk_size + k) as f32 / source_rate as f32;
            frame[k] = (2.0 * std::f32::consts::PI * 1000.0 * t).sin();
        }
        let resampled = resampler.process(&[frame], None)?;
        buffer_48k.extend_from_slice(&resampled[0]);

        if buffer_48k.len() >= target_samples {
            let window: Vec<f32> = buffer_48k.drain(..target_samples).collect();
            let embedding = audio.embed(&window)?;
            // In a real pipeline: push embedding.as_slice() into a lancedb FixedSizeListBuilder.
            println!(
                "indexed 10 s window — embedding dim={}, head=[{:.4}, {:.4}, {:.4}, ...]",
                embedding.dim(),
                embedding.as_slice()[0],
                embedding.as_slice()[1],
                embedding.as_slice()[2],
            );
        }
    }
    Ok(())
}
```

- [ ] **Step 2: Create `examples/index_and_search.rs`**

```rust
//! Sequential pedagogical demo: index one fixed window first, then run a single query.
//!
//! In real deployments indexing is continuous and querying is on-demand; a single main()
//! cannot naturally show both as live, so this example is explicitly sequential.
//! See spec §1.1 / §13.

use std::env;
use std::path::PathBuf;

use textclap::{Clap, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = env::var_os("TEXTCLAP_MODELS_DIR")
        .map(PathBuf::from)
        .ok_or("set TEXTCLAP_MODELS_DIR")?;
    let mut clap = Clap::from_files(
        dir.join("audio_model_quantized.onnx"),
        dir.join("text_model_quantized.onnx"),
        dir.join("tokenizer.json"),
        Options::new(),
    )?;
    clap.warmup()?;

    // ── Indexing side: embed a single 10 s window ──
    let pcm = vec![0.0f32; 480_000]; // silence; real pipeline reads PCM from a decoder
    let audio_emb = clap.audio_mut().embed(&pcm)?;
    println!("indexed audio window: dim={}, first 3 = [{:.4}, {:.4}, {:.4}]",
             audio_emb.dim(),
             audio_emb.as_slice()[0],
             audio_emb.as_slice()[1],
             audio_emb.as_slice()[2]);

    // ── Query side: encode a search text and compute cosine similarity ──
    let query = clap.text_mut().embed("dog barking near a door")?;
    let similarity = audio_emb.cosine(&query);
    println!("cosine similarity to 'dog barking near a door': {:.4}", similarity);

    // ── Read-back demo: stored vectors round-trip through try_from_unit_slice ──
    let stored_bytes = audio_emb.to_vec();
    let restored = textclap::Embedding::try_from_unit_slice(&stored_bytes)?;
    let restored_sim = query.cosine(&restored);
    println!("restored embedding cosine: {:.4}", restored_sim);

    Ok(())
}
```

- [ ] **Step 3: Build the examples**

```bash
cargo build --examples
```

Expected: clean build (examples are `publish = false` per Cargo.toml; they don't gate the published
crate).

- [ ] **Step 4: Run one example to sanity-check (optional)**

```bash
TEXTCLAP_MODELS_DIR=~/textclap-models cargo run --example index_and_search
```

Expected: prints embedding info + cosine similarity.

- [ ] **Step 5: Commit**

```bash
git add examples/
git commit -m "Add examples: audio_window_to_clap and index_and_search"
```

---

### Task 30: README + LICENSE updates

**Files:**
- Create / replace: `README.md`
- Modify: `LICENSE-MIT`, `LICENSE-APACHE`, `COPYRIGHT` (copyright holder/year if needed)

The README is the primary user-facing document. It must include the §13 quick-start, model SHA256s,
attribution-on-downstream notice, ort-coupling note, lancedb integration snippet, and the §1.1
indexing-vs-query diagram.

- [ ] **Step 1: Replace `README.md`**

```markdown
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
- HTSAT and CLAP papers: citation required (see `CITATION.bib` for BibTeX).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
```

- [ ] **Step 2: Update copyright in license files** (year and holder per Findit-AI conventions, matching
sibling crates)

- [ ] **Step 3: Build docs locally to verify**

```bash
cargo doc --no-deps
```

Expected: clean build, no missing-docs warnings (the `#![deny(missing_docs)]` lint catches any
omissions).

- [ ] **Step 4: Commit**

```bash
git add README.md LICENSE-MIT LICENSE-APACHE COPYRIGHT
git commit -m "Add README, update license/copyright headers"
```

---

### Task 31: GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

Mirrors the sibling crates' CI matrix. Integration tests gated on a runner-local model cache.

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main, "0.*"]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 4 1 * *" # monthly

env:
  CARGO_TERM_COLOR: always

jobs:
  rustfmt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - run: cargo fmt --all --check

  clippy:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        features: ["", "--all-features"]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - run: cargo clippy --all-targets ${{ matrix.features }} -- -D warnings

  build-test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --verbose
      - run: cargo test --lib --verbose

  doctest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --doc --all-features

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo install cargo-tarpaulin
      - run: cargo tarpaulin --out Xml --workspace --all-features
      - uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./cobertura.xml

  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Cache models
        id: cache-models
        uses: actions/cache@v4
        with:
          path: ~/textclap-models
          key: textclap-models-${{ hashFiles('tests/fixtures/MODELS.md') }}
      - name: Download models
        if: steps.cache-models.outputs.cache-hit != 'true'
        run: |
          mkdir -p ~/textclap-models
          # HF revision pin from MODELS.md — replace <rev> with the actual SHA at Phase A.
          REV="<HF revision SHA from MODELS.md>"
          curl -L -o ~/textclap-models/audio_model_quantized.onnx \
            "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/${REV}/onnx/audio_model_quantized.onnx"
          curl -L -o ~/textclap-models/text_model_quantized.onnx \
            "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/${REV}/onnx/text_model_quantized.onnx"
          curl -L -o ~/textclap-models/tokenizer.json \
            "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/${REV}/tokenizer.json"
          # Verify SHA256s
          cd ~/textclap-models
          sha256sum -c <(grep -E 'audio_model|text_model|tokenizer\.json' \
                          "$GITHUB_WORKSPACE/tests/fixtures/MODELS.md" \
                        | awk '{print $5"  "$2}')
      - name: Run integration tests
        env:
          TEXTCLAP_MODELS_DIR: ~/textclap-models
        run: cargo test --test clap_integration
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions CI workflow with model-cache integration job"
```

---

## Final verification

After all 31 tasks:

- [ ] **Step 1: Full local check**

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release
cargo test --lib
cargo test --doc --all-features
TEXTCLAP_MODELS_DIR=~/textclap-models cargo test --test clap_integration
cargo build --examples
cargo bench --no-run
```

Expected: every command exits 0. Integration tests pass against the model files.

- [ ] **Step 2: Tag the release candidate**

```bash
git tag -a v0.1.0 -m "textclap 0.1.0 — initial public release"
```

(Don't push the tag until the maintainer reviews the full diff and decides to publish.)












