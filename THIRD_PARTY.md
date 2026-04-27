# Third-Party Notices

textclap depends on third-party models, audio fixtures, and Rust crates whose
licenses, attribution requirements, and citation requests are listed here.
This file is informational; it does not change or supersede the licensing
terms of any of those works.

## Models

### LAION CLAP weights (HTSAT audio + RoBERTa text)

- **Source:** [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused)
- **License:** CC-BY 4.0
- **Attribution:** Required when redistributing weights. textclap does **not**
  bundle the ONNX model files; users download them at runtime. Downstream
  applications that redistribute the ONNX files inherit the CC-BY 4.0
  attribution obligation.

### Xenova ONNX export (quantized int8)

- **Source:** [`Xenova/clap-htsat-unfused`](https://huggingface.co/Xenova/clap-htsat-unfused)
- **License:** Apache-2.0 (export tooling and metadata)
- **Pinned revision:** see `models/MODELS.md`
- **Files used:**
  - `audio_model_quantized.onnx` (33 MB, not bundled — download at runtime)
  - `text_model_quantized.onnx` (121 MB, not bundled — download at runtime)
  - `tokenizer.json` (2 MB, **bundled** with this crate at `models/tokenizer.json`)
- The bundled `tokenizer.json` retains the upstream Xenova export's terms.

### Reference papers (citation requested)

- **HTSAT:** Chen, K., Du, X., Zhu, B., Ma, Z., Berg-Kirkpatrick, T.,
  Dubnov, S. *"HTS-AT: A Hierarchical Token-Semantic Audio Transformer for
  Sound Classification and Detection."* ICASSP 2022.
- **CLAP:** Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., Dubnov,
  S. *"Large-scale Contrastive Language-Audio Pretraining with Feature Fusion
  and Keyword-to-Caption Augmentation."* ICASSP 2023.
- **Earlier CLAP:** Elizalde, B., Deshmukh, S., Al Ismail, M., Wang, H. *"CLAP:
  Learning Audio Concepts from Natural Language Supervision."* ICASSP 2023.

## Test fixtures

### `tests/fixtures/sample.wav`

- **Source:** [`Xenova/transformers.js-docs/dog_barking.wav`](https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/dog_barking.wav)
- **Provenance:** originally 44.1 kHz mono; resampled to 48 kHz mono PCM via
  ffmpeg for use as a 5 s integration-test fixture.
- **License:** the transformers.js-docs dataset is published under Apache-2.0
  alongside the transformers.js documentation. The audio itself is short and
  likely from a public-domain or CC-licensed source. For stricter provenance
  (e.g. broader redistribution), replace with a known CC0 dog-bark recording
  and update `tests/fixtures/README.md`.

### Golden tensors (`tests/fixtures/golden_*.npy`, `golden_params.json`)

- Derived deterministically from the LAION/Xenova model weights and the
  `sample.wav` fixture by `tests/fixtures/regen_golden.py`. They inherit the
  upstream model and audio licenses noted above.

## Runtime dependencies

| Crate | Version | License |
|---|---|---|
| [ort](https://crates.io/crates/ort) | 2.0.0-rc.12 | Apache-2.0 OR MIT |
| [rustfft](https://crates.io/crates/rustfft) | 6 | Apache-2.0 OR MIT |
| [tokenizers](https://crates.io/crates/tokenizers) | 0.22 | Apache-2.0 |
| [thiserror](https://crates.io/crates/thiserror) | 2 | Apache-2.0 OR MIT |
| [derive_more](https://crates.io/crates/derive_more) | 2 | MIT |
| [smol_str](https://crates.io/crates/smol_str) | 0.3 | Apache-2.0 OR MIT |
| [serde](https://crates.io/crates/serde) (optional) | 1 | Apache-2.0 OR MIT |
| [serde_with](https://crates.io/crates/serde_with) (optional) | 3 | Apache-2.0 OR MIT |

Transitive dependencies inherit licenses through the dependency graph; run
`cargo tree --format '{p} {l}'` for an up-to-date listing.

`ort` links against ONNX Runtime, which is itself MIT-licensed. ONNX Runtime
incorporates several upstream third-party components — see the [ONNX Runtime
THIRD_PARTY_NOTICES](https://github.com/microsoft/onnxruntime/blob/main/ThirdPartyNotices.txt)
for the full chain.

`rustfft` includes vendored DSP routines under permissive terms. See
[`rustfft`'s repository](https://github.com/ejmahler/RustFFT) for details.

## Dev dependencies

Used only for tests, benches, and examples; not part of the published library.

| Crate | License |
|---|---|
| criterion | Apache-2.0 OR MIT |
| rubato | MIT |
| npyz | Apache-2.0 OR MIT |
| hound | Apache-2.0 OR MIT |

## Sibling-crate coupling

textclap is part of a trio with `silero` (VAD) and `soundevents` (sound
classification). All three pin `ort = "2.0.0-rc.12"` to keep ONNX Runtime
binaries shared across crates in a deployment. Bumping `ort` requires
coordinated updates across the trio.
