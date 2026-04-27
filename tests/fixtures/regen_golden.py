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

    onnx_io_path = fixtures / "golden_onnx_io.json"
    if not onnx_io_path.exists():
        raise SystemExit(
            f"{onnx_io_path} not found — run `inspect_onnx.py` first (Task 5 step 2)."
        )
    onnx_io = json.load(open(onnx_io_path))

    _step_extractor_and_mel(fixtures, onnx_io)
    _step_warmup_text(fixtures, models)
    _step_audio_golden(fixtures, models, onnx_io)
    _step_text_goldens(fixtures, models, onnx_io)
    _step_filterbank_rows(fixtures)

    print(f"All goldens written to {fixtures}/")


def _step_extractor_and_mel(fixtures: Path, onnx_io: dict) -> None:
    """ §3.1 step 2-3: extract parameters + golden mel features."""
    extractor = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

    audio, _ = librosa.load(str(fixtures / "sample.wav"), sr=48000, mono=True)
    features = extractor(audio, sampling_rate=48000, return_tensors="pt")
    mel = features["input_features"].numpy()
    # The HF extractor produces [batch, channel, T, mel_bins] (time-major), NOT [batch, channel,
    # mel_bins, T] as the original spec §8.1 assumed. We save in the extractor's native layout —
    # the ONNX audio model consumes the same layout via TensorRef. Task 6 (spec backfill) corrects
    # the §8.1 / §8.2 spec wording.
    if mel.ndim == 4:
        mel_2d = mel[0, 0]            # [T, 64]
    elif mel.ndim == 3:
        mel_2d = mel[0]                # [T, 64]
    else:
        raise ValueError(f"unexpected mel shape from extractor: {mel.shape}")
    assert mel_2d.shape[1] == 64, (
        f"expected mel_bins=64 on axis 1; got shape {mel_2d.shape}. The extractor's layout may "
        f"have changed in this transformers version."
    )
    np.save(fixtures / "golden_mel.npy", mel_2d.astype(np.float32))

    T = int(mel_2d.shape[0])
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


def _step_text_goldens(fixtures: Path, models: Path, onnx_io: dict) -> None:
    tok = Tokenizer.from_file(str(models / "tokenizer.json"))
    text_session = ort.InferenceSession(str(models / "text_model_quantized.onnx"))

    embs = []
    for label in LABELS:
        enc = tok.encode(label)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feeds = {onnx_io["text_input_ids_name"]: ids}
        if onnx_io.get("text_attention_mask_name"):
            feeds[onnx_io["text_attention_mask_name"]] = mask
        if onnx_io.get("text_position_ids_name"):
            pad_id = onnx_io.get("text_pad_id", 1)
            # RoBERTa create_position_ids_from_input_ids reference (HuggingFace transformers):
            #   incremental_indices = cumsum(mask) * mask
            #   return incremental_indices + padding_idx
            # The * mask must come BEFORE + padding_idx so padded positions get padding_idx,
            # NOT zero. (cumsum + pad_id) * mask is wrong — it zeroes pad positions.
            pos = np.cumsum(mask, axis=1, dtype=np.int64) * mask + pad_id
            feeds[onnx_io["text_position_ids_name"]] = pos.astype(np.int64)

        raw = text_session.run([onnx_io["text_output_name"]], feeds)[0]
        raw = raw.astype(np.float32).reshape(-1)
        assert raw.shape == (512,), f"unexpected text projection shape for {label!r}: {raw.shape}"
        norm = np.linalg.norm(raw).astype(np.float32)
        embs.append((raw / norm).astype(np.float32))

    np.save(fixtures / "golden_text_embs.npy", np.stack(embs))


def _step_filterbank_rows(fixtures: Path) -> None:
    """ Pre-computed librosa references for the §8.1.1 mel-filter row test."""
    fb = librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64,
                             fmin=50, fmax=14000, htk=False, norm="slaney")
    np.save(fixtures / "filterbank_row_0.npy",  fb[0].astype(np.float32))
    np.save(fixtures / "filterbank_row_10.npy", fb[10].astype(np.float32))  # near 1 kHz Slaney inflection
    np.save(fixtures / "filterbank_row_32.npy", fb[32].astype(np.float32))


if __name__ == "__main__":
    main()
