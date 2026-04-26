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
import torch
import torch.nn.functional as F
import librosa
from transformers import ClapFeatureExtractor, ClapModel


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


def _inspect_text(proto: onnx.ModelProto) -> dict:
    g = proto.graph
    inputs_by_name = {i.name: i for i in g.input}

    # input_ids is always present. attention_mask and position_ids may or may not be externalized
    # depending on the export (the Xenova clap-htsat-unfused export inlines BOTH derivations into
    # the graph and exposes only input_ids).
    text_input_ids_name      = next(n for n in inputs_by_name if "input_ids" in n.lower())
    text_attention_mask_name = next((n for n in inputs_by_name if "attention" in n.lower() and "mask" in n.lower()), None)
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
    # transformers 5.x returns BaseModelOutputWithPooling from get_audio_features; the projected
    # embedding lives at .pooler_output (shape [B, 512]). Older 4.x versions returned the tensor
    # directly — handle both.
    with torch.no_grad():
        pt_out = pt_model.get_audio_features(**features)
        pt_emb_tensor = pt_out.pooler_output if hasattr(pt_out, "pooler_output") else pt_out
        pt_emb = F.normalize(pt_emb_tensor, dim=-1)  # robust to any batch size
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

if __name__ == "__main__":
    main()
