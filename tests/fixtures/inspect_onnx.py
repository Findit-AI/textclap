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
