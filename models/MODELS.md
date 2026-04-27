# textclap model artifacts

textclap pins three model artifacts; this document records the verified versions. The two ONNX
files are loaded at runtime from caller-supplied paths; `tokenizer.json` is **bundled with the
crate** under `models/tokenizer.json` and exposed as `textclap::BUNDLED_TOKENIZER`.

**Source:** [Xenova/clap-htsat-unfused](https://huggingface.co/Xenova/clap-htsat-unfused)

Revision: c28f2883575e590e04d3146ff0713c2448d691ba

| File                          | Size   | SHA256 (from `shasum -a 256`) |
|-------------------------------|--------|--------------------------------|
| `audio_model_quantized.onnx`  | 33 MB  | (see MODELS.sha256)            |
| `text_model_quantized.onnx`   | 121 MB | (see MODELS.sha256)            |
| `tokenizer.json`              | 2.0 MB | (see MODELS.sha256)            |

The native `sha256sum -c` format lives in the sidecar `models/MODELS.sha256` so CI can verify
without parsing markdown:

```text
3fcff2c8824e7bcb83a983f2a49edab3b60cbcf4872ac70efee517355173bd1f  audio_model_quantized.onnx
1a3df8b197e249816e08415fd040434c44762b2eea7eb7bf8a48a0f0bf3c14e5  text_model_quantized.onnx
dc239041d98de27ffc3975473a1a23e3db4c937b23c138c38bbc66588bd247e5  tokenizer.json
```

Mismatched SHA256s produce undefined results — typically `Error::SessionSchema` or
`Error::UnexpectedTensorShape`, or in the worst case silent embedding drift. Verify before deployment.

**Original LAION model:** [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused)
(CC-BY 4.0 — attribution required when redistributing model files; see README §11.6).

## Download

```bash
mkdir -p ~/textclap-models
cd ~/textclap-models
REV=$(awk '/^Revision:/ {print $2; exit}' /path/to/textclap/models/MODELS.md)
curl -L -o audio_model_quantized.onnx \
  "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/${REV}/onnx/audio_model_quantized.onnx"
curl -L -o text_model_quantized.onnx \
  "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/${REV}/onnx/text_model_quantized.onnx"
curl -L -o tokenizer.json \
  "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/${REV}/tokenizer.json"
sha256sum -c /path/to/textclap/models/MODELS.sha256
```
