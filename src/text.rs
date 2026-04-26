//! Text encoder (CLAP RoBERTa side). See spec §7.4 / §9.

use std::path::Path;

use crate::{clap_model::Embedding, error::Result, options::Options};

// Backfilled from golden_onnx_io.json per §3.4. Module-private.
const TEXT_INPUT_IDS_NAME: &str = "input_ids"; // PLACEHOLDER
const TEXT_ATTENTION_MASK_NAME: &str = "attention_mask"; // PLACEHOLDER
// `TEXT_POSITION_IDS_NAME` exists only if §3.2 found position_ids as an external input.
// const TEXT_POSITION_IDS_NAME: &str = "position_ids";
const TEXT_OUTPUT_NAME: &str = "text_embeds"; // PLACEHOLDER

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
