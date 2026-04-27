//! Text encoder (CLAP RoBERTa side). See spec §7.4 / §9.

use std::path::Path;

use ort::session::Session;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use crate::clap_model::Embedding;
use crate::error::{Error, Result};
use crate::options::Options;

// Backfilled from golden_onnx_io.json per §3.4. Module-private.
// The Xenova clap-htsat-unfused export inlines attention_mask and position_ids derivation
// into the graph; only input_ids is externalized. See spec §7.4 / §9.2.
const TEXT_INPUT_IDS_NAME: &str = "input_ids";
const TEXT_OUTPUT_NAME: &str = "text_embeds";

#[allow(dead_code)] // Used by Task 23 finalize_embedding.
const TEXT_OUTPUT_IS_UNIT_NORM: bool = false;

#[allow(dead_code)] // Used by Task 23.
const EMBEDDING_DIM: usize = 512;

/// Text encoder. See spec §7.4.
pub struct TextEncoder {
  #[allow(dead_code)] // Used by Tasks 23-24.
  session: Session,
  #[allow(dead_code)] // Used by Tasks 23-24.
  tokenizer: Tokenizer,
  /// Cached at construction. Reserved for future exports that externalize position_ids
  /// (the Xenova export inlines it; pad_id is the diagnostic source of truth here).
  #[allow(dead_code)] // Used if §3.2 externalizes position_ids; preserved for diagnostic/reflection.
  pad_id: i64,
  /// Reused scratch for input_ids tensor binding.
  #[allow(dead_code)] // Used by Tasks 23-24.
  ids_scratch: Vec<i64>,
}

/// Resolve the tokenizer's pad token id. Looks at `get_padding().pad_id` first, then falls back
/// to `token_to_id("<pad>")`. Returns `Error::NoPadToken` if neither is configured.
fn resolve_pad_id(tokenizer: &Tokenizer) -> Result<i64> {
  if let Some(p) = tokenizer.get_padding() {
    return Ok(p.pad_id as i64);
  }
  if let Some(id) = tokenizer.token_to_id("<pad>") {
    return Ok(id as i64);
  }
  Err(Error::NoPadToken)
}

/// Force `BatchLongest` padding on the tokenizer regardless of what the JSON declared.
/// Used by `from_files` / `from_memory`. See spec §7.4.
fn force_batch_longest_padding(tokenizer: &mut Tokenizer, pad_id: i64) {
  let pad_token = tokenizer
    .id_to_token(pad_id as u32)
    .unwrap_or_else(|| "<pad>".to_string());
  tokenizer.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::BatchLongest,
    pad_id: pad_id as u32,
    pad_token,
    pad_type_id: 0,
    direction: PaddingDirection::Right,
    pad_to_multiple_of: None,
  }));
}

/// Reject `Padding::Fixed` for `from_ort_session` callers. See spec §7.4. Used by Task 24.
#[allow(dead_code)] // Used by Task 24 from_ort_session.
fn reject_fixed_padding(tokenizer: &Tokenizer) -> Result<()> {
  if let Some(p) = tokenizer.get_padding() {
    if matches!(p.strategy, PaddingStrategy::Fixed(_)) {
      return Err(Error::PaddingFixedRejected);
    }
  }
  Ok(())
}

impl TextEncoder {
  /// Load from file paths (ONNX + tokenizer.json).
  pub fn from_files<P: AsRef<Path>>(
    onnx_path: P,
    tokenizer_json_path: P,
    opts: Options,
  ) -> Result<Self> {
    let onnx = onnx_path.as_ref();
    let tok_path = tokenizer_json_path.as_ref();
    let session = Session::builder()
      .map_err(|source| Error::OnnxLoadFromFile {
        path: onnx.to_path_buf(),
        source,
      })?
      .with_optimization_level(opts.optimization_level())
      .map_err(|source| Error::OnnxLoadFromFile {
        path: onnx.to_path_buf(),
        source: ort::Error::from(source),
      })?
      .commit_from_file(onnx)
      .map_err(|source| Error::OnnxLoadFromFile {
        path: onnx.to_path_buf(),
        source,
      })?;
    let mut tokenizer =
      Tokenizer::from_file(tok_path).map_err(|source| Error::TokenizerLoadFromFile {
        path: tok_path.to_path_buf(),
        source,
      })?;
    let pad_id = resolve_pad_id(&tokenizer)?;
    force_batch_longest_padding(&mut tokenizer, pad_id);
    Self::from_pieces(session, tokenizer, pad_id)
  }

  /// Load from caller-supplied bytes.
  pub fn from_memory(
    onnx_bytes: &[u8],
    tokenizer_json_bytes: &[u8],
    opts: Options,
  ) -> Result<Self> {
    let session = Session::builder()
      .map_err(Error::OnnxLoadFromMemory)?
      .with_optimization_level(opts.optimization_level())
      .map_err(|source| Error::OnnxLoadFromMemory(ort::Error::from(source)))?
      .commit_from_memory(onnx_bytes)
      .map_err(Error::OnnxLoadFromMemory)?;
    let mut tokenizer =
      Tokenizer::from_bytes(tokenizer_json_bytes).map_err(Error::TokenizerLoadFromMemory)?;
    let pad_id = resolve_pad_id(&tokenizer)?;
    force_batch_longest_padding(&mut tokenizer, pad_id);
    Self::from_pieces(session, tokenizer, pad_id)
  }

  /// Wrap a pre-built ORT session and tokenizer. See spec §7.4. Implemented in Task 24.
  pub fn from_ort_session(
    _session: Session,
    _tokenizer: Tokenizer,
    _opts: Options,
  ) -> Result<Self> {
    unimplemented!("TextEncoder::from_ort_session — implemented in Task 24")
  }

  fn from_pieces(session: Session, tokenizer: Tokenizer, pad_id: i64) -> Result<Self> {
    let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    if !inputs.iter().any(|n| *n == TEXT_INPUT_IDS_NAME) {
      return Err(Error::SessionSchema {
        detail: format!(
          "text session missing expected input {:?}; got inputs {:?}",
          TEXT_INPUT_IDS_NAME, inputs,
        ),
      });
    }
    let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    if !outputs.iter().any(|n| *n == TEXT_OUTPUT_NAME) {
      return Err(Error::SessionSchema {
        detail: format!(
          "text session missing expected output {:?}; got outputs {:?}",
          TEXT_OUTPUT_NAME, outputs,
        ),
      });
    }
    Ok(Self {
      session,
      tokenizer,
      pad_id,
      ids_scratch: Vec::new(),
    })
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
    unimplemented!("TextEncoder::warmup — implemented in Task 24")
  }
}
