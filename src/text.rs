//! Text encoder (CLAP RoBERTa side). See spec §7.4 / §9.

use std::path::Path;

use ort::{session::Session, value::TensorRef};
use tokenizers::{
  PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
  TruncationParams, TruncationStrategy,
};

use crate::{
  audio::validate_shape,
  clap::{Embedding, NORM_BUDGET},
  error::{Error, Result},
  options::Options,
};

// Backfilled from golden_onnx_io.json per §3.4. Module-private.
// The Xenova clap-htsat-unfused export inlines attention_mask and position_ids derivation
// into the graph; only input_ids is externalized. See spec §7.4 / §9.2.
const TEXT_INPUT_IDS_NAME: &str = "input_ids";
const TEXT_OUTPUT_NAME: &str = "text_embeds";

const TEXT_OUTPUT_IS_UNIT_NORM: bool = false;

/// Maximum input tokens accepted by the text encoder. Backfilled from the pinned ONNX
/// graph's position-embedding table size (514 = 512 actual tokens + 2 RoBERTa special
/// offsets). Tokens beyond this length would index out of the position table and crash
/// the ORT Gather op. All three constructors install tokenizer-side truncation at this
/// length, matching standard RoBERTa convention.
const TEXT_MAX_TOKENS: usize = 512;

const EMBEDDING_DIM: usize = 512;

/// Text encoder. See spec §7.4.
pub struct TextEncoder {
  session: Session,
  tokenizer: Tokenizer,
  /// Cached at construction. Reserved for future exports that externalize `attention_mask` /
  /// `position_ids`; the current Xenova export inlines both, so `embed_batch` dispatches per-text
  /// (no batched ORT call) and `pad_id` is not load-bearing in the hot path. See `embed_batch`.
  #[allow(dead_code)]
  pad_id: i64,
  /// Reused scratch for input_ids tensor binding.
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

/// Force `BatchLongest` padding on the tokenizer, regardless of what the JSON declared.
/// Used by `from_files` / `from_memory` as a tokenizer-side hint, even though `embed_batch`
/// processes texts per-label rather than batching them through ORT (see `embed_batch` doc).
/// Kept for forward-compat with future exports that consume an externalized attention_mask.
/// See spec §7.4.
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

/// Force `LongestFirst` truncation at `TEXT_MAX_TOKENS` on the tokenizer. Used by all
/// three constructors. The model's position-embedding table is fixed at 514 entries, so
/// truncation is a hard model constraint rather than a tunable preference. See spec §7.4.
fn force_max_length_truncation(tokenizer: &mut Tokenizer) -> Result<()> {
  tokenizer
    .with_truncation(Some(TruncationParams {
      max_length: TEXT_MAX_TOKENS,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .map_err(Error::TokenizerLoadFromMemory)?;
  Ok(())
}

/// Reject `Padding::Fixed` for `from_ort_session` callers. See spec §7.4.
fn reject_fixed_padding(tokenizer: &Tokenizer) -> Result<()> {
  if let Some(p) = tokenizer.get_padding()
    && matches!(p.strategy, PaddingStrategy::Fixed(_))
  {
    return Err(Error::PaddingFixedRejected);
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
    force_max_length_truncation(&mut tokenizer)?;
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
    force_max_length_truncation(&mut tokenizer)?;
    Self::from_pieces(session, tokenizer, pad_id)
  }

  /// Wrap a pre-built ORT session and tokenizer. The tokenizer's padding configuration is
  /// preserved (`BatchLongest`/`Longest`/none); `Padding::Fixed` is rejected to prevent silent
  /// max_length truncation. Truncation is forced to `TEXT_MAX_TOKENS = 512` because the ONNX
  /// position-embedding table cannot be exceeded — overlong input would crash the Gather op
  /// rather than produce a meaningful embedding. See spec §7.4.
  pub fn from_ort_session(
    session: Session,
    mut tokenizer: Tokenizer,
    _opts: Options,
  ) -> Result<Self> {
    reject_fixed_padding(&tokenizer)?;
    force_max_length_truncation(&mut tokenizer)?;
    let pad_id = resolve_pad_id(&tokenizer)?;
    Self::from_pieces(session, tokenizer, pad_id)
  }

  fn from_pieces(session: Session, tokenizer: Tokenizer, pad_id: i64) -> Result<Self> {
    let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    if !inputs.contains(&TEXT_INPUT_IDS_NAME) {
      return Err(Error::SessionSchema {
        detail: format!(
          "text session missing expected input {:?}; got inputs {:?}",
          TEXT_INPUT_IDS_NAME, inputs,
        ),
      });
    }
    let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    if !outputs.contains(&TEXT_OUTPUT_NAME) {
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

  /// Embed a single text query. See spec §7.4 / §9.2.
  pub fn embed(&mut self, text: &str) -> Result<Embedding> {
    if text.is_empty() {
      return Err(Error::EmptyInput { batch_index: None });
    }
    let encoding = self.tokenizer.encode(text, true).map_err(Error::Tokenize)?;
    let t = encoding.get_ids().len();

    // Resize scratch + cast u32 → i64.
    self.ids_scratch.clear();
    self.ids_scratch.reserve(t);
    for &id in encoding.get_ids() {
      self.ids_scratch.push(id as i64);
    }

    // Bind tensor view — Phase A: only input_ids is externalized; attention_mask is inlined.
    let ids_view = TensorRef::from_array_view(([1usize, t], self.ids_scratch.as_slice()))?;

    let outputs = self.session.run(ort::inputs![
      TEXT_INPUT_IDS_NAME => ids_view,
    ])?;

    let (shape, data) = outputs[TEXT_OUTPUT_NAME].try_extract_tensor::<f32>()?;
    validate_shape("text_output", shape.as_ref(), &[1, EMBEDDING_DIM as i64])?;

    let mut row = [0.0f32; EMBEDDING_DIM];
    row.copy_from_slice(&data[..EMBEDDING_DIM]);
    Self::finalize_embedding(row)
  }

  /// Convert a raw projection row into a unit-norm `Embedding`. Branch identical to audio side
  /// (see `src/audio.rs::finalize_embedding`).
  fn finalize_embedding(row: [f32; 512]) -> Result<Embedding> {
    if TEXT_OUTPUT_IS_UNIT_NORM {
      let norm_sq: f32 = row.iter().map(|x| x * x).sum();
      let dev = (norm_sq - 1.0).abs();
      if dev > NORM_BUDGET {
        return Err(Error::EmbeddingNotUnitNorm {
          norm_sq_deviation: dev,
        });
      }
      Ok(Embedding::from_array_trusted_unit_norm(row))
    } else {
      Embedding::from_slice_normalizing(&row)
    }
  }

  /// Embed a batch of text queries. Each text is encoded **independently** through `embed`,
  /// producing an embedding equivalent to calling `embed(text)` per text.
  ///
  /// Why per-text rather than a single ORT call: the Xenova export inlines attention_mask /
  /// position_ids derivation, and empirically does not perfectly mask pad tokens. Batched ORT
  /// runs produce embeddings that depend on batch composition (a label's embedding shifts when
  /// batched with a longer label). Per-text dispatch trades ~10× ORT calls for batch-invariant
  /// semantics, which `classify_*` requires for correctness. See spec §7.4 / §9.2.
  pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }
    // Per-text validation is performed by `embed`. We loop here so the empty-text error
    // index reflects the batch position, not the single-clip None.
    let mut out = Vec::with_capacity(texts.len());
    for (i, text) in texts.iter().enumerate() {
      if text.is_empty() {
        return Err(Error::EmptyInput {
          batch_index: Some(i),
        });
      }
      out.push(self.embed(text)?);
    }
    Ok(out)
  }

  /// Run a dummy forward to amortize ORT operator specialization. See spec §11.4.
  pub fn warmup(&mut self) -> Result<()> {
    // 9 repetitions of "the quick brown fox jumps over the lazy dog " (trailing space).
    // Backfilled from tests/fixtures/golden_params.json (warmup_text_repetitions = 9,
    // warmup_text_token_count = 84). See spec §3.4 / §11.4.
    const WARMUP_TEXT: &str = "the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog ";
    let _ = self.embed(WARMUP_TEXT)?;
    Ok(())
  }
}
