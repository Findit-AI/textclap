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
  #[error(
    "tokenizer has no pad token (configure padding in tokenizer.json or include a <pad> token)"
  )]
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
  #[error(
    "invalid chunking options: window={window_samples}, hop={hop_samples}, batch={batch_size}; \
             all must be > 0, hop ≤ window, and window ≤ 480000 samples (10 s @ 48 kHz)"
  )]
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
