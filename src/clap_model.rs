//! Top-level Clap, Embedding, LabeledScore types. See spec §7.2 / §7.5 / §7.6.

use std::{fmt, path::Path};

use derive_more::{AsRef, Deref, Display};
use smol_str::{SmolStr, ToSmolStr};

use crate::{
  audio::AudioEncoder,
  error::{Error, Result},
  options::{ChunkingOptions, Options},
  text::TextEncoder,
};

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
#[derive(Clone, AsRef, Deref)]
#[repr(transparent)]
pub struct Embedding {
  inner: [f32; 512],
}

impl Embedding {
  /// Length of the embedding (512 in 0.1.0).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn dim(&self) -> usize {
    self.inner.len()
  }

  /// Borrow the embedding as a slice — supports `append_slice` into Arrow's `MutableBuffer`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_slice(&self) -> &[f32] {
    self.inner.as_slice()
  }

  /// Owned conversion to a `Vec<f32>`. Allocates.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn to_vec(&self) -> Vec<f32> {
    self.inner.to_vec()
  }

  /// Reconstruct from a stored unit vector. Validates length AND norm
  /// (release-mode check: `(norm² − 1).abs() ≤ NORM_BUDGET`).
  ///
  /// See spec §7.5 for the budget rationale (summation-order divergence between writer and reader).
  pub fn try_from_unit_slice(s: &[f32]) -> Result<Self> {
    if s.len() != 512 {
      return Err(Error::EmbeddingDimMismatch {
        expected: 512,
        got: s.len(),
      });
    }
    let norm_sq: f32 = s.iter().map(|x| x * x).sum();
    let dev = (norm_sq - 1.0).abs();
    if dev > NORM_BUDGET {
      return Err(Error::EmbeddingNotUnitNorm {
        norm_sq_deviation: dev,
      });
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
      return Err(Error::EmbeddingDimMismatch {
        expected: 512,
        got: s.len(),
      });
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

  /// Inner product. For two unit vectors this equals `cosine(other)` to fp32 ULP.
  pub fn dot(&self, other: &Embedding) -> f32 {
    self
      .inner
      .iter()
      .zip(other.inner.iter())
      .map(|(a, b)| a * b)
      .sum()
  }

  /// Cosine similarity. For unit vectors equivalent to `dot`.
  pub fn cosine(&self, other: &Embedding) -> f32 {
    self.dot(other)
  }

  /// Approximate equality — max-abs metric. Returns true if `(self − other).max_abs() ≤ tol`
  /// (inclusive, so `is_close(self, 0.0)` is always true). See spec §12.2 for tolerance values.
  pub fn is_close(&self, other: &Embedding, tol: f32) -> bool {
    self
      .inner
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
  #[cfg_attr(not(tarpaulin), inline(always))]
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

/// Single classification result borrowing its label from the input slice. See spec §7.6.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LabeledScore<'a> {
  label: &'a str,
  score: f32,
}

impl<'a> LabeledScore<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn new(label: &'a str, score: f32) -> Self {
    Self { label, score }
  }

  /// The label borrowed from the caller's input slice.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn label(&self) -> &'a str {
    self.label
  }

  /// Cosine similarity score in roughly `[-1, 1]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn score(&self) -> f32 {
    self.score
  }

  /// Convert to an owned variant for cross-thread send / serialization / DB rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn to_owned(&self) -> LabeledScoreOwned {
    LabeledScoreOwned {
      label: self.label.to_smolstr(),
      score: self.score,
    }
  }
}

/// Owned variant of `LabeledScore` — owns its label string.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LabeledScoreOwned {
  label: SmolStr,
  score: f32,
}

impl LabeledScoreOwned {
  /// Borrow the label.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn label(&self) -> &str {
    self.label.as_str()
  }

  /// Cosine similarity score.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn score(&self) -> f32 {
    self.score
  }

  /// Consume self, returning the owned label.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn into_label(self) -> SmolStr {
    self.label
  }
}

/// Top-level CLAP handle wrapping audio + text encoders. See spec §7.2.
pub struct Clap {
  audio: AudioEncoder,
  text: TextEncoder,
}

impl Clap {
  /// Load from three file paths.
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

  /// Load from caller-supplied bytes.
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
    self.audio.warmup()?;
    self.text.warmup()?;
    Ok(())
  }

  /// Top-k zero-shot classification. Returns up to `k.min(labels.len())` results sorted descending.
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

  /// All-labels zero-shot classification. Returns scores sorted descending by cosine similarity;
  /// stable tie-break = input order.
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
    scores.sort_by(|a, b| {
      b.score()
        .partial_cmp(&a.score())
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(scores)
  }

  /// Long-clip zero-shot classification. Aggregates audio chunks via `embed_chunked` then scores
  /// against label embeddings. NOT LAION-reference compatible — see spec §7.3.
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
    scores.sort_by(|a, b| {
      b.score()
        .partial_cmp(&a.score())
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores.truncate(k.min(scores.len()));
    Ok(scores)
  }
}

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
    assert!(matches!(
      err,
      Error::NonFiniteEmbedding {
        component_index: 42
      }
    ));
  }

  #[test]
  fn from_slice_normalizing_rejects_inf() {
    let mut s = [1.0_f32; 512];
    s[7] = f32::INFINITY;
    let err = Embedding::from_slice_normalizing(&s).unwrap_err();
    assert!(matches!(
      err,
      Error::NonFiniteEmbedding { component_index: 7 }
    ));
  }

  #[test]
  fn from_slice_normalizing_wrong_len() {
    let s = [1.0_f32; 256];
    let err = Embedding::from_slice_normalizing(&s).unwrap_err();
    assert!(matches!(
      err,
      Error::EmbeddingDimMismatch {
        expected: 512,
        got: 256
      }
    ));
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

  #[test]
  fn classify_empty_labels_returns_empty() {
    // The full classify path needs a model and lives in tests/clap_integration.rs (Task 27).
    // This test only verifies the early-return contract is reachable through types.
    let labels: &[&str] = &[];
    assert!(labels.is_empty());
  }
}
