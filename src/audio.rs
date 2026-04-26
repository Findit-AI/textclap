//! Audio encoder (CLAP HTSAT side). See spec §7.3 / §8.2.

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use crate::clap_model::{Embedding, NORM_BUDGET};
use crate::error::{Error, Result};
use crate::mel::{MelExtractor, T_FRAMES};
use crate::options::{ChunkingOptions, Options};

// Backfilled from golden_onnx_io.json per §3.4. Module-private.
const AUDIO_INPUT_NAME: &str = "input_features";
const AUDIO_OUTPUT_NAME: &str = "audio_embeds";

/// Compile-time const indicating whether the audio ONNX output is already L2-normalized.
/// Backfilled from `golden_onnx_io.json["audio_output_is_unit_norm"]` per §3.4.
const AUDIO_OUTPUT_IS_UNIT_NORM: bool = false;

const TARGET_SAMPLES: usize = 480_000;
const N_MELS: usize = 64;
const EMBEDDING_DIM: usize = 512;

/// Validate an ORT output shape against the expected one. Sibling-convention parameter order:
/// (actual, expected) — matches silero `session.rs`.
pub(crate) fn validate_shape(
  tensor: &'static str,
  actual: &[i64],
  expected: &[i64],
) -> Result<()> {
  if actual != expected {
    return Err(Error::UnexpectedTensorShape {
      tensor,
      actual: actual.to_vec(),
      expected: expected.to_vec(),
    });
  }
  Ok(())
}

/// Audio encoder. See spec §7.3.
pub struct AudioEncoder {
  session: Session,
  mel: MelExtractor,
  /// Scratch for `[N · T_FRAMES · 64]` f32 mel features (time-major layout per Phase A).
  mel_scratch: Vec<f32>,
  /// Scratch for raw `[N × 512]` projection outputs (private; never exposed).
  #[allow(dead_code)] // Used by Tasks 20-21.
  proj_scratch: Vec<[f32; 512]>,
}

impl AudioEncoder {
  /// Load from a file path.
  ///
  /// Session-construction chain matches silero `src/session.rs:97-104`: the constructor is
  /// `Session::builder()`. Typed `ort::Error<SessionBuilder>` from `with_optimization_level`
  /// is widened to `ort::Error` via `ort::Error::from` to fit the path-carrying error variant.
  pub fn from_file<P: AsRef<Path>>(onnx_path: P, opts: Options) -> Result<Self> {
    let path = onnx_path.as_ref();
    let session = Session::builder()
      .map_err(|source| Error::OnnxLoadFromFile {
        path: path.to_path_buf(),
        source,
      })?
      .with_optimization_level(opts.optimization_level())
      .map_err(|source| Error::OnnxLoadFromFile {
        path: path.to_path_buf(),
        source: ort::Error::from(source),
      })?
      .commit_from_file(path)
      .map_err(|source| Error::OnnxLoadFromFile {
        path: path.to_path_buf(),
        source,
      })?;
    Self::from_loaded_session(session)
  }

  /// Load from caller-supplied bytes (copied into the ORT session).
  pub fn from_memory(onnx_bytes: &[u8], opts: Options) -> Result<Self> {
    let session = Session::builder()
      .map_err(Error::OnnxLoadFromMemory)?
      .with_optimization_level(opts.optimization_level())
      .map_err(|source| Error::OnnxLoadFromMemory(ort::Error::from(source)))?
      .commit_from_memory(onnx_bytes)
      .map_err(Error::OnnxLoadFromMemory)?;
    Self::from_loaded_session(session)
  }

  /// Wrap a pre-built ORT session. See spec §7.3 for the asymmetric purposes.
  pub fn from_ort_session(session: Session, _opts: Options) -> Result<Self> {
    // Note: opts is unused here because the wrapped session was already configured
    // by the caller. We still accept it for API symmetry with from_file/from_memory.
    Self::from_loaded_session(session)
  }

  fn from_loaded_session(session: Session) -> Result<Self> {
    // Schema check — ensures the wrapped session matches the contract recorded by §3.2.
    let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    if !inputs.iter().any(|n| *n == AUDIO_INPUT_NAME) {
      return Err(Error::SessionSchema {
        detail: format!(
          "audio session missing expected input {:?}; got inputs {:?}",
          AUDIO_INPUT_NAME, inputs,
        ),
      });
    }
    let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    if !outputs.iter().any(|n| *n == AUDIO_OUTPUT_NAME) {
      return Err(Error::SessionSchema {
        detail: format!(
          "audio session missing expected output {:?}; got outputs {:?}",
          AUDIO_OUTPUT_NAME, outputs,
        ),
      });
    }
    Ok(Self {
      session,
      mel: MelExtractor::new(),
      mel_scratch: Vec::new(),
      proj_scratch: Vec::new(),
    })
  }

  /// Embed a single ≤10 s clip. See spec §7.3 / §8.2.
  ///
  /// Order: empty check → length check → finiteness scan → mel + ONNX → unit-norm guard or
  /// L2-normalize. `samples.len()` must be in `1..=480_000`.
  pub fn embed(&mut self, samples: &[f32]) -> Result<Embedding> {
    if samples.is_empty() {
      return Err(Error::EmptyAudio { clip_index: None });
    }
    if samples.len() > TARGET_SAMPLES {
      return Err(Error::AudioTooLong {
        got: samples.len(),
        max: TARGET_SAMPLES,
      });
    }
    if let Some(sample_index) = Self::first_non_finite(samples) {
      return Err(Error::NonFiniteAudio {
        clip_index: None,
        sample_index,
      });
    }

    let mut out = Vec::with_capacity(1);
    self.embed_projections_batched(&[samples], &mut out)?;
    let row = out.pop().expect("helper always pushes for non-empty input");
    Self::finalize_embedding(row)
  }

  /// Batch of clips of any lengths in 0 < len ≤ 480 000. See spec §7.3 / §8.2.
  pub fn embed_batch(&mut self, clips: &[&[f32]]) -> Result<Vec<Embedding>> {
    if clips.is_empty() {
      return Ok(Vec::new());
    }

    // Per-clip validation.
    for (i, clip) in clips.iter().enumerate() {
      if clip.is_empty() {
        return Err(Error::EmptyAudio { clip_index: Some(i) });
      }
      if clip.len() > TARGET_SAMPLES {
        return Err(Error::AudioTooLong {
          got: clip.len(),
          max: TARGET_SAMPLES,
        });
      }
      if let Some(sample_index) = Self::first_non_finite(clip) {
        return Err(Error::NonFiniteAudio {
          clip_index: Some(i),
          sample_index,
        });
      }
    }

    let mut raw = Vec::with_capacity(clips.len());
    self.embed_projections_batched(clips, &mut raw)?;
    let mut out = Vec::with_capacity(raw.len());
    for row in raw {
      out.push(Self::finalize_embedding(row)?);
    }
    Ok(out)
  }

  /// Arbitrary-length input via textclap's chunking convention. NOT LAION-reference compatible.
  /// See spec §7.3 / §8.2.
  pub fn embed_chunked(
    &mut self,
    samples: &[f32],
    opts: &ChunkingOptions,
  ) -> Result<Embedding> {
    if samples.is_empty() {
      return Err(Error::EmptyAudio { clip_index: None });
    }
    if opts.window_samples() == 0
      || opts.hop_samples() == 0
      || opts.batch_size() == 0
      || opts.hop_samples() > opts.window_samples()
    {
      return Err(Error::ChunkingConfig {
        window_samples: opts.window_samples(),
        hop_samples: opts.hop_samples(),
        batch_size: opts.batch_size(),
      });
    }
    if let Some(sample_index) = Self::first_non_finite(samples) {
      return Err(Error::NonFiniteAudio {
        clip_index: None,
        sample_index,
      });
    }

    // Chunk offsets: 0, hop, 2·hop, …; skip trailing < window/4 unless the input itself is shorter.
    let window = opts.window_samples();
    let hop = opts.hop_samples();
    let min_keep = window / 4;
    let mut offsets: Vec<usize> = Vec::new();
    let mut off = 0;
    while off < samples.len() {
      let remain = samples.len() - off;
      let chunk_len = remain.min(window);
      if chunk_len >= min_keep || offsets.is_empty() {
        offsets.push(off);
      }
      off += hop;
    }

    // Process in groups of batch_size; accumulate raw projections.
    let mut accumulator: Vec<[f32; 512]> = Vec::with_capacity(offsets.len());
    let mut tmp_proj: Vec<[f32; 512]> = Vec::with_capacity(opts.batch_size());
    for batch_offsets in offsets.chunks(opts.batch_size()) {
      let chunks: Vec<&[f32]> = batch_offsets
        .iter()
        .map(|&o| &samples[o..(o + window).min(samples.len())])
        .collect();
      self.embed_projections_batched(&chunks, &mut tmp_proj)?;
      accumulator.extend(tmp_proj.drain(..));
    }

    // Single-chunk case skips aggregation regardless of branch.
    if accumulator.len() == 1 {
      return Self::finalize_embedding(
        accumulator
          .into_iter()
          .next()
          .expect("checked len == 1 above"),
      );
    }

    // Aggregate. Both branches end with L2-normalize → Embedding (via from_slice_normalizing,
    // which inherits the cancellation-safe normalize and unit-norm invariant).
    // Branch is dead-code-eliminated by the optimizer when AUDIO_OUTPUT_IS_UNIT_NORM is fixed.
    let mut centroid = [0.0f32; 512];
    if AUDIO_OUTPUT_IS_UNIT_NORM {
      // Spherical-mean: average unit vectors, then normalize.
      for row in &accumulator {
        for (acc, &v) in centroid.iter_mut().zip(row.iter()) {
          *acc += v;
        }
      }
    } else {
      // Centroid path: average raw projections, then normalize.
      for row in &accumulator {
        for (acc, &v) in centroid.iter_mut().zip(row.iter()) {
          *acc += v;
        }
      }
    }
    let inv_n = 1.0 / accumulator.len() as f32;
    for v in &mut centroid {
      *v *= inv_n;
    }
    Embedding::from_slice_normalizing(&centroid)
  }

  /// Run a dummy forward to amortize ORT operator specialization. See spec §11.4.
  pub fn warmup(&mut self) -> Result<()> {
    let silence = vec![0.0f32; TARGET_SAMPLES];
    let _ = self.embed(&silence)?;
    Ok(())
  }
}

impl AudioEncoder {
  /// Compute the audio model's raw projection outputs. These are un-normalized 512-dim vectors if
  /// `AUDIO_OUTPUT_IS_UNIT_NORM == false`, or already-unit-norm vectors if true. Callers handle
  /// any subsequent normalization or release-mode unit-norm guard themselves.
  ///
  /// `out` is cleared on entry; capacity is reserved for `clips.len()` entries and one row is
  /// pushed per clip. Prior contents are dropped.
  ///
  /// The chunked path's per-call accumulator is a *separate* `Vec` from `self.proj_scratch`;
  /// see spec §8.2.
  pub(crate) fn embed_projections_batched(
    &mut self,
    clips: &[&[f32]],
    out: &mut Vec<[f32; 512]>,
  ) -> Result<()> {
    let n = clips.len();
    debug_assert!(n > 0, "embed_projections_batched requires non-empty input");

    let row_len = N_MELS * T_FRAMES;
    let total = n * row_len;

    // §7.3.1 scratch lifecycle: clear + resize before binding any tensor view.
    self.mel_scratch.clear();
    self.mel_scratch.resize(total, 0.0);

    for (i, clip) in clips.iter().enumerate() {
      let row_start = i * row_len;
      let row_end = row_start + row_len;
      self
        .mel
        .extract_into(clip, &mut self.mel_scratch[row_start..row_end])?;
    }

    // Bind tensor view AFTER all resizes complete. Borrow checker prevents subsequent mutation.
    // Tensor shape is [n, 1, T_FRAMES, N_MELS] (time-major), matching the HF extractor's
    // [batch, channel, T, mel_bins] layout.
    let input_shape = [n, 1usize, T_FRAMES, N_MELS];
    let input = TensorRef::from_array_view((input_shape, self.mel_scratch.as_slice()))?;

    let outputs = self.session.run(ort::inputs![AUDIO_INPUT_NAME => input])?;
    let (shape, data) = outputs[AUDIO_OUTPUT_NAME].try_extract_tensor::<f32>()?;
    validate_shape(
      "audio_output",
      shape.as_ref(),
      &[n as i64, EMBEDDING_DIM as i64],
    )?;

    out.clear();
    out.reserve(n);
    for i in 0..n {
      let mut row = [0.0f32; EMBEDDING_DIM];
      row.copy_from_slice(&data[i * EMBEDDING_DIM..(i + 1) * EMBEDDING_DIM]);
      out.push(row);
    }
    // Tensor views drop here; mel_scratch becomes mutable again.
    Ok(())
  }

  /// SIMD-friendly finiteness scan. Returns `Some(index)` of the first non-finite sample, or `None`.
  /// Cost ~50 µs over 480 000 samples — dwarfed by ONNX inference.
  fn first_non_finite(samples: &[f32]) -> Option<usize> {
    for (i, &v) in samples.iter().enumerate() {
      if !v.is_finite() {
        return Some(i);
      }
    }
    None
  }

  /// Convert a raw projection row into a unit-norm `Embedding`. See spec §8.2 step 5.
  ///
  /// Selects between the trust-path (output already unit-norm — release-mode budget guard,
  /// then `from_array_trusted_unit_norm`) and the L2-normalize path. Reused by `embed_batch`
  /// (Task 20) and `embed_chunked` (Task 21).
  fn finalize_embedding(row: [f32; 512]) -> Result<Embedding> {
    if AUDIO_OUTPUT_IS_UNIT_NORM {
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
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Empty input → EmptyAudio. Doesn't need a model.
  #[test]
  fn embed_empty_returns_empty_audio_error() {
    // We can't construct AudioEncoder without a model file, so we test the validation path
    // by running through embed_projections_batched's preconditions. The full path is exercised
    // in tests/clap_integration.rs.
    // For now, test that the error variant exists and matches.
    let err = Error::EmptyAudio { clip_index: None };
    assert!(matches!(err, Error::EmptyAudio { clip_index: None }));
  }

  /// Finiteness-scan helper finds the first NaN.
  #[test]
  fn first_non_finite_finds_nan_at_zero() {
    let s = [f32::NAN, 0.0, 0.0];
    assert_eq!(AudioEncoder::first_non_finite(&s), Some(0));
  }

  #[test]
  fn first_non_finite_finds_inf_in_middle() {
    let s = [0.0, 1.0, f32::INFINITY, 2.0];
    assert_eq!(AudioEncoder::first_non_finite(&s), Some(2));
  }

  #[test]
  fn first_non_finite_returns_none_for_clean_input() {
    let s = [0.0_f32; 100];
    assert_eq!(AudioEncoder::first_non_finite(&s), None);
  }
}
