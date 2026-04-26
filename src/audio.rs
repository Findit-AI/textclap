//! Audio encoder (CLAP HTSAT side). See spec §7.3 / §8.2.

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use crate::clap_model::Embedding;
use crate::error::{Error, Result};
use crate::mel::{MelExtractor, T_FRAMES};
use crate::options::Options;

// Backfilled from golden_onnx_io.json per §3.4. Module-private.
const AUDIO_INPUT_NAME: &str = "input_features";
const AUDIO_OUTPUT_NAME: &str = "audio_embeds";

/// Compile-time const indicating whether the audio ONNX output is already L2-normalized.
/// Backfilled from `golden_onnx_io.json["audio_output_is_unit_norm"]` per §3.4.
#[allow(dead_code)] // Used by §8.2 trust-path guard in Task 19.
const AUDIO_OUTPUT_IS_UNIT_NORM: bool = false;

#[allow(dead_code)] // Used by Tasks 18-21.
const TARGET_SAMPLES: usize = 480_000;
#[allow(dead_code)] // Used by Tasks 18-21.
const N_MELS: usize = 64;
#[allow(dead_code)] // Used by Tasks 18-21.
const EMBEDDING_DIM: usize = 512;

/// Validate an ORT output shape against the expected one. Sibling-convention parameter order:
/// (actual, expected) — matches silero `session.rs`.
#[allow(dead_code)] // Used by Tasks 19-21.
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
  #[allow(dead_code)] // Used by Tasks 19-21.
  session: Session,
  #[allow(dead_code)] // Used by Tasks 19-21.
  mel: MelExtractor,
  /// Scratch for `[N · T_FRAMES · 64]` f32 mel features (time-major layout per Phase A).
  #[allow(dead_code)] // Used by Tasks 19-21.
  mel_scratch: Vec<f32>,
  /// Scratch for raw `[N × 512]` projection outputs (private; never exposed).
  #[allow(dead_code)] // Used by Tasks 19-21.
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

  /// Embed a single ≤10 s clip.
  pub fn embed(&mut self, _samples: &[f32]) -> Result<Embedding> {
    unimplemented!("AudioEncoder::embed — implemented in Task 19")
  }

  /// Embed N clips of arbitrary length 1..=480_000 each.
  pub fn embed_batch(&mut self, _clips: &[&[f32]]) -> Result<Vec<Embedding>> {
    unimplemented!("AudioEncoder::embed_batch — implemented in Task 20")
  }

  /// Embed an arbitrary-length clip via textclap's chunking. NOT LAION-reference compatible.
  pub fn embed_chunked(
    &mut self,
    _samples: &[f32],
    _opts: &crate::options::ChunkingOptions,
  ) -> Result<Embedding> {
    unimplemented!("AudioEncoder::embed_chunked — implemented in Task 21")
  }

  /// Run a dummy forward to amortize ORT operator specialization and size scratch.
  pub fn warmup(&mut self) -> Result<()> {
    unimplemented!("AudioEncoder::warmup — implemented in Task 21")
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
  #[allow(dead_code)] // Used by Tasks 19-21.
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
  #[allow(dead_code)] // Used by Tasks 19-21.
  fn first_non_finite(samples: &[f32]) -> Option<usize> {
    for (i, &v) in samples.iter().enumerate() {
      if !v.is_finite() {
        return Some(i);
      }
    }
    None
  }
}
