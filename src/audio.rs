//! Audio encoder (CLAP HTSAT side). See spec §7.3 / §8.2.

use std::path::Path;

use ort::session::Session;

use crate::clap_model::Embedding;
use crate::error::{Error, Result};
use crate::mel::MelExtractor;
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
