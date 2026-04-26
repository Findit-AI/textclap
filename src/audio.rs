//! Audio encoder (CLAP HTSAT side). See spec §7.3 / §8.2.

use std::path::Path;

use crate::{clap_model::Embedding, error::Result, options::Options};

// Backfilled from golden_onnx_io.json per §3.4. Module-private.
const AUDIO_INPUT_NAME: &str = "input_features";
const AUDIO_OUTPUT_NAME: &str = "audio_embeds";

/// Compile-time const indicating whether the audio ONNX output is already L2-normalized.
/// Backfilled from `golden_onnx_io.json["audio_output_is_unit_norm"]` per §3.4.
const AUDIO_OUTPUT_IS_UNIT_NORM: bool = false;

/// Audio encoder. See spec §7.3.
pub struct AudioEncoder {
  // Real fields land in Task 17–21.
}

impl AudioEncoder {
  /// Load from a file path.
  pub fn from_file<P: AsRef<Path>>(_onnx_path: P, _opts: Options) -> Result<Self> {
    unimplemented!("AudioEncoder::from_file — implemented in Task 17")
  }

  /// Load from caller-supplied bytes (copied into the ORT session).
  pub fn from_memory(_onnx_bytes: &[u8], _opts: Options) -> Result<Self> {
    unimplemented!("AudioEncoder::from_memory — implemented in Task 17")
  }

  /// Wrap a pre-built ORT session. See spec §7.3 for the asymmetric purposes.
  pub fn from_ort_session(_session: ort::session::Session, _opts: Options) -> Result<Self> {
    unimplemented!("AudioEncoder::from_ort_session — implemented in Task 21")
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
