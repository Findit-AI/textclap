//! Mel-spectrogram extractor (private to the crate). See spec §8.1 for the full pipeline.
//!
//! `T_FRAMES` and (optional) `HTSAT_INPUT_MEAN` / `HTSAT_INPUT_STD` are backfilled from
//! `golden_params.json` per §3.4 step 3 → step 4. The skeleton uses placeholder values that
//! Phase C replaces.

use crate::error::Result;

/// Mel time-frame count. Backfilled from `golden_params.json["T_frames"]` per §3.4.
pub(crate) const T_FRAMES: usize = 1000; // PLACEHOLDER — replace with golden_params.json["T_frames"] in Task 6

// Optional HTSAT input-normalization constants. Defined only if §3.2's functional check chose
// `global_mean_std`; otherwise mel.rs has no normalization step.
//
// pub(crate) const HTSAT_INPUT_MEAN: f32 = -4.27;
// pub(crate) const HTSAT_INPUT_STD:  f32 =  4.57;

/// Mel-spectrogram extractor. Owns the Hann window, mel filterbank, and FFT planner.
pub(crate) struct MelExtractor {
  // Real fields land in Task 13–16.
}

impl MelExtractor {
  pub(crate) fn new() -> Self {
    unimplemented!("MelExtractor::new — implemented in Phase C")
  }

  /// Compute mel features and write into `out`. Caller must size `out` to exactly `64 * T_FRAMES`.
  pub(crate) fn extract_into(&mut self, _samples: &[f32], _out: &mut [f32]) -> Result<()> {
    unimplemented!("MelExtractor::extract_into — implemented in Phase C")
  }
}
