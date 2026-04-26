//! Mel-spectrogram extractor (private to the crate). See spec §8.1 for the full pipeline.
//!
//! `T_FRAMES` is backfilled from `golden_params.json["T_frames"]` per §3.4. HTSAT input
//! normalization is `none` for this export (functional verification chose 'none' with drift
//! 1.10e-2 in the yellow zone — see `golden_params.json["htsat_norm_drift"]`); the optional
//! `HTSAT_INPUT_MEAN` / `HTSAT_INPUT_STD` constants stay commented out.

use crate::error::Result;

/// Mel time-frame count. Backfilled from `golden_params.json["T_frames"]` per §3.4.
pub(crate) const T_FRAMES: usize = 1001;

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

  /// Compute mel features and write into `out`. Caller must size `out` to exactly `T_FRAMES * 64`
  /// (time-major layout: one row per frame, 64 mel values per row).
  pub(crate) fn extract_into(&mut self, _samples: &[f32], _out: &mut [f32]) -> Result<()> {
    unimplemented!("MelExtractor::extract_into — implemented in Phase C")
  }
}
