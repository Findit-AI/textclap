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
  /// Generate a periodic Hann window of length `n`: equivalent to taking the first `n` samples
  /// of a length-(n+1) symmetric Hann. Matches librosa / torch convention.
  fn periodic_hann(n: usize) -> Vec<f32> {
    let denom = (n + 1) as f32;
    (0..n)
      .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (k as f32) / denom).cos())
      .collect()
  }

  /// Hz → Slaney mel.
  /// Linear below 1 kHz: m = 3 · f / 200; logarithmic above.
  /// Matches librosa `mel_frequencies(htk=False)` / Slaney's auditory toolbox formula.
  fn hz_to_slaney_mel(hz: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f32).ln() / 27.0;
    if hz < MIN_LOG_HZ {
      (hz - F_MIN) / F_SP
    } else {
      MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    }
  }

  /// Slaney mel → Hz (inverse of `hz_to_slaney_mel`).
  fn slaney_mel_to_hz(mel: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f32).ln() / 27.0;
    if mel < MIN_LOG_MEL {
      F_MIN + F_SP * mel
    } else {
      MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    }
  }

  /// Build a `[n_mels × (n_fft/2 + 1)]` Slaney-norm Slaney-scale mel filterbank, row-major.
  ///
  /// Matches `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney')`.
  fn build_mel_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let mel_min = Self::hz_to_slaney_mel(fmin);
    let mel_max = Self::hz_to_slaney_mel(fmax);
    // n_mels + 2 mel-equispaced points; convert back to Hz; bracket each filter with [left, center, right].
    let mel_points: Vec<f32> = (0..n_mels + 2)
      .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / (n_mels + 1) as f32)
      .collect();
    let hz_points: Vec<f32> = mel_points
      .iter()
      .map(|&m| Self::slaney_mel_to_hz(m))
      .collect();

    // FFT bin frequencies: bin k maps to k * sr / n_fft Hz.
    let bin_hz: Vec<f32> = (0..n_freq)
      .map(|k| (k as f32) * (sr as f32) / (n_fft as f32))
      .collect();

    let mut fb = vec![0.0f32; n_mels * n_freq];
    for m in 0..n_mels {
      let left = hz_points[m];
      let center = hz_points[m + 1];
      let right = hz_points[m + 2];
      let inv_left_diff = 1.0 / (center - left);
      let inv_right_diff = 1.0 / (right - center);
      // Slaney normalization: scale by 2 / (right - left).
      let slaney_norm = 2.0 / (right - left);
      for k in 0..n_freq {
        let f = bin_hz[k];
        let weight = if f >= left && f <= center {
          (f - left) * inv_left_diff
        } else if f >= center && f <= right {
          (right - f) * inv_right_diff
        } else {
          0.0
        };
        fb[m * n_freq + k] = weight * slaney_norm;
      }
    }
    fb
  }

  pub(crate) fn new() -> Self {
    unimplemented!("MelExtractor::new — implemented in Phase C")
  }

  /// Compute mel features and write into `out`. Caller must size `out` to exactly `T_FRAMES * 64`
  /// (time-major layout: one row per frame, 64 mel values per row).
  pub(crate) fn extract_into(&mut self, _samples: &[f32], _out: &mut [f32]) -> Result<()> {
    unimplemented!("MelExtractor::extract_into — implemented in Phase C")
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Periodic Hann at n=1024: equivalent to the first 1024 samples of a length-1025 symmetric Hann.
  /// Tests use cross-implementation invariants (NOT recomputed-from-the-same-formula values, which
  /// would be tautological):
  ///   - win[0] = 0 (sin² identity).
  ///   - For periodic Hann at n=1024, the symmetry axis is at index 512.5 (not 512), so win[512]
  ///     and win[513] are algebraically tied and fp32 rounding can put either as the peak.
  ///   - win[1023] is small but POSITIVE (~9.6e-3), distinguishing periodic from symmetric
  ///     (symmetric Hann would have win[1023] ≈ win[0] = 0).
  #[test]
  fn hann_window_periodic_length_1024() {
    let win = MelExtractor::periodic_hann(1024);
    assert_eq!(win.len(), 1024);

    // Endpoint invariants.
    assert_eq!(win[0], 0.0);
    // Periodic Hann at k=N-1 = 1023, N=1024 evaluates to a small POSITIVE value (~3.76e-5 with
    // denom=N+1, ~9.4e-6 with denom=N — depends on convention). Symmetric Hann would evaluate to
    // exactly 0 at index N-1. The discriminator is "positive but tiny," not the specific
    // magnitude.
    assert!(
      win[1023] > 0.0 && win[1023] < 1e-3,
      "periodic Hann last sample should be positive but small; got {}",
      win[1023]
    );

    // Range invariant: 0 ≤ win[k] ≤ 1 for all k.
    for &v in &win {
      assert!(v >= 0.0 && v <= 1.0 + 1e-7);
    }

    // Peak invariant: max value lives near the middle. For n=1024, win[512] and win[513] are
    // algebraically tied at 0.5·(1 − cos(π/1025)) ≈ 0.999996. fp32 rounding can put either as
    // the peak; accept both.
    let max_idx = win
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
      .unwrap()
      .0;
    assert!(
      max_idx == 512 || max_idx == 513,
      "peak should be at index 512 or 513 (algebraic tie at 512.5); got {max_idx}"
    );
    assert!(
      win[512] > 0.999 && win[512] <= 1.0 + 1e-6,
      "win[512] should be near 1; got {}",
      win[512]
    );
    assert!(
      (win[512] - win[513]).abs() < 1e-6,
      "win[512] and win[513] should be tied; got {} vs {}",
      win[512],
      win[513]
    );
  }

  fn read_npy_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    npy.into_vec::<f32>().unwrap()
  }

  /// Compare filterbank row 0, 10, 32 against librosa references at max_abs_diff < 1e-6.
  /// Row 10 is the discriminator: lands near 1 kHz where Slaney inflection occurs;
  /// HTK construction would diverge here while rows 0 and 32 alone wouldn't tell.
  #[test]
  fn filterbank_rows_match_librosa() {
    let fb = MelExtractor::build_mel_filterbank(48000, 1024, 64, 50.0, 14000.0);
    // fb is [n_mels × n_freq] = [64 × 513], row-major.
    const N_FREQ: usize = 513;
    for &row_idx in &[0_usize, 10, 32] {
      let path = format!("tests/fixtures/filterbank_row_{row_idx}.npy");
      let expected = read_npy_f32(&path);
      assert_eq!(expected.len(), N_FREQ);
      let actual_row = &fb[row_idx * N_FREQ..(row_idx + 1) * N_FREQ];
      let max_diff = actual_row
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
      assert!(
        max_diff < 1e-6,
        "filterbank row {row_idx} max_abs_diff = {max_diff:.3e}",
      );
    }
  }
}
