//! Mel-spectrogram extractor (private to the crate). See spec §8.1 for the full pipeline.
//!
//! `T_FRAMES` is backfilled from `golden_params.json["T_frames"]` per §3.4. HTSAT input
//! normalization is `none` for this export (functional verification chose 'none' with drift
//! 1.10e-2 in the yellow zone — see `golden_params.json["htsat_norm_drift"]`); the optional
//! `HTSAT_INPUT_MEAN` / `HTSAT_INPUT_STD` constants stay commented out.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::error::Result;

/// Mel time-frame count. Backfilled from `golden_params.json["T_frames"]` per §3.4.
pub(crate) const T_FRAMES: usize = 1001;

const N_FFT: usize = 1024;
const HOP: usize = 480;
const N_MELS: usize = 64;
const SR: u32 = 48_000;
const TARGET_SAMPLES: usize = 480_000;
const FMIN: f64 = 50.0;
const FMAX: f64 = 14_000.0;
const POWER_TO_DB_AMIN: f64 = 1e-10;

// Optional HTSAT input-normalization constants. Defined only if §3.2's functional check chose
// `global_mean_std`; otherwise mel.rs has no normalization step.
//
// pub(crate) const HTSAT_INPUT_MEAN: f32 = -4.27;
// pub(crate) const HTSAT_INPUT_STD:  f32 =  4.57;

/// Mel-spectrogram extractor. Owns the Hann window, mel filterbank, and FFT planner.
///
/// FFT is computed in f64 to match HuggingFace's `transformers.audio_utils.spectrogram`, which
/// promotes to float64 internally. Using f32 leaves a ~1.24e-4 drift in the dB output that
/// exceeds the spec §12.2 1e-4 budget.
pub(crate) struct MelExtractor {
  window: Vec<f64>,       // length N_FFT
  filterbank: Vec<f64>,   // length N_MELS × (N_FFT/2 + 1)
  fft: Arc<dyn Fft<f64>>, // FftPlanner output for N_FFT
}

impl MelExtractor {
  /// Generate a periodic Hann window of length `n`: equivalent to taking the first `n` samples
  /// of a length-(n+1) symmetric Hann. Matches librosa / torch convention.
  ///
  /// Formula: `w[k] = 0.5 − 0.5·cos(2π·k / n)` for `k ∈ [0, n)`. This is what
  /// `numpy.hanning(n+1)[:-1]`, `torch.hann_window(n, periodic=True)`, and
  /// `librosa.filters.get_window("hann", n, fftbins=True)` all return; see HF
  /// `transformers.audio_utils.window_function("hann", periodic=True)`.
  fn periodic_hann(n: usize) -> Vec<f64> {
    let denom = n as f64;
    (0..n)
      .map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (k as f64) / denom).cos())
      .collect()
  }

  /// Hz → Slaney mel.
  /// Linear below 1 kHz: m = 3 · f / 200; logarithmic above.
  /// Matches librosa `mel_frequencies(htk=False)` / Slaney's auditory toolbox formula.
  /// Computed in f64 to match HF's float64 reference filterbank.
  fn hz_to_slaney_mel(hz: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f64).ln() / 27.0;
    if hz < MIN_LOG_HZ {
      (hz - F_MIN) / F_SP
    } else {
      MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    }
  }

  /// Slaney mel → Hz (inverse of `hz_to_slaney_mel`).
  fn slaney_mel_to_hz(mel: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f64).ln() / 27.0;
    if mel < MIN_LOG_MEL {
      F_MIN + F_SP * mel
    } else {
      MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    }
  }

  /// Build a `[n_mels × (n_fft/2 + 1)]` Slaney-norm Slaney-scale mel filterbank, row-major,
  /// in f64 to match HF's `mel_filter_bank` float64 reference.
  ///
  /// Matches `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney')`.
  fn build_mel_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let n_freq = n_fft / 2 + 1;
    let mel_min = Self::hz_to_slaney_mel(fmin);
    let mel_max = Self::hz_to_slaney_mel(fmax);
    // n_mels + 2 mel-equispaced points; convert back to Hz; bracket each filter with [left, center, right].
    let mel_points: Vec<f64> = (0..n_mels + 2)
      .map(|i| mel_min + (mel_max - mel_min) * (i as f64) / (n_mels + 1) as f64)
      .collect();
    let hz_points: Vec<f64> = mel_points
      .iter()
      .map(|&m| Self::slaney_mel_to_hz(m))
      .collect();

    // FFT bin frequencies: bin k maps to k * sr / n_fft Hz.
    let bin_hz: Vec<f64> = (0..n_freq)
      .map(|k| (k as f64) * (sr as f64) / (n_fft as f64))
      .collect();

    let mut fb = vec![0.0f64; n_mels * n_freq];
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
    let window = Self::periodic_hann(N_FFT);
    let filterbank = Self::build_mel_filterbank(SR, N_FFT, N_MELS, FMIN, FMAX);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    Self {
      window,
      filterbank,
      fft,
    }
  }

  /// Compute |X[k]|² for a single Hann-windowed frame of length N_FFT.
  /// `power` must have length N_FFT/2 + 1 = 513. Computed in f64 to match HF.
  fn stft_one_frame_power(&self, frame: &[f64], power: &mut [f64]) {
    debug_assert_eq!(frame.len(), N_FFT);
    debug_assert_eq!(power.len(), N_FFT / 2 + 1);
    // Window the frame, build a Complex<f64> buffer, run a full complex FFT, then take the
    // first N_FFT/2 + 1 bins (real-FFT identity).
    let mut buf: Vec<Complex<f64>> = frame
      .iter()
      .zip(self.window.iter())
      .map(|(&s, &w)| Complex::<f64>::new(s * w, 0.0))
      .collect();
    self.fft.process(&mut buf);
    for k in 0..(N_FFT / 2 + 1) {
      let c = buf[k];
      power[k] = c.re * c.re + c.im * c.im;
    }
  }

  /// Compute mel features and write into `out`. Caller must size `out` to exactly `T_FRAMES * 64`
  /// (time-major layout: one row per frame, 64 mel values per row).
  ///
  /// Internally promotes to f64 to match HF `transformers.audio_utils.spectrogram` (which casts
  /// `waveform.astype(np.float64)` before STFT). Only the final dB output is cast back to f32.
  pub(crate) fn extract_into(&mut self, samples: &[f32], out: &mut [f32]) -> Result<()> {
    debug_assert_eq!(out.len(), N_MELS * T_FRAMES);

    // 1. Repeat-pad or head-truncate to TARGET_SAMPLES, mirroring HF `repeatpad`:
    //   n_repeat = floor(max_length / len(waveform)); waveform = tile(waveform, n_repeat)
    //   then zero-pad the remainder.
    let mut padded: Vec<f64> = Vec::with_capacity(TARGET_SAMPLES);
    if samples.len() >= TARGET_SAMPLES {
      padded.extend(samples[..TARGET_SAMPLES].iter().map(|&s| s as f64));
    } else {
      let n_repeat = TARGET_SAMPLES / samples.len();
      for _ in 0..n_repeat {
        padded.extend(samples.iter().map(|&s| s as f64));
      }
      // Zero-pad the remainder (HF uses np.pad mode='constant', constant_values=0).
      padded.resize(TARGET_SAMPLES, 0.0);
    }

    // 2. center=True reflection padding: prepend and append N_FFT/2 reflected samples.
    // librosa's `center=True` mode pads the signal with reflection at both ends so the
    // first STFT frame is centered at sample 0, not starting at sample 0.
    let half_fft = N_FFT / 2;
    let mut centered: Vec<f64> = Vec::with_capacity(TARGET_SAMPLES + 2 * half_fft);
    // Prefix: reflect samples 1..=half_fft from the start in reverse.
    for i in 0..half_fft {
      centered.push(padded[half_fft - i]);
    }
    centered.extend_from_slice(&padded);
    // Suffix: reflect samples (TARGET_SAMPLES - 2)..(TARGET_SAMPLES - 2 - half_fft) in reverse.
    for i in 0..half_fft {
      centered.push(padded[TARGET_SAMPLES - 2 - i]);
    }
    debug_assert_eq!(centered.len(), TARGET_SAMPLES + 2 * half_fft);

    // 3. STFT loop: T_FRAMES frames of N_FFT samples, hop=HOP, time-major output.
    let mut frame = vec![0.0f64; N_FFT];
    let mut power = vec![0.0f64; N_FFT / 2 + 1];

    for t in 0..T_FRAMES {
      let start = t * HOP;
      let end = start + N_FFT;
      // The reflection-padded signal has length TARGET_SAMPLES + N_FFT.
      // For T = 1001, last frame ends at 1000 * 480 + 1024 = 481024 = TARGET_SAMPLES + N_FFT.
      frame.copy_from_slice(&centered[start..end]);

      self.stft_one_frame_power(&frame, &mut power);

      // 4. Mel filterbank multiply + 5. power_to_dB floor (single 10·log10 application,
      // ref=1.0, amin=1e-10 ⇒ floor at -100 dB; matches HF `power_to_db`).
      for mel_bin in 0..N_MELS {
        let row = &self.filterbank[mel_bin * (N_FFT / 2 + 1)..(mel_bin + 1) * (N_FFT / 2 + 1)];
        let mut acc = 0.0f64;
        for (w, p) in row.iter().zip(power.iter()) {
          acc += w * p;
        }
        let clipped = acc.max(POWER_TO_DB_AMIN);
        let db = 10.0 * clipped.log10();
        // 6. Time-major output: out[t * 64 + mel_bin]
        out[t * N_MELS + mel_bin] = db as f32;
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Periodic Hann at n=1024: equivalent to `numpy.hanning(1025)[:-1]` /
  /// `torch.hann_window(1024, periodic=True)` / `librosa.filters.get_window("hann", 1024, fftbins=True)`.
  /// Formula `w[k] = 0.5 − 0.5·cos(2π·k / n)` puts the unique peak (1.0) exactly at index n/2.
  /// Tests use cross-implementation invariants:
  ///   - win[0] = 0 (cos(0) = 1).
  ///   - win[512] = 1 exactly (cos(π) = −1).
  ///   - win[1023] is small but POSITIVE (~9.4e-6), distinguishing periodic from symmetric
  ///     (symmetric Hann would have win[N-1] = 0).
  #[test]
  fn hann_window_periodic_length_1024() {
    let win = MelExtractor::periodic_hann(1024);
    assert_eq!(win.len(), 1024);

    // Endpoint invariants.
    assert_eq!(win[0], 0.0);
    // Periodic Hann at k=N-1 = 1023, N=1024 evaluates to 0.5·(1 − cos(2π·1023/1024))
    // ≈ 9.4e-6 — POSITIVE but tiny. Symmetric Hann would evaluate to exactly 0 at index N-1.
    assert!(
      win[1023] > 0.0 && win[1023] < 1e-3,
      "periodic Hann last sample should be positive but small; got {}",
      win[1023]
    );

    // Range invariant: 0 ≤ win[k] ≤ 1 for all k.
    for &v in &win {
      assert!(v >= 0.0 && v <= 1.0 + 1e-7);
    }

    // Peak invariant: for periodic Hann with denom=N, the unique peak (1.0) sits at index N/2.
    let max_idx = win
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
      .unwrap()
      .0;
    assert_eq!(max_idx, 512, "peak must be exactly at index N/2 = 512");
    assert_eq!(win[512], 1.0, "win[512] should be exactly 1.0");
    // win[513] = 0.5·(1 − cos(2π·513/1024)) ≈ 1 − 9.4e-6 (slightly less than 1 by symmetry).
    assert!(
      win[513] < 1.0 && win[513] > 0.999,
      "win[513] should be just below 1; got {}",
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
    // fb is [n_mels × n_freq] = [64 × 513], row-major (f64 internally).
    const N_FREQ: usize = 513;
    for &row_idx in &[0_usize, 10, 32] {
      let path = format!("tests/fixtures/filterbank_row_{row_idx}.npy");
      let expected = read_npy_f32(&path);
      assert_eq!(expected.len(), N_FREQ);
      let actual_row = &fb[row_idx * N_FREQ..(row_idx + 1) * N_FREQ];
      let max_diff = actual_row
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| ((*a as f32) - b).abs())
        .fold(0.0f32, f32::max);
      assert!(
        max_diff < 1e-6,
        "filterbank row {row_idx} max_abs_diff = {max_diff:.3e}",
      );
    }
  }

  /// Hann-windowed STFT of a 1 kHz sine wave at 48 kHz should peak at the bin closest to
  /// k = 1000 / (48000 / 1024) = 21.33 → bin 21 (or 22).
  #[test]
  fn stft_peaks_at_expected_bin() {
    let mel = MelExtractor::new();
    let sr = 48000_f64;
    let freq = 1000.0_f64;
    let mut samples = Vec::with_capacity(1024);
    for k in 0..1024 {
      samples.push((2.0 * std::f64::consts::PI * freq * (k as f64) / sr).sin());
    }
    let mut power = vec![0.0f64; 513];
    mel.stft_one_frame_power(&samples, &mut power);
    // Find peak bin
    let (peak_bin, _) = power
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
      .unwrap();
    assert!(
      peak_bin == 21 || peak_bin == 22,
      "expected peak at bin 21 or 22, got bin {peak_bin}",
    );
  }

  /// Compare full mel-extraction output against `tests/fixtures/golden_mel.npy` at max_abs_diff < 1e-4.
  /// Fixtures are committed; this test does NOT require TEXTCLAP_MODELS_DIR.
  #[test]
  fn extract_into_matches_golden_mel() {
    let golden = read_npy_f32("tests/fixtures/golden_mel.npy");
    assert_eq!(golden.len(), N_MELS * T_FRAMES, "golden mel shape mismatch");

    // Read the WAV via hound. WAV is 48 kHz mono, may be i16 or f32.
    let mut reader = hound::WavReader::open("tests/fixtures/sample.wav").expect("open sample.wav");
    let samples: Vec<f32> = match reader.spec().sample_format {
      hound::SampleFormat::Int => {
        let bits = reader.spec().bits_per_sample;
        let scale = 1.0 / (1_i64 << (bits - 1)) as f32;
        reader.samples::<i32>().map(|s| s.unwrap() as f32 * scale).collect()
      }
      hound::SampleFormat::Float => {
        reader.samples::<f32>().map(|s| s.unwrap()).collect()
      }
    };

    let mut mel = MelExtractor::new();
    let mut out = vec![0.0f32; N_MELS * T_FRAMES];
    mel.extract_into(&samples, &mut out).expect("extract_into");

    let max_diff = out
      .iter()
      .zip(golden.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0f32, f32::max);
    assert!(
      max_diff < 1e-4,
      "mel features max_abs_diff = {max_diff:.3e} (budget 1e-4)",
    );
  }

  #[test]
  fn power_to_db_applied_once() {
    // Sanity: a unit-amplitude 1 kHz sine should produce mel values in the single-log10 range,
    // not the (much smaller) double-log range. The HF reference (`ClapFeatureExtractor` on
    // identical samples) lands at max ≈ 29.29 dB and min = -100 dB for this signal — energy
    // concentrates in one mel bin so the peak is well above the spec's "+5 dB typical-audio"
    // estimate. Bounds widened to match the HF-verified single-application range; a
    // double-application bug would compress the peak to `10·log10(29) ≈ 14.6` dB or NaN
    // (when the inner result is < amin), and a missing log10 entirely would push the peak
    // far above 50 (raw power values reach ~10⁵).
    let mut mel = MelExtractor::new();
    let sr = 48_000_f32;
    let mut samples = vec![0.0f32; TARGET_SAMPLES];
    for k in 0..TARGET_SAMPLES {
      samples[k] = (2.0 * std::f32::consts::PI * 1000.0 * (k as f32) / sr).sin();
    }
    let mut out = vec![0.0f32; N_MELS * T_FRAMES];
    mel.extract_into(&samples, &mut out).unwrap();
    let max = out.iter().fold(f32::MIN, |a, &b| a.max(b));
    let min = out.iter().fold(f32::MAX, |a, &b| a.min(b));
    // Single-application range for unit sine: max ≈ 29.29, min = -100 (amin floor).
    assert!(
      max > 20.0 && max < 50.0,
      "single-application 10·log10 of unit-sine mel should peak near 29.3 dB; got max = {max}",
    );
    assert!(
      min >= -100.0 - 1e-3 && min < -50.0,
      "amin floor should clip silent bins to -100 dB; got min = {min}",
    );
  }
}
