//! Scalar reference implementations of the three hot kernels.
//!
//! Always compiled. SIMD backends in [`super::neon`] and [`super::x86_avx2`]
//! dispatch to these as their tail fallback (and currently as their full
//! body until Tasks SIMD-2/3/4 fill in the intrinsics). The dispatcher in
//! [`super`] selects the best backend at call time.
//!
//! These implementations are bit-identical to the original inline loops
//! that lived in `mel.rs` and `audio.rs` before the SIMD scaffold was
//! introduced. SIMD backends must produce results that are byte-identical
//! to these references on every input — equivalence tests in later tasks
//! enforce that contract.

use rustfft::num_complex::Complex;

/// `out[i] = buf[i].re² + buf[i].im²` for every element. Bit-identical to
/// the original loop body in `Mel::stft_one_frame_power`.
///
/// # Panics (debug builds)
///
/// `buf.len()` must equal `out.len()`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn power_spectrum_into(buf: &[Complex<f64>], out: &mut [f64]) {
  debug_assert_eq!(buf.len(), out.len());
  for (c, p) in buf.iter().zip(out.iter_mut()) {
    *p = c.re * c.re + c.im * c.im;
  }
}

/// `Σ weights[i] * power[i]`. Bit-identical to the inner mel-filterbank
/// dot product in `Mel::extract_into`.
///
/// # Panics (debug builds)
///
/// `weights.len()` must equal `power.len()`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
  debug_assert_eq!(weights.len(), power.len());
  let mut acc = 0.0f64;
  for (w, p) in weights.iter().zip(power.iter()) {
    acc += w * p;
  }
  acc
}

/// Returns `Some(i)` of the first non-finite (NaN or ±∞) sample, or `None`
/// if every sample is finite. Bit-identical to the original
/// `AudioEncoder::first_non_finite` scan.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn first_non_finite(samples: &[f32]) -> Option<usize> {
  for (i, &v) in samples.iter().enumerate() {
    if !v.is_finite() {
      return Some(i);
    }
  }
  None
}
