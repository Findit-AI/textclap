//! aarch64 NEON backend for the three hot kernels.
//!
//! Selected by [`crate::simd`]'s dispatcher after
//! `is_aarch64_feature_detected!("neon")` returns true. Each kernel
//! carries `#[target_feature(enable = "neon")]` so its intrinsics execute
//! in an explicitly NEON-enabled context rather than one merely inherited
//! from the aarch64 target's default feature set.
//!
//! Filled in by Tasks SIMD-2/3/4. Until then every kernel forwards to
//! [`super::scalar`], which keeps the dispatcher's contract (byte-identical
//! to the scalar reference) trivially satisfied while the scaffold lands.

use rustfft::num_complex::Complex;

use super::scalar;

/// NEON `power_spectrum_into`. Currently a placeholder that forwards to
/// [`scalar::power_spectrum_into`]; Task SIMD-2 replaces the body with
/// `core::arch::aarch64` intrinsics.
///
/// # Safety
///
/// The caller must ensure NEON is available on the current CPU. The
/// dispatcher in [`crate::simd`] verifies this with
/// `is_aarch64_feature_detected!("neon")`. Calling this kernel on a CPU
/// without NEON is undefined behavior.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn power_spectrum_into(buf: &[Complex<f64>], out: &mut [f64]) {
  // Placeholder — falls back to scalar until Task SIMD-2.
  scalar::power_spectrum_into(buf, out);
}

/// NEON `mel_filterbank_dot`. Currently forwards to scalar (Task SIMD-3).
///
/// # Safety
///
/// Same NEON-availability contract as [`power_spectrum_into`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
  scalar::mel_filterbank_dot(weights, power)
}

/// NEON `first_non_finite`. Currently forwards to scalar (Task SIMD-4).
///
/// # Safety
///
/// Same NEON-availability contract as [`power_spectrum_into`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn first_non_finite(samples: &[f32]) -> Option<usize> {
  scalar::first_non_finite(samples)
}
