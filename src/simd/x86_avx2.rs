//! x86_64 AVX2 + FMA backend for the three hot kernels.
//!
//! Selected by [`crate::simd`]'s dispatcher after
//! `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`
//! returns true. Each kernel carries `#[target_feature(enable = "avx2,fma")]`
//! so its intrinsics execute in an explicitly AVX2+FMA-enabled context
//! rather than one merely inherited from the x86_64 target's default
//! feature set.
//!
//! Filled in by Tasks SIMD-2/3/4. Until then every kernel forwards to
//! [`super::scalar`], which keeps the dispatcher's contract (byte-identical
//! to the scalar reference) trivially satisfied while the scaffold lands.

use rustfft::num_complex::Complex;

use super::scalar;

/// AVX2+FMA `power_spectrum_into`. Currently a placeholder that forwards
/// to [`scalar::power_spectrum_into`]; Task SIMD-2 replaces the body with
/// `core::arch::x86_64` intrinsics.
///
/// # Safety
///
/// The caller must ensure AVX2 and FMA are both available on the current
/// CPU. The dispatcher in [`crate::simd`] verifies this with
/// `is_x86_feature_detected!`. Calling this kernel on a CPU without those
/// features is undefined behavior.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn power_spectrum_into(buf: &[Complex<f64>], out: &mut [f64]) {
  // Placeholder — falls back to scalar until Task SIMD-2.
  scalar::power_spectrum_into(buf, out);
}

/// AVX2+FMA `mel_filterbank_dot`. Currently forwards to scalar (Task SIMD-3).
///
/// # Safety
///
/// Same AVX2+FMA availability contract as [`power_spectrum_into`].
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
  scalar::mel_filterbank_dot(weights, power)
}

/// AVX2+FMA `first_non_finite`. Currently forwards to scalar (Task SIMD-4).
///
/// # Safety
///
/// Same AVX2+FMA availability contract as [`power_spectrum_into`].
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn first_non_finite(samples: &[f32]) -> Option<usize> {
  scalar::first_non_finite(samples)
}
