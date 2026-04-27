//! x86_64 AVX2 + FMA backend for the three hot kernels.
//!
//! Selected by [`crate::simd`]'s dispatcher after
//! `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`
//! returns true. Each kernel carries `#[target_feature(enable = "avx2,fma")]`
//! so its intrinsics execute in an explicitly AVX2+FMA-enabled context
//! rather than one merely inherited from the x86_64 target's default
//! feature set. Same `#[target_feature]` placement and `#[inline]`
//! policy as `colconv::row::arch::x86_avx2`.
//!
//! `mel_filterbank_dot` and `first_non_finite` remain placeholder
//! forwarders to [`super::scalar`] until Tasks SIMD-3 / SIMD-4 fill in
//! their intrinsics.

use core::arch::x86_64::*;
use rustfft::num_complex::Complex;

use super::scalar;

/// AVX2+FMA `power_spectrum_into`. Computes `out[i] = buf[i].re² + buf[i].im²`
/// using a 256-bit unaligned load of 2 `Complex<f64>` (= 4 f64
/// `[r0, i0, r1, i1]`) per iteration, then squares the lanes and
/// horizontally pair-adds the two 128-bit halves with `_mm_hadd_pd` to
/// produce `[r0² + i0², r1² + i1²]`. The odd-tail element (if any)
/// falls back to the scalar reference.
///
/// Output is bit-identical to [`scalar::power_spectrum_into`]: only
/// element-wise mul + add are used; no FMA reassociation.
///
/// # Safety
///
/// The caller must ensure AVX2 and FMA are both available on the current
/// CPU. The dispatcher in [`crate::simd`] verifies this with
/// `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`.
/// Calling this kernel on a CPU without those features is undefined
/// behavior.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn power_spectrum_into(buf: &[Complex<f64>], out: &mut [f64]) {
  debug_assert_eq!(buf.len(), out.len());
  let n = buf.len();
  let n_pairs = n / 2;

  // `Complex<f64>` is `#[repr(C)]` with `[re, im]` layout, so a
  // `&[Complex<f64>]` of length `n` aliases a contiguous f64 stream
  // `[r0, i0, r1, i1, …]` of length `2 * n`. The slice is f64-aligned
  // (8 bytes), not 32-byte; the kernel uses `_mm256_loadu_pd` /
  // `_mm_storeu_pd` and never demands 16- or 32-byte alignment.
  let buf_ptr = buf.as_ptr() as *const f64;
  let out_ptr = out.as_mut_ptr();

  for i in 0..n_pairs {
    // SAFETY: `i < n_pairs = n / 2` ⇒ `i*4 + 3 < 2*n`, so 4 contiguous
    // f64 reads sit inside the `2*n`-long f64 view of `buf`.
    let v = unsafe { _mm256_loadu_pd(buf_ptr.add(i * 4)) }; // [r0, i0, r1, i1]
    // Square element-wise: [r0², i0², r1², i1²]. No FMA — same round
    // structure as the scalar reference (`re*re` and `im*im` finish to
    // f64 before the add).
    let sq = _mm256_mul_pd(v, v);
    // Split into 128-bit halves: lo = [r0², i0²], hi = [r1², i1²].
    // `_mm256_castpd256_pd128` is a no-op cast that keeps lanes 0..1;
    // `_mm256_extractf128_pd::<1>` lifts lanes 2..3 down. Both are free
    // on Intel and Zen.
    let lo = _mm256_castpd256_pd128(sq);
    let hi = _mm256_extractf128_pd::<1>(sq);
    // `_mm_hadd_pd(a, b)` produces `[a0 + a1, b0 + b1]`. Feeding lo and
    // hi yields `[r0² + i0², r1² + i1²]` — exactly the two powers we
    // need, in scalar-reference order.
    let p0 = _mm_hadd_pd(lo, hi);
    // SAFETY: `i*2 + 1 < n` (since `i < n/2`) and `out.len() == n`,
    // so writing 2 f64 starting at `out_ptr.add(i*2)` is in-bounds.
    unsafe { _mm_storeu_pd(out_ptr.add(i * 2), p0) };
  }

  // Tail: 1 element if `n` is odd. Body matches the scalar inner loop.
  if n & 1 == 1 {
    let last = n - 1;
    let c = buf[last];
    out[last] = c.re * c.re + c.im * c.im;
  }
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
