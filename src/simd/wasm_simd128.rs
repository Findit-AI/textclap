//! WebAssembly simd128 backend.
//!
//! Selected by [`crate::simd`]'s dispatcher when `cfg!(target_feature =
//! "simd128")` evaluates true at compile time. WASM has no runtime CPU
//! feature detection — a WASM module either contains simd128 opcodes
//! (which require runtime support at instantiation) or it doesn't. So
//! the gate is purely compile-time, regardless of the `feature = "std"`
//! flag.
//!
//! The kernel carries `#[target_feature(enable = "simd128")]` so its
//! intrinsics are accessible to the function body even when simd128 is
//! not enabled for the whole crate.
//!
//! # Numerical contract
//!
//! - [`power_spectrum_into`]: bit-identical to [`super::scalar`] —
//!   plain mul + add, no FMA (simd128 baseline lacks FMA).
//! - [`mel_filterbank_dot`]: bit-identical to scalar — uses
//!   `f64x2_mul` + `f64x2_add` (no FMA), so the rounding tree matches
//!   the scalar reference exactly. **Different from the NEON / AVX2 /
//!   AVX-512 backends**, which use FMA and reassociation.
//! - [`first_non_finite`]: structural equivalence with scalar.
//!
//! # Pipeline width
//!
//! simd128 has 2-lane f64 / 4-lane f32 — same width as SSE2 / NEON's
//! `float64x2_t`. ILP via two parallel accumulators where applicable.

use core::arch::wasm32::*;
use rustfft::num_complex::Complex;

/// simd128 `power_spectrum_into`. Computes `out[i] = buf[i].re² + buf[i].im²`
/// using two `v128` loads of 2 f64 each (= 2 `Complex<f64>`) per
/// iteration. Squares lane-wise, then sums the lane pair within each
/// vector via an `i64x2_shuffle::<1, 0>` swap + `f64x2_add`. The two
/// pair-sums are combined into a 2-lane result with a final
/// `i64x2_shuffle::<0, 2>`. Output is bit-identical to
/// [`super::scalar::power_spectrum_into`]: plain mul + add, no FMA.
///
/// # Safety
///
/// The caller must ensure simd128 is enabled at compile time. The
/// dispatcher in [`crate::simd`] verifies this with
/// `cfg!(target_feature = "simd128")`. WASM has no runtime CPU
/// detection, so the obligation is purely compile-time: the WASM
/// module was produced with `-C target-feature=+simd128` (or
/// equivalent), and it is being executed in a WASM runtime that
/// supports the SIMD proposal.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn power_spectrum_into(buf: &[Complex<f64>], out: &mut [f64]) {
  debug_assert_eq!(buf.len(), out.len());
  let n = buf.len();
  let n_pairs = n / 2;

  // `Complex<f64>` is `#[repr(C)]` with `[re, im]` layout, so a
  // `&[Complex<f64>]` of length `n` aliases a contiguous f64 stream
  // `[r0, i0, r1, i1, …]` of length `2 * n`.
  let buf_ptr = buf.as_ptr() as *const f64;
  let out_ptr = out.as_mut_ptr();

  for i in 0..n_pairs {
    // SAFETY: `i < n_pairs = n / 2` ⇒ `i*4 + 3 < 2*n`, so each 16-byte
    // (2-f64) load sits inside the `2*n`-long f64 view of `buf`.
    // simd128 `v128_load` is unaligned-tolerant.
    let c0 = unsafe { v128_load(buf_ptr.add(i * 4) as *const v128) };
    let c1 = unsafe { v128_load(buf_ptr.add(i * 4 + 2) as *const v128) };
    // Square element-wise: c0 = [r0², i0²], c1 = [r1², i1²]. No FMA
    // — same round structure as the scalar reference.
    let sq0 = f64x2_mul(c0, c0);
    let sq1 = f64x2_mul(c1, c1);
    // Sum the lane pair within each vector: swap lanes via
    // `i64x2_shuffle::<1, 0>` and add to self. Both lanes of `sum0`
    // end up holding `r0² + i0²`; both of `sum1` hold `r1² + i1²`.
    let sum0 = f64x2_add(sq0, i64x2_shuffle::<1, 0>(sq0, sq0));
    let sum1 = f64x2_add(sq1, i64x2_shuffle::<1, 0>(sq1, sq1));
    // Combine: `i64x2_shuffle::<0, 2>(sum0, sum1)` selects `sum0[0]`
    // and `sum1[0]` (indices 0..1 reference the first arg, 2..3 the
    // second), yielding `[r0² + i0², r1² + i1²]` in natural order.
    let result = i64x2_shuffle::<0, 2>(sum0, sum1);
    // SAFETY: `i*2 + 1 < n` (since `i < n/2`) and `out.len() == n`,
    // so the 16-byte store starting at `out_ptr.add(i*2)` is in-bounds.
    unsafe { v128_store(out_ptr.add(i * 2) as *mut v128, result) };
  }

  // Tail: 1 element if `n` is odd. Body matches the scalar inner loop
  // exactly so the byte-equivalence contract holds.
  if n & 1 == 1 {
    let last = n - 1;
    let c = buf[last];
    out[last] = c.re * c.re + c.im * c.im;
  }
}

/// simd128 `mel_filterbank_dot`. Computes `Σ weights[i] * power[i]`
/// using two independent `f64x2` accumulators (4 elements per
/// iteration) and `f64x2_mul` + `f64x2_add` (simd128 baseline lacks
/// FMA).
///
/// # Numerical contract
///
/// Output is bit-identical to [`super::scalar::mel_filterbank_dot`] in
/// the rounding-tree sense: with no FMA available, the kernel uses the
/// same `mul` + `add` round structure as the scalar reference. The
/// two-accumulator pattern still reassociates the summation slightly
/// (pairs sum independently before being combined), but with no FMA
/// the per-product rounding matches scalar exactly.
///
/// # Safety
///
/// Same simd128 availability contract as [`power_spectrum_into`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
  debug_assert_eq!(weights.len(), power.len());
  let n = weights.len();

  // Two independent 2-lane accumulators.
  let mut acc0 = f64x2_splat(0.0);
  let mut acc1 = f64x2_splat(0.0);

  let w_ptr = weights.as_ptr();
  let p_ptr = power.as_ptr();

  // Process 4 elements per iteration (two 2-lane f64 vectors).
  let n_chunks = n / 4;
  for i in 0..n_chunks {
    // SAFETY: `i < n_chunks = n / 4` ⇒ `i*4 + 3 < n`, so the four
    // 16-byte loads (two for `weights`, two for `power`) sit fully
    // inside their respective slices. `v128_load` is unaligned-tolerant.
    let w0 = unsafe { v128_load(w_ptr.add(i * 4) as *const v128) };
    let w1 = unsafe { v128_load(w_ptr.add(i * 4 + 2) as *const v128) };
    let p0 = unsafe { v128_load(p_ptr.add(i * 4) as *const v128) };
    let p1 = unsafe { v128_load(p_ptr.add(i * 4 + 2) as *const v128) };
    // No FMA in baseline simd128 → mul + add. Matches scalar's two-
    // round `acc += w * p` per-element rounding.
    acc0 = f64x2_add(acc0, f64x2_mul(w0, p0));
    acc1 = f64x2_add(acc1, f64x2_mul(w1, p1));
  }

  // Reduce two 2-lane accumulators to a scalar.
  let acc = f64x2_add(acc0, acc1);
  let mut total = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);

  // Tail: 0..3 leftover elements. Falls through the scalar formula.
  let tail_start = n_chunks * 4;
  for i in tail_start..n {
    total += weights[i] * power[i];
  }

  total
}

/// simd128 `first_non_finite`. Linear scan for the first NaN or ±∞
/// sample using a per-chunk bit-mask test of the IEEE 754 exponent
/// field.
///
/// # IEEE 754 bit-mask trick
///
/// An f32 is non-finite (NaN or ±∞) iff its biased 8-bit exponent is
/// `0xFF`. Masking with `0x7F800000` isolates the exponent bits;
/// equality with the mask classifies the lane as non-finite. See the
/// AVX2 backend for a longer write-up.
///
/// # Strategy
///
/// Body processes 8 elements per iteration (two `v128` of 4 f32 each)
/// for instruction-level parallelism. `i32x4_eq` produces per-lane
/// all-ones / all-zeros masks; `v128_or` combines them; `v128_any_true`
/// is a single-instruction "any bit set?" reduction. On a hit, fall
/// back to scalar within the chunk to find the exact first index.
///
/// # Safety
///
/// Same simd128 availability contract as [`power_spectrum_into`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn first_non_finite(samples: &[f32]) -> Option<usize> {
  let n = samples.len();
  let ptr = samples.as_ptr();

  // Exponent-bits mask: an f32's biased exponent occupies bits 23..30,
  // so masking with `0x7F800000` isolates them. A non-finite value
  // (NaN or ±∞) has every exponent bit set.
  let exp_mask = u32x4_splat(0x7F800000);

  // Process 8 elements per iteration (two v128 of 4 f32 each).
  let n_chunks = n / 8;
  for chunk in 0..n_chunks {
    let base = chunk * 8;
    // SAFETY: `chunk < n_chunks = n / 8` ⇒ `base + 7 < n`, so the two
    // 16-byte (4-f32) loads sit fully inside the slice. `v128_load`
    // is unaligned-tolerant.
    let v0 = unsafe { v128_load(ptr.add(base) as *const v128) };
    let v1 = unsafe { v128_load(ptr.add(base + 4) as *const v128) };
    // Isolate the exponent field of each lane; v128 has no type, so
    // the same `v128_and` works for the f32-as-bits AND.
    let e0 = v128_and(v0, exp_mask);
    let e1 = v128_and(v1, exp_mask);
    // Per-lane mask: `0xFFFFFFFF` if the exponent is all-ones (lane
    // is non-finite), `0` otherwise.
    let c0 = i32x4_eq(e0, exp_mask);
    let c1 = i32x4_eq(e1, exp_mask);
    // OR the two 4-lane masks; any set bit means at least one of the
    // 8 lanes was non-finite. `v128_any_true` reduces across all 128
    // bits to a single boolean.
    let combined = v128_or(c0, c1);
    if v128_any_true(combined) {
      // Hit somewhere in this 8-element chunk; fall back to scalar
      // to find the exact first index.
      for i in base..base + 8 {
        if !samples[i].is_finite() {
          return Some(i);
        }
      }
    }
  }

  // Tail: 0..7 leftover elements that didn't fit a full 8-lane chunk.
  let tail_start = n_chunks * 8;
  for i in tail_start..n {
    if !samples[i].is_finite() {
      return Some(i);
    }
  }
  None
}
