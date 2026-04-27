//! aarch64 NEON backend for the three hot kernels.
//!
//! Selected by [`crate::simd`]'s dispatcher after
//! `is_aarch64_feature_detected!("neon")` returns true. Each kernel
//! carries `#[target_feature(enable = "neon")]` so its intrinsics execute
//! in an explicitly NEON-enabled context rather than one merely inherited
//! from the aarch64 target's default feature set. Same `#[target_feature]`
//! placement and `#[inline]` policy as `colconv::row::arch::neon`.
//!
//! `first_non_finite` remains a placeholder forwarder to
//! [`super::scalar`] until Task SIMD-4 fills in its intrinsics.

use core::arch::aarch64::*;
use rustfft::num_complex::Complex;

use super::scalar;

/// NEON `power_spectrum_into`. Computes `out[i] = buf[i].re² + buf[i].im²`
/// using `vld2q_f64` to deinterleave 2 `Complex<f64>` per iteration into
/// real- and imag-lane vectors, then `vmulq_f64` + `vfmaq_f64` to form
/// `re*re + im*im`. The odd-tail element (if any) falls back to the
/// scalar reference.
///
/// Output is bit-identical to [`scalar::power_spectrum_into`]: this
/// kernel does **not** use FMA — `re*re` is a separate multiply, not a
/// fused `fma(re, re, im*im)`, so no operand sees an extra round.
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
  debug_assert_eq!(buf.len(), out.len());
  let n = buf.len();
  let n_pairs = n / 2;

  // `Complex<f64>` is `#[repr(C)]` with `[re, im]` layout, so a
  // `&[Complex<f64>]` of length `n` aliases a contiguous f64 stream
  // `[r0, i0, r1, i1, …, r_{n-1}, i_{n-1}]` of length `2 * n`. This
  // pointer reinterpret is sound: alignment (`align_of::<f64>() == 8`)
  // matches and the lifetime ends at the end of this fn.
  let buf_ptr = buf.as_ptr() as *const f64;
  let out_ptr = out.as_mut_ptr();

  for i in 0..n_pairs {
    // SAFETY: `i < n_pairs = n / 2` ⇒ `i*4 + 3 < 2*n`, so `vld2q_f64`'s
    // 4 contiguous f64 reads sit inside the `2*n`-long f64 view of
    // `buf`. NEON loads are unaligned-tolerant; the slice is f64-aligned.
    let pair = unsafe { vld2q_f64(buf_ptr.add(i * 4)) };
    // `vld2q_f64` deinterleaves: `pair.0 = [re_{2i}, re_{2i+1}]`,
    // `pair.1 = [im_{2i}, im_{2i+1}]`.
    let re = pair.0;
    let im = pair.1;
    // |c|² = re*re + im*im, lane-wise. The kernel avoids `vfmaq_f64`
    // here on purpose: a fused multiply-add would skip the rounding of
    // the `im*im` product and silently produce results that differ from
    // the scalar reference's `re*re + im*im` (3 rounds: two muls + one
    // add). Two `vmulq_f64`s plus a `vaddq_f64` reproduce the scalar
    // round structure exactly, so the kernel is bit-identical to
    // `super::scalar::power_spectrum_into` rather than merely close.
    let re_sq = vmulq_f64(re, re);
    let im_sq = vmulq_f64(im, im);
    let pwr = vaddq_f64(re_sq, im_sq);
    // SAFETY: `i*2 + 1 < n` (since `i < n/2`) and `out.len() == n`,
    // so writing 2 f64 starting at `out_ptr.add(i*2)` is in-bounds.
    unsafe { vst1q_f64(out_ptr.add(i * 2), pwr) };
  }

  // Tail: 1 element if `n` is odd. The body matches the scalar inner
  // loop exactly so the byte-equivalence contract holds.
  if n & 1 == 1 {
    let last = n - 1;
    let c = buf[last];
    out[last] = c.re * c.re + c.im * c.im;
  }
}

/// NEON `mel_filterbank_dot`. Computes `Σ weights[i] * power[i]` using two
/// independent `float64x2_t` accumulators (4 elements per iteration) and
/// `vfmaq_f64` for fused multiply-add. The two-accumulator pattern lets the
/// CPU dispatch dependent FMAs in parallel — each fmadd has ~3-cycle
/// latency on Apple Silicon, so a single accumulator would stall on its
/// own dependency chain; two accumulators keep the FMA pipeline full
/// (one issue per cycle).
///
/// # Numerical contract
///
/// Output is **not** bit-identical to [`scalar::mel_filterbank_dot`]:
///
/// 1. The two parallel accumulators reassociate the summation tree —
///    pairs of products are summed separately before being combined,
///    rather than the strictly left-fold `((0 + a₀b₀) + a₁b₁) + …` of
///    the scalar reference. Different operand orders → different f64
///    results, typically off by ~1 ULP per element.
/// 2. `vfmaq_f64` performs `acc + w*p` with **single rounding**, whereas
///    the scalar's `acc += w * p` rounds twice (once after the multiply,
///    once after the add). FMA generally tightens accuracy.
///
/// Both effects are bounded well below the equivalence-test budget of
/// `1e-10 * scale` and far below the integration golden mel tolerance of
/// `1e-4`. The crate-level "byte-identical" claim in [`super`] applies
/// to [`power_spectrum_into`] only; this kernel deliberately accepts ULP
/// drift to win the FMA + ILP throughput.
///
/// # Safety
///
/// Same NEON-availability contract as [`power_spectrum_into`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
  debug_assert_eq!(weights.len(), power.len());
  let n = weights.len();

  // Two independent accumulators allow the CPU to issue dependent FMAs
  // in parallel: each `vfmaq_f64` has ~3-cycle latency on Apple Silicon
  // but the FMA pipe accepts one issue per cycle.
  let mut acc0 = vdupq_n_f64(0.0);
  let mut acc1 = vdupq_n_f64(0.0);

  let w_ptr = weights.as_ptr();
  let p_ptr = power.as_ptr();

  // Process 4 elements per iteration (two 2-lane f64 vectors).
  let n_chunks = n / 4;
  for i in 0..n_chunks {
    // SAFETY: `i < n_chunks = n / 4` ⇒ `i*4 + 3 < n`, so two
    // `vld1q_f64` loads of 2 f64 each (4 f64 total) sit fully inside
    // the slice. Slices are f64-aligned (8 bytes); NEON loads are
    // unaligned-tolerant.
    let w0 = unsafe { vld1q_f64(w_ptr.add(i * 4)) };
    let w1 = unsafe { vld1q_f64(w_ptr.add(i * 4 + 2)) };
    let p0 = unsafe { vld1q_f64(p_ptr.add(i * 4)) };
    let p1 = unsafe { vld1q_f64(p_ptr.add(i * 4 + 2)) };
    // Fused multiply-add: `acc + w*p` with a single round. Differs
    // from scalar's two-round `acc += w * p` — see the numerical
    // contract above.
    acc0 = vfmaq_f64(acc0, w0, p0);
    acc1 = vfmaq_f64(acc1, w1, p1);
  }

  // Reduce the two 2-lane accumulators to a scalar. `vaddvq_f64` is
  // ARMv8's single-instruction horizontal sum (lane 0 + lane 1).
  let acc = vaddq_f64(acc0, acc1);
  let mut total = vaddvq_f64(acc);

  // Tail: 0..3 leftover elements. Falls through the scalar formula so
  // it stays in the same FMA-vs-mul-add round structure as the body.
  let tail_start = n_chunks * 4;
  for i in tail_start..n {
    total += weights[i] * power[i];
  }

  total
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
