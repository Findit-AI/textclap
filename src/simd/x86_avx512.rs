//! x86_64 AVX-512 (F + FMA) backend.
//!
//! Selected by [`crate::simd`]'s dispatcher when
//! `is_x86_feature_detected!("avx512f")` returns true. AVX-512F implies
//! AVX2 + FMA on every shipped CPU; the kernel carries
//! `#[target_feature(enable = "avx512f,avx2,fma")]` to make 256-bit
//! reduction ops accessible inside the function body.
//!
//! # Numerical contract
//!
//! - [`power_spectrum_into`]: bit-identical to [`super::scalar`] —
//!   plain mul + horizontal pair-add, no FMA reassociation.
//! - [`mel_filterbank_dot`]: FMA + 2x ILP, same drift budget as the
//!   NEON / AVX2 backends (≤ `1e-10 * scale` for any reasonable input).
//! - [`first_non_finite`]: structural equivalence — finds the same
//!   index as the scalar reference.
//!
//! # Pipeline width
//!
//! - power_spectrum: 4 complex per iteration (`__m512d`, 8 f64 lanes,
//!   horizontal-pair-add via 256-bit halves)
//! - mel_filterbank_dot: 16 elements per iteration (two `__m512d`
//!   accumulators × 8 lanes), reduced with `_mm512_reduce_add_pd`
//! - first_non_finite: 32 f32 per iteration (two `__m512` vectors)
//!   with mask-register reductions

use core::arch::x86_64::*;
use rustfft::num_complex::Complex;

/// AVX-512 `power_spectrum_into`. Computes `out[i] = buf[i].re² + buf[i].im²`
/// using a 512-bit unaligned load of 4 `Complex<f64>` (= 8 f64
/// `[r0, i0, r1, i1, r2, i2, r3, i3]`) per iteration. Squares lane-wise,
/// then splits into two 256-bit halves and pair-sums via
/// `_mm256_hadd_pd`, restoring natural lane order with a
/// `_mm256_permute4x64_pd::<0xD8>` fixup. Output is bit-identical to
/// [`super::scalar::power_spectrum_into`]: plain mul + add, no FMA.
///
/// # Safety
///
/// The caller must ensure AVX-512F is available on the current CPU.
/// The dispatcher in [`crate::simd`] verifies this with
/// `is_x86_feature_detected!("avx512f")`. AVX-512F implies AVX2 + FMA
/// on every shipped CPU. Calling this kernel on a CPU without those
/// features is undefined behavior.
#[inline]
#[target_feature(enable = "avx512f,avx2,fma")]
pub(crate) unsafe fn power_spectrum_into(buf: &[Complex<f64>], out: &mut [f64]) {
  debug_assert_eq!(buf.len(), out.len());
  let n = buf.len();
  let n_quads = n / 4;

  // `Complex<f64>` is `#[repr(C)]` with `[re, im]` layout, so a
  // `&[Complex<f64>]` of length `n` aliases a contiguous f64 stream
  // `[r0, i0, r1, i1, …]` of length `2 * n`.
  let buf_ptr = buf.as_ptr() as *const f64;
  let out_ptr = out.as_mut_ptr();

  for i in 0..n_quads {
    // SAFETY: `i < n_quads = n / 4` ⇒ `i*8 + 7 < 2*n`, so the 8 f64
    // load sits inside the `2*n`-long f64 view of `buf`. Unaligned
    // loads tolerate the slice's f64 (8-byte) alignment.
    let v = unsafe { _mm512_loadu_pd(buf_ptr.add(i * 8)) };
    // Square element-wise: [r0², i0², r1², i1², r2², i2², r3², i3²].
    // No FMA — same round structure as the scalar reference.
    let sq = _mm512_mul_pd(v, v);
    // Split into 256-bit halves for the horizontal pair-add.
    // `_mm512_castpd512_pd256` keeps lanes 0..3 (free); the extract
    // lifts lanes 4..7 down.
    let lo = _mm512_castpd512_pd256(sq);
    // `_mm512_extractf64x4_pd::<1>` lifts lanes 4..7 down. The const
    // generic IMM8 = 1 is statically validated; the compiler accepts
    // this call as safe under the AVX-512F target feature.
    let hi = _mm512_extractf64x4_pd::<1>(sq);
    // `_mm256_hadd_pd(a, b)` has lane-pair semantics across 128-bit
    // halves: it returns
    //   `[a0+a1, b0+b1, a2+a3, b2+b3]`
    // which with `a = lo`, `b = hi` is
    //   `[r0²+i0², r2²+i2², r1²+i1², r3²+i3²]`.
    // We need natural order `[r0²+i0², r1²+i1², r2²+i2², r3²+i3²]`.
    // `_mm256_permute4x64_pd::<0xD8>` selects lanes `[0, 2, 1, 3]`
    // from its input (imm8 = 0b11_01_10_00) — exactly the swap that
    // turns the hadd output into natural order.
    let pairs = _mm256_hadd_pd(lo, hi);
    let result = _mm256_permute4x64_pd::<0xD8>(pairs);
    // SAFETY: `i*4 + 3 < n` (since `i < n/4`) and `out.len() == n`,
    // so writing 4 f64 starting at `out_ptr.add(i*4)` is in-bounds.
    unsafe { _mm256_storeu_pd(out_ptr.add(i * 4), result) };
  }

  // Tail: 0..3 leftover elements. Body matches the scalar inner loop
  // exactly so the byte-equivalence contract holds.
  let tail_start = n_quads * 4;
  for i in tail_start..n {
    let c = buf[i];
    out[i] = c.re * c.re + c.im * c.im;
  }
}

/// AVX-512 `mel_filterbank_dot`. Computes `Σ weights[i] * power[i]` using
/// two independent `__m512d` accumulators (16 elements per iteration) and
/// `_mm512_fmadd_pd` for fused multiply-add. The two-accumulator pattern
/// keeps the FMA pipeline busy by giving each lane two independent
/// dependency chains.
///
/// # Numerical contract
///
/// Output is **not** bit-identical to [`super::scalar::mel_filterbank_dot`]
/// — same reasons (reassociation + FMA single-rounding) and the same
/// `1e-10 * scale` drift budget as the NEON / AVX2 backends.
///
/// # Safety
///
/// Same AVX-512F availability contract as [`power_spectrum_into`].
#[inline]
#[target_feature(enable = "avx512f,avx2,fma")]
pub(crate) unsafe fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
  debug_assert_eq!(weights.len(), power.len());
  let n = weights.len();

  let mut acc0 = _mm512_setzero_pd();
  let mut acc1 = _mm512_setzero_pd();

  let w_ptr = weights.as_ptr();
  let p_ptr = power.as_ptr();

  // Process 16 elements per iteration (two __m512d × 8 lanes).
  let n_chunks = n / 16;
  for i in 0..n_chunks {
    // SAFETY: `i < n_chunks = n / 16` ⇒ `i*16 + 15 < n`, so the four
    // 8-lane loads (two for `weights`, two for `power`) sit fully
    // inside their respective slices. Unaligned loads tolerate f64
    // (8-byte) alignment.
    let w0 = unsafe { _mm512_loadu_pd(w_ptr.add(i * 16)) };
    let w1 = unsafe { _mm512_loadu_pd(w_ptr.add(i * 16 + 8)) };
    let p0 = unsafe { _mm512_loadu_pd(p_ptr.add(i * 16)) };
    let p1 = unsafe { _mm512_loadu_pd(p_ptr.add(i * 16 + 8)) };
    // Fused multiply-add: `acc + w*p` with a single round. Differs
    // from scalar's two-round `acc += w * p` — see the numerical
    // contract above.
    acc0 = _mm512_fmadd_pd(w0, p0, acc0);
    acc1 = _mm512_fmadd_pd(w1, p1, acc1);
  }

  // Reduce the two 8-lane accumulators to a scalar.
  // 1) Combine accumulators lane-wise.
  // 2) `_mm512_reduce_add_pd` is the single-instruction horizontal sum
  //    for an `__m512d` (lane 0 + lane 1 + … + lane 7).
  let acc = _mm512_add_pd(acc0, acc1);
  let mut total = _mm512_reduce_add_pd(acc);

  // Tail: 0..15 leftover elements. Falls through the scalar formula.
  let tail_start = n_chunks * 16;
  for i in tail_start..n {
    total += weights[i] * power[i];
  }

  total
}

/// AVX-512 `first_non_finite`. Linear scan for the first NaN or ±∞
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
/// Body processes 32 elements per iteration (two `__m512` vectors of
/// 16 f32 each) for instruction-level parallelism. Each comparison
/// produces a `__mmask16`; OR-ing the two masks and testing for
/// non-zero is a single integer op on a mask register. On a hit, fall
/// back to scalar within the chunk to find the exact first index.
///
/// # Safety
///
/// Same AVX-512F availability contract as [`power_spectrum_into`].
#[inline]
#[target_feature(enable = "avx512f,avx2,fma")]
pub(crate) unsafe fn first_non_finite(samples: &[f32]) -> Option<usize> {
  let n = samples.len();
  let ptr = samples.as_ptr();

  // Exponent-bits mask: an f32's biased exponent occupies bits 23..30,
  // so masking with `0x7F800000` isolates them. A non-finite value
  // (NaN or ±∞) has every exponent bit set.
  let exp_mask_i = _mm512_set1_epi32(0x7F800000_u32 as i32);

  // Process 32 elements per iteration (two __m512 of 16 f32 each).
  let n_chunks = n / 32;
  for chunk in 0..n_chunks {
    let base = chunk * 32;
    // SAFETY: `chunk < n_chunks = n / 32` ⇒ `base + 31 < n`, so the
    // two 16-lane loads sit fully inside the slice. Unaligned loads
    // tolerate the slice's f32 (4-byte) alignment.
    let v0 = unsafe { _mm512_loadu_ps(ptr.add(base)) };
    let v1 = unsafe { _mm512_loadu_ps(ptr.add(base + 16)) };
    // Reinterpret the f32 lanes as i32 to manipulate the bit pattern.
    let b0 = _mm512_castps_si512(v0);
    let b1 = _mm512_castps_si512(v1);
    // Isolate the exponent field of each lane.
    let e0 = _mm512_and_epi32(b0, exp_mask_i);
    let e1 = _mm512_and_epi32(b1, exp_mask_i);
    // `_mm512_cmpeq_epi32_mask` returns a 16-bit mask: bit `k` is set
    // iff lane `k` of the two operands is equal — i.e., that lane is
    // non-finite.
    let m0: __mmask16 = _mm512_cmpeq_epi32_mask(e0, exp_mask_i);
    let m1: __mmask16 = _mm512_cmpeq_epi32_mask(e1, exp_mask_i);
    // OR the two 16-bit masks; any set bit means at least one of the
    // 32 lanes was non-finite. Mask-register OR is a single integer op.
    if (m0 | m1) != 0 {
      // Hit somewhere in this 32-element chunk; fall back to scalar
      // to find the exact first index.
      #[allow(clippy::needless_range_loop)]
      for i in base..base + 32 {
        if !samples[i].is_finite() {
          return Some(i);
        }
      }
    }
  }

  // Tail: 0..31 leftover elements that didn't fit a full 32-lane chunk.
  let tail_start = n_chunks * 32;
  (tail_start..n).find(|&i| !samples[i].is_finite())
}
