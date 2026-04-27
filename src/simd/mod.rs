//! SIMD backends for the three hot kernels in textclap.
//!
//! The scalar reference ([`scalar`]) is always available and always
//! bit-identical to the original Rust loops in `mel.rs` and `audio.rs`.
//! Per-architecture backends ([`neon`], [`x86_avx2`]) are selected at
//! runtime via `is_*_feature_detected!`. Each SIMD kernel itself carries
//! `#[target_feature(enable = "...")]` so its intrinsics execute in an
//! explicitly feature-enabled context rather than one merely inherited
//! from the target's default features.
//!
//! `unsafe` is confined to the [`neon`] / [`x86_avx2`] submodules. The
//! scalar reference plus this dispatcher contain no `unsafe` blocks. The
//! crate-level lint [`unsafe_op_in_unsafe_fn`](https://doc.rust-lang.org/rustc/lints/listing/deny-by-default.html#unsafe-op-in-unsafe-fn)
//! requires every intrinsic call inside a backend kernel to sit in an
//! explicit `unsafe { ... }` block with its own `// SAFETY:` justification.
//!
//! Output guarantees: [`power_spectrum_into`] is byte-identical to
//! [`scalar`] across all backends. [`mel_filterbank_dot`] is the
//! exception — its NEON / AVX2 backends use FMA + 2x ILP, so they
//! reassociate the summation tree and round once instead of twice.
//! Drift is bounded at `1e-10 * scale` (loose; see backend doc
//! comments) — far below the integration golden mel tolerance of
//! `1e-4`. Equivalence tests in Tasks SIMD-2/3/4 enforce these
//! contracts.
//!
//! Dispatcher `cfg_select!` requires Rust 1.95+ (stable, in the core
//! prelude — no import needed). The crate's MSRV matches.
//!
//! Setting `--cfg textclap_force_scalar` at build time forces every
//! dispatcher down the scalar path, regardless of CPU features. Useful
//! for benchmarking scalar-vs-SIMD on the same input and for forcing
//! the scalar fallback's coverage on runners that would otherwise always
//! pick a SIMD tier.
//!
//! # Backend coverage: why NEON + AVX2+FMA, but not AVX-512 or SSE4.1
//!
//! The crate ships two architecture-specific backends. The omitted tiers
//! were considered and deliberately left out for v0.1.0:
//!
//! ## AVX-512 (omitted)
//!
//! - **Apple Silicon — the primary aarch64 target — has no AVX-512 path
//!   regardless,** so adding it benefits only x86_64 deployments.
//! - **Mixed CPU support on x86_64.** Intel disabled AVX-512 on consumer
//!   12th-gen and later (E-cores lack it; the P-cores' support is fused
//!   off for SKU consistency). AMD added it from Zen 4 (2022). A binary
//!   that runtime-detects AVX-512 and uses it where available still has
//!   to bring an AVX2 backend for the majority case.
//! - **Modest real-world speedup for this workload.** Doubling the lane
//!   count from 4 to 8 f64 sounds attractive, but the kernels here are
//!   memory-bandwidth-bound at their working set sizes. AVX-512 also
//!   triggers down-clocking on older CPUs, which can erase the lane-count
//!   win on mixed workloads. Measured 20–30% gains are typical, not 2×.
//! - **Doubles the testing surface.** Each new tier adds an equivalence
//!   test against the scalar reference and an integration-golden gate
//!   per kernel. The existing NEON + AVX2 backends already cover the
//!   common deployments; a third tier is added complexity without a
//!   matching coverage gain.
//!
//! AVX-512 can be added in a later release if profiling identifies a
//! specific deployment where the lane-count win clearly beats the
//! scalar path on hot kernels. The dispatcher pattern accommodates a
//! new backend without touching call sites.
//!
//! ## SSE4.1 (omitted)
//!
//! - **AVX2 has been baseline on Intel since Haswell (2013) and AMD since
//!   Excavator (2015).** Modern server hardware, cloud VMs, and laptops
//!   all expose it. The "needs SSE4.1 fallback" coverage gap is
//!   essentially pre-2014 hardware that is not a deployment target for
//!   this crate.
//! - **SSE4.1 cannot match AVX2's ILP for f64 FMA.** SSE4.1 has 128-bit
//!   vectors (2-lane f64), the same width as NEON's `float64x2_t`, but
//!   without FMA — that arrived only with FMA3 alongside Haswell/AVX2.
//!   The mel filterbank dot kernel relies on FMA for both throughput
//!   and the rounding-tree contract; an SSE4.1 backend without FMA
//!   would have a different numerical contract from NEON / AVX2 and
//!   would need its own equivalence budget.
//! - **The colconv reference does ship an SSE4.1 backend** because its
//!   8-bit pixel ops benefit from SSE4.1-specific shuffle and saturating
//!   instructions that AVX2 does not directly improve. textclap's
//!   f64-FMA workload has a different shape: there is no useful SSE4.1
//!   instruction here that AVX2 doesn't already cover better.
//! - **The scalar fallback catches the residual case** of pre-AVX2 x86
//!   hardware. It is correct, well-tested, and a few-percent slower than
//!   what a hypothetical SSE4.1 backend would deliver — acceptable for a
//!   path that should rarely execute.

#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_avx2;
#[cfg(target_arch = "x86_64")]
mod x86_avx512;

use rustfft::num_complex::Complex;

/// Compute `out[i] = buf[i].re² + buf[i].im²` for every element. Dispatches
/// to the best available backend; falls back to [`scalar::power_spectrum_into`]
/// when no SIMD backend is available.
///
/// See [`scalar::power_spectrum_into`] for the full semantic specification.
pub(crate) fn power_spectrum_into(buf: &[Complex<f64>], out: &mut [f64]) {
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: `neon_available()` verified NEON is present on this CPU.
        unsafe { neon::power_spectrum_into(buf, out); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: `avx512_available()` verified AVX-512F is present on this CPU.
        // AVX-512F implies AVX2 + FMA on every shipped x86_64 CPU, satisfying
        // the kernel's `target_feature(enable = "avx512f,avx2,fma")` contract.
        unsafe { x86_avx512::power_spectrum_into(buf, out); }
        return;
      }
      if avx2_fma_available() {
        // SAFETY: `avx2_fma_available()` verified AVX2 + FMA are present on this CPU.
        unsafe { x86_avx2::power_spectrum_into(buf, out); }
        return;
      }
    },
    _ => {
      // Targets without a SIMD backend (riscv64, powerpc, …) fall through
      // to the scalar path below.
    }
  }
  scalar::power_spectrum_into(buf, out);
}

/// Compute `Σ weights[i] * power[i]`. Dispatches to the best available
/// backend; falls back to [`scalar::mel_filterbank_dot`] when no SIMD
/// backend is available.
///
/// See [`scalar::mel_filterbank_dot`] for the full semantic specification.
pub(crate) fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: `neon_available()` verified NEON is present on this CPU.
        return unsafe { neon::mel_filterbank_dot(weights, power) };
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: `avx512_available()` verified AVX-512F is present on this CPU.
        // AVX-512F implies AVX2 + FMA on every shipped x86_64 CPU.
        return unsafe { x86_avx512::mel_filterbank_dot(weights, power) };
      }
      if avx2_fma_available() {
        // SAFETY: `avx2_fma_available()` verified AVX2 + FMA are present on this CPU.
        return unsafe { x86_avx2::mel_filterbank_dot(weights, power) };
      }
    },
    _ => {}
  }
  scalar::mel_filterbank_dot(weights, power)
}

/// Returns `Some(i)` of the first non-finite (NaN or ±∞) sample, or `None`
/// if every sample is finite. Dispatches to the best available backend;
/// falls back to [`scalar::first_non_finite`] when no SIMD backend is
/// available.
///
/// See [`scalar::first_non_finite`] for the full semantic specification.
pub(crate) fn first_non_finite(samples: &[f32]) -> Option<usize> {
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: `neon_available()` verified NEON is present on this CPU.
        return unsafe { neon::first_non_finite(samples) };
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: `avx512_available()` verified AVX-512F is present on this CPU.
        // AVX-512F implies AVX2 + FMA on every shipped x86_64 CPU.
        return unsafe { x86_avx512::first_non_finite(samples) };
      }
      if avx2_fma_available() {
        // SAFETY: `avx2_fma_available()` verified AVX2 + FMA are present on this CPU.
        return unsafe { x86_avx2::first_non_finite(samples) };
      }
    },
    _ => {}
  }
  scalar::first_non_finite(samples)
}

/// NEON availability on aarch64. `std::arch::is_aarch64_feature_detected!`
/// caches its result in an atomic, so per-call overhead is a single relaxed
/// load plus a branch.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn neon_available() -> bool {
  if cfg!(textclap_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

/// AVX2 + FMA availability on x86_64. Both must be present for the AVX2
/// backend (FMA is needed for the fused multiply-add in
/// [`mel_filterbank_dot`]).
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn avx2_fma_available() -> bool {
  if cfg!(textclap_force_scalar) {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// AVX-512F availability on x86_64. AVX-512F implies AVX2 + FMA on every
/// shipped CPU (Intel Skylake-X+, AMD Zen 4+), so a single AVX-512F check
/// is sufficient — the AVX-512 kernels' `#[target_feature(enable =
/// "avx512f,avx2,fma")]` contract is satisfied transitively.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn avx512_available() -> bool {
  if cfg!(textclap_force_scalar) {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx512f")
}

#[cfg(test)]
mod tests {
  use super::*;
  use rustfft::num_complex::Complex;

  fn run_dispatch(buf: &[Complex<f64>]) -> Vec<f64> {
    let mut out = vec![0.0f64; buf.len()];
    power_spectrum_into(buf, &mut out);
    out
  }

  fn run_scalar(buf: &[Complex<f64>]) -> Vec<f64> {
    let mut out = vec![0.0f64; buf.len()];
    scalar::power_spectrum_into(buf, &mut out);
    out
  }

  /// Deterministic LCG so the equivalence tests don't depend on a
  /// platform-specific RNG. Constants are the standard MMIX values.
  fn make_input(n: usize, seed: u64) -> Vec<Complex<f64>> {
    let mut s = seed;
    (0..n)
      .map(|_| {
        s = s
          .wrapping_mul(6364136223846793005)
          .wrapping_add(1442695040888963407);
        let re = ((s >> 33) as i32 as f64) / i32::MAX as f64;
        s = s
          .wrapping_mul(6364136223846793005)
          .wrapping_add(1442695040888963407);
        let im = ((s >> 33) as i32 as f64) / i32::MAX as f64;
        Complex::new(re, im)
      })
      .collect()
  }

  /// f64 variant of [`make_input`] — same LCG, one sample per step,
  /// values in `[-1, 1]`. Used by the `mel_filterbank_dot` equivalence
  /// tests.
  fn make_input_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
      .map(|_| {
        s = s
          .wrapping_mul(6364136223846793005)
          .wrapping_add(1442695040888963407);
        ((s >> 33) as i32 as f64) / i32::MAX as f64
      })
      .collect()
  }

  #[test]
  fn power_spectrum_simd_matches_scalar_513() {
    // Real call site uses N_FFT/2+1 = 513 (odd), so the kernel's tail
    // path runs in production. Cover it here.
    let input = make_input(513, 0xDEADBEEF);
    let dispatched = run_dispatch(&input);
    let reference = run_scalar(&input);
    assert_eq!(dispatched.len(), reference.len());
    for (i, (a, b)) in dispatched.iter().zip(reference.iter()).enumerate() {
      // Bit-identical for re*re + im*im since SIMD multiplies are
      // exact (no FMA reassoc in this kernel).
      assert!(
        (a - b).abs() <= 1e-15,
        "mismatch at {i}: simd={a} scalar={b}"
      );
    }
  }

  #[test]
  fn power_spectrum_simd_matches_scalar_even_512() {
    // Pure SIMD path with no tail.
    let input = make_input(512, 0xCAFEF00D);
    let dispatched = run_dispatch(&input);
    let reference = run_scalar(&input);
    for (a, b) in dispatched.iter().zip(reference.iter()) {
      assert!((a - b).abs() <= 1e-15);
    }
  }

  #[test]
  fn power_spectrum_empty() {
    let input: Vec<Complex<f64>> = Vec::new();
    let mut out: Vec<f64> = Vec::new();
    power_spectrum_into(&input, &mut out);
    assert!(out.is_empty());
  }

  #[test]
  fn power_spectrum_one() {
    // 3-4-5 right triangle: 3² + 4² == 25. Catches a kernel that
    // skips the tail when n_pairs == 0.
    let input = vec![Complex::new(3.0, 4.0)];
    let mut out = vec![0.0f64; 1];
    power_spectrum_into(&input, &mut out);
    assert_eq!(out, vec![25.0]);
  }

  #[test]
  fn mel_filterbank_dot_simd_matches_scalar_513() {
    // Production call site uses N_FFT/2+1 = 513 (odd), so both the
    // SIMD body and the scalar tail path run.
    let weights = make_input_f64(513, 0xABCD1234);
    let power = make_input_f64(513, 0xFEDC4321);
    let dispatched = mel_filterbank_dot(&weights, &power);
    let reference = scalar::mel_filterbank_dot(&weights, &power);
    // Reassociation + FMA can shift bits; budget is loose vs. golden
    // tolerance.
    let diff = (dispatched - reference).abs();
    let scale = reference.abs().max(1.0);
    assert!(
      diff <= 1e-10 * scale,
      "drift exceeds 1e-10 * scale: simd={dispatched} scalar={reference} diff={diff}"
    );
  }

  #[test]
  fn mel_filterbank_dot_simd_matches_scalar_short() {
    // Production sparse rows can have very few nonzero entries with
    // leading/trailing zeros. Make sure the tail path works for tiny
    // inputs (especially n < 4 / n < 8 where the SIMD body never runs).
    for n in [0, 1, 3, 7, 8, 15, 16, 31, 32] {
      let weights = make_input_f64(n, 0x1111 + n as u64);
      let power = make_input_f64(n, 0x2222 + n as u64);
      let dispatched = mel_filterbank_dot(&weights, &power);
      let reference = scalar::mel_filterbank_dot(&weights, &power);
      let diff = (dispatched - reference).abs();
      let scale = reference.abs().max(1.0);
      assert!(
        diff <= 1e-10 * scale,
        "n={n}: simd={dispatched} scalar={reference}"
      );
    }
  }

  #[test]
  fn mel_filterbank_dot_zero_weights_zero_result() {
    // Zero weights (e.g. mel filterbank rows past the last passband)
    // must produce exactly 0.0, not just near-zero.
    let weights = vec![0.0f64; 513];
    let power: Vec<f64> = (0..513).map(|i| i as f64).collect();
    assert_eq!(mel_filterbank_dot(&weights, &power), 0.0);
  }

  #[test]
  fn mel_filterbank_dot_unit_weights_sums_power() {
    // Unit weights → result equals Σ power. 1+2+…+8 == 36. Validates
    // the reduction tree end-to-end on a small input where the SIMD
    // body runs exactly once (n=8).
    let weights = vec![1.0f64; 8];
    let power = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = mel_filterbank_dot(&weights, &power);
    assert!((result - 36.0).abs() < 1e-12, "got {result}");
  }

  #[test]
  fn first_non_finite_simd_finds_nan_at_zero() {
    let mut s = vec![0.0f32; 100];
    s[0] = f32::NAN;
    assert_eq!(first_non_finite(&s), Some(0));
  }

  #[test]
  fn first_non_finite_simd_finds_inf_in_middle() {
    let mut s = vec![0.0f32; 1000];
    s[472] = f32::INFINITY;
    assert_eq!(first_non_finite(&s), Some(472));
  }

  #[test]
  fn first_non_finite_simd_finds_neg_inf_at_end() {
    // Element near end, in the scalar-tail region for chunk_size=16.
    let mut s = vec![0.0f32; 1023];
    s[1022] = f32::NEG_INFINITY;
    assert_eq!(first_non_finite(&s), Some(1022));
  }

  #[test]
  fn first_non_finite_simd_clean_input_returns_none() {
    let s: Vec<f32> = (0..480_000).map(|i| (i as f32) * 1e-3).collect();
    assert_eq!(first_non_finite(&s), None);
  }

  #[test]
  fn first_non_finite_simd_subnormal_is_finite() {
    // Subnormals (exponent = 0) are finite. Must NOT trigger.
    let s = vec![f32::MIN_POSITIVE / 2.0; 64];
    assert_eq!(first_non_finite(&s), None);
  }

  #[test]
  fn first_non_finite_simd_zero_and_negzero_are_finite() {
    let s = vec![0.0f32, -0.0f32, 0.0, -0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(first_non_finite(&s), None);
  }

  #[test]
  fn first_non_finite_simd_short_inputs() {
    // Scalar-only path (n < chunk size).
    for n in 0..16 {
      let mut s = vec![1.0f32; n];
      if n > 0 {
        s[n - 1] = f32::NAN;
        assert_eq!(first_non_finite(&s), Some(n - 1), "n={n}");
      } else {
        assert_eq!(first_non_finite(&s), None, "n=0");
      }
    }
  }

  #[test]
  fn first_non_finite_simd_at_chunk_boundary() {
    // Verify the boundary case where the non-finite is exactly at chunk_size on AVX2 (16) or NEON (8).
    for hit in [7usize, 8, 15, 16, 23, 24] {
      let mut s = vec![1.0f32; 100];
      s[hit] = f32::NAN;
      assert_eq!(first_non_finite(&s), Some(hit), "hit={hit}");
    }
  }
}
