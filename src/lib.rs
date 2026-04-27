//! textclap — CLAP (Contrastive Language-Audio Pre-training) inference library.
//!
//! See `docs/superpowers/specs/` for the full design spec.
//!
//! # `unsafe` policy
//!
//! The scalar reference is always available, and the rest of the crate is
//! `unsafe`-free by convention. SIMD backends in [`mod@simd`] use `unsafe`
//! exclusively for `core::arch::*` intrinsics, gated behind runtime CPU
//! feature detection (`is_*_feature_detected!`). The
//! [`unsafe_op_in_unsafe_fn`] lint requires every intrinsic call inside an
//! `unsafe fn` to sit in an explicit `unsafe { ... }` block with its own
//! `// SAFETY:` justification — no implicit `unsafe`-fn body inheritance.
//!
//! [`unsafe_op_in_unsafe_fn`]: https://doc.rust-lang.org/rustc/lints/listing/deny-by-default.html#unsafe-op-in-unsafe-fn

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod audio;
mod clap;
mod error;
mod mel;
mod options;
// Dead-code is permitted while the SIMD scaffold (Task SIMD-1) lands without
// call-site rewiring. Tasks SIMD-2 through SIMD-5 fill in the kernels and
// switch `mel.rs` / `audio.rs` over to the dispatchers, at which point the
// scalar reference, dispatchers, and per-arch backends all become live.
#[allow(dead_code)]
mod simd;
mod text;

pub use crate::{
  audio::AudioEncoder,
  clap::{Clap, Embedding, LabeledScore, LabeledScoreOwned},
  error::{Error, Result},
  options::{ChunkingOptions, GraphOptimizationLevel, Options},
  text::TextEncoder,
};
