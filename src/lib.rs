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
mod simd;
mod text;

pub use crate::{
  audio::AudioEncoder,
  clap::{Clap, Embedding, LabeledScore, LabeledScoreOwned},
  error::{Error, Result},
  options::{ChunkingOptions, GraphOptimizationLevel, Options},
  text::TextEncoder,
};

/// Bytes of the pinned Xenova `tokenizer.json` shipped with the crate (~2 MB).
/// SHA256 verified in CI against `models/MODELS.sha256`. Exposed for callers who
/// want to construct the encoders via `from_memory`; for typical use, prefer the
/// bundled-tokenizer constructors `Clap::from_onnx_files` and
/// `TextEncoder::from_onnx_file`.
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");
