//! textclap — CLAP (Contrastive Language-Audio Pre-training) inference library.
//!
//! See `docs/superpowers/specs/` for the full design spec.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod audio;
mod clap;
mod error;
mod mel;
mod options;
mod text;

pub use crate::{
  audio::AudioEncoder,
  clap::{Clap, Embedding, LabeledScore, LabeledScoreOwned},
  error::{Error, Result},
  options::{ChunkingOptions, GraphOptimizationLevel, Options},
  text::TextEncoder,
};
