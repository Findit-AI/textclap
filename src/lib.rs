//! textclap — CLAP (Contrastive Language-Audio Pre-training) inference library.
//!
//! See `docs/superpowers/specs/` for the full design spec.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod audio;
mod clap_model;            // not `mod clap` to avoid shadowing the `clap` CLI crate if a future
                           // dev-dep pulls it in
mod error;
mod mel;
mod options;
mod text;

pub use crate::audio::AudioEncoder;
pub use crate::clap_model::{Clap, Embedding, LabeledScore, LabeledScoreOwned};
pub use crate::error::{Error, Result};
pub use crate::options::{ChunkingOptions, GraphOptimizationLevel, Options};
pub use crate::text::TextEncoder;
