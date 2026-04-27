//! Configuration options for textclap. See spec §7.7 (Options) and §7.8 (ChunkingOptions).
//!
//! Sibling-convention notes:
//! - No `with_intra_threads` knob — thread tuning is configured outside textclap via
//!   `from_ort_session` (matches silero `options.rs:128–145`).
//! - `Options` follows soundevents' unqualified naming over silero's `SessionOptions` because
//!   textclap has no second options type to disambiguate.
//! - All getters / builders / setters are `pub const fn` (matches silero pattern).

pub use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "serde")]
mod graph_optimization_level {
  use super::GraphOptimizationLevel;
  use serde::*;

  #[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize,
  )]
  #[serde(rename_all = "snake_case")]
  enum OptimizationLevel {
    #[default]
    Disable,
    Level1,
    Level2,
    Level3,
    All,
  }

  impl From<GraphOptimizationLevel> for OptimizationLevel {
    #[inline]
    fn from(value: GraphOptimizationLevel) -> Self {
      match value {
        GraphOptimizationLevel::Disable => Self::Disable,
        GraphOptimizationLevel::Level1 => Self::Level1,
        GraphOptimizationLevel::Level2 => Self::Level2,
        GraphOptimizationLevel::Level3 => Self::Level3,
        GraphOptimizationLevel::All => Self::All,
      }
    }
  }

  impl From<OptimizationLevel> for GraphOptimizationLevel {
    #[inline]
    fn from(value: OptimizationLevel) -> Self {
      match value {
        OptimizationLevel::Disable => Self::Disable,
        OptimizationLevel::Level1 => Self::Level1,
        OptimizationLevel::Level2 => Self::Level2,
        OptimizationLevel::Level3 => Self::Level3,
        OptimizationLevel::All => Self::All,
      }
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn serialize<S>(level: &GraphOptimizationLevel, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    OptimizationLevel::from(*level).serialize(serializer)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn deserialize<'de, D>(deserializer: D) -> Result<GraphOptimizationLevel, D::Error>
  where
    D: Deserializer<'de>,
  {
    OptimizationLevel::deserialize(deserializer).map(Into::into)
  }

  /// Serde-defaults helper for `Options::optimization_level`.
  ///
  /// Returns `GraphOptimizationLevel::Level3` to match `Options::new()`,
  /// so deserializing an `Options` with `optimization_level` missing
  /// produces the same value as `Options::default()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn default() -> GraphOptimizationLevel {
    GraphOptimizationLevel::Level3
  }
}

/// Construction-time options for textclap encoders.
///
/// `Default` is implemented manually rather than derived because
/// `ort::session::builder::GraphOptimizationLevel` does not implement `Default` in
/// `ort 2.0.0-rc.12`. silero takes the same approach (`silero/src/options.rs:147`).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Options {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "graph_optimization_level::default",
      with = "graph_optimization_level"
    )
  )]
  optimization_level: GraphOptimizationLevel,
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl Options {
  /// Construct with default values (== `Self::default()`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      optimization_level: GraphOptimizationLevel::Level3,
    }
  }

  /// Set the ORT graph-optimization level (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.set_optimization_level(level);
    self
  }

  /// Set the ORT graph-optimization level (in-place setter).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self {
    self.optimization_level = level;
    self
  }

  /// Get the configured graph-optimization level.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_window_samples() -> usize {
  480_000
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_hop_samples() -> usize {
  480_000
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_batch_size() -> usize {
  8
}

/// Chunking-window configuration for `embed_chunked`.
///
/// In 0.1.0 the aggregation strategy is fixed (centroid or spherical-mean, chosen at construction
/// per §3.2). See spec §7.8.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkingOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_window_samples"))]
  window_samples: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_hop_samples"))]
  hop_samples: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_batch_size"))]
  batch_size: usize,
}

impl Default for ChunkingOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl ChunkingOptions {
  /// Construct with default values (window=480_000, hop=480_000, batch_size=8).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      window_samples: default_window_samples(),
      hop_samples: default_hop_samples(),
      batch_size: default_batch_size(),
    }
  }

  /// Set the window length in samples (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_window_samples(mut self, n: usize) -> Self {
    self.set_window_samples(n);
    self
  }

  /// Set the window length in samples (in-place setter).
  pub const fn set_window_samples(&mut self, n: usize) -> &mut Self {
    self.window_samples = n;
    self
  }

  /// Get the window length in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn window_samples(&self) -> usize {
    self.window_samples
  }

  /// Set the hop length in samples (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_hop_samples(mut self, n: usize) -> Self {
    self.set_hop_samples(n);
    self
  }

  /// Set the hop length in samples (in-place setter).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_hop_samples(&mut self, n: usize) -> &mut Self {
    self.hop_samples = n;
    self
  }

  /// Get the hop length in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hop_samples(&self) -> usize {
    self.hop_samples
  }

  /// Set the maximum batch size (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_batch_size(mut self, n: usize) -> Self {
    self.set_batch_size(n);
    self
  }

  /// Set the maximum batch size (in-place setter).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_batch_size(&mut self, n: usize) -> &mut Self {
    self.batch_size = n;
    self
  }

  /// Get the maximum batch size.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn batch_size(&self) -> usize {
    self.batch_size
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn options_default_matches_new() {
    let a = Options::new();
    let b = Options::default();
    assert_eq!(a.optimization_level(), b.optimization_level());
  }

  #[test]
  fn options_with_round_trips() {
    let opts = Options::new().with_optimization_level(GraphOptimizationLevel::Level1);
    assert_eq!(opts.optimization_level(), GraphOptimizationLevel::Level1);
  }

  #[test]
  fn options_set_round_trips() {
    let mut opts = Options::new();
    opts.set_optimization_level(GraphOptimizationLevel::Disable);
    assert_eq!(opts.optimization_level(), GraphOptimizationLevel::Disable);
  }

  #[test]
  fn chunking_default_values() {
    let c = ChunkingOptions::default();
    assert_eq!(c.window_samples(), 480_000);
    assert_eq!(c.hop_samples(), 480_000);
    assert_eq!(c.batch_size(), 8);
  }

  #[test]
  fn chunking_builders_round_trip() {
    let c = ChunkingOptions::new()
      .with_window_samples(240_000)
      .with_hop_samples(120_000)
      .with_batch_size(4);
    assert_eq!(c.window_samples(), 240_000);
    assert_eq!(c.hop_samples(), 120_000);
    assert_eq!(c.batch_size(), 4);
  }

  #[test]
  fn chunking_setters_round_trip() {
    let mut c = ChunkingOptions::new();
    c.set_window_samples(100)
      .set_hop_samples(50)
      .set_batch_size(1);
    assert_eq!(c.window_samples(), 100);
    assert_eq!(c.hop_samples(), 50);
    assert_eq!(c.batch_size(), 1);
  }
}
