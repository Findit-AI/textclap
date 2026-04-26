//! Configuration options for textclap. See spec §7.7 (Options) and §7.8 (ChunkingOptions).
//!
//! Sibling-convention notes:
//! - No `with_intra_threads` knob — thread tuning is configured outside textclap via
//!   `from_ort_session` (matches silero `options.rs:128–145`).
//! - `Options` follows soundevents' unqualified naming over silero's `SessionOptions` because
//!   textclap has no second options type to disambiguate.
//! - All getters / builders / setters are `pub const fn` (matches silero pattern).

pub use ort::session::builder::GraphOptimizationLevel;

/// Construction-time options for textclap encoders.
///
/// `Default` is implemented manually rather than derived because
/// `ort::session::builder::GraphOptimizationLevel` does not implement `Default` in
/// `ort 2.0.0-rc.12`. silero takes the same approach (`silero/src/options.rs:147`).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Options {
  graph_optimization_level: GraphOptimizationLevel,
}

impl Default for Options {
  fn default() -> Self {
    Self::new()
  }
}

impl Options {
  /// Construct with default values (== `Self::default()`).
  pub const fn new() -> Self {
    Self {
      graph_optimization_level: GraphOptimizationLevel::Level3,
    }
  }

  /// Set the ORT graph-optimization level (consuming builder).
  pub const fn with_graph_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.graph_optimization_level = level;
    self
  }

  /// Set the ORT graph-optimization level (in-place setter).
  pub const fn set_graph_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self {
    self.graph_optimization_level = level;
    self
  }

  /// Get the configured graph-optimization level.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn graph_optimization_level(&self) -> GraphOptimizationLevel {
    self.graph_optimization_level
  }
}

/// Chunking-window configuration for `embed_chunked`.
///
/// In 0.1.0 the aggregation strategy is fixed (centroid or spherical-mean, chosen at construction
/// per §3.2). See spec §7.8.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkingOptions {
  window_samples: usize,
  hop_samples: usize,
  batch_size: usize,
}

impl Default for ChunkingOptions {
  fn default() -> Self {
    Self {
      window_samples: 480_000,
      hop_samples: 480_000,
      batch_size: 8,
    }
  }
}

impl ChunkingOptions {
  /// Construct with default values (window=480_000, hop=480_000, batch_size=8).
  pub fn new() -> Self {
    Self::default()
  }

  /// Set the window length in samples (consuming builder).
  pub const fn with_window_samples(mut self, n: usize) -> Self {
    self.window_samples = n;
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
  pub const fn with_hop_samples(mut self, n: usize) -> Self {
    self.hop_samples = n;
    self
  }

  /// Set the hop length in samples (in-place setter).
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
  pub const fn with_batch_size(mut self, n: usize) -> Self {
    self.batch_size = n;
    self
  }

  /// Set the maximum batch size (in-place setter).
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
    assert_eq!(a.graph_optimization_level(), b.graph_optimization_level());
  }

  #[test]
  fn options_with_round_trips() {
    let opts = Options::new().with_graph_optimization_level(GraphOptimizationLevel::Level1);
    assert_eq!(
      opts.graph_optimization_level(),
      GraphOptimizationLevel::Level1
    );
  }

  #[test]
  fn options_set_round_trips() {
    let mut opts = Options::new();
    opts.set_graph_optimization_level(GraphOptimizationLevel::Disable);
    assert_eq!(
      opts.graph_optimization_level(),
      GraphOptimizationLevel::Disable
    );
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
