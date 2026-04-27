//! Integration test for textclap. Gated on TEXTCLAP_MODELS_DIR env var.
//!
//! When unset, all tests print a skip message and pass (so `cargo test` doesn't fail in
//! environments without the model files). When set, runs the §12.2 assertion battery.

use std::{env, path::PathBuf};

use textclap::{ChunkingOptions, Clap, Embedding, Options};

fn models_dir() -> Option<PathBuf> {
  env::var_os("TEXTCLAP_MODELS_DIR").map(PathBuf::from)
}

fn read_npy_f32(path: &str) -> Vec<f32> {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
  npyz::NpyFile::new(&bytes[..])
    .unwrap()
    .into_vec::<f32>()
    .unwrap()
}

fn read_wav_48k_mono(path: &str) -> Vec<f32> {
  let mut reader = hound::WavReader::open(path).expect("open WAV");
  let spec = reader.spec();
  assert_eq!(spec.sample_rate, 48000, "fixture must be 48 kHz");
  assert_eq!(spec.channels, 1, "fixture must be mono");
  match spec.sample_format {
    hound::SampleFormat::Int => {
      let scale = 1.0 / (1_i64 << (spec.bits_per_sample - 1)) as f32;
      reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 * scale)
        .collect()
    }
    hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
  }
}

#[test]
fn audio_embedding_matches_golden() {
  let Some(dir) = models_dir() else {
    eprintln!("skipping: TEXTCLAP_MODELS_DIR not set");
    return;
  };
  let mut clap = Clap::from_files(
    dir.join("audio_model_quantized.onnx"),
    dir.join("text_model_quantized.onnx"),
    dir.join("tokenizer.json"),
    Options::new(),
  )
  .expect("Clap::from_files");
  clap.warmup().expect("warmup");

  let samples = read_wav_48k_mono("tests/fixtures/sample.wav");
  let emb = clap.audio_mut().embed(&samples).expect("embed");

  let golden = read_npy_f32("tests/fixtures/golden_audio_emb.npy");
  let golden_emb = Embedding::try_from_unit_slice(&golden).expect("golden is unit-norm");

  // §12.2: max-abs ≤ 5e-4 for int8-vs-int8 audio embedding.
  assert!(
    emb.is_close(&golden_emb, 5e-4),
    "audio embedding drift exceeds 5e-4 (max-abs)",
  );
  assert!(
    emb.is_close_cosine(&golden_emb, 1e-4),
    "audio embedding cosine drift exceeds 1e-4",
  );
}

#[test]
fn text_embeddings_match_golden() {
  let Some(dir) = models_dir() else {
    eprintln!("skipping: TEXTCLAP_MODELS_DIR not set");
    return;
  };
  let mut clap = Clap::from_files(
    dir.join("audio_model_quantized.onnx"),
    dir.join("text_model_quantized.onnx"),
    dir.join("tokenizer.json"),
    Options::new(),
  )
  .expect("Clap::from_files");
  clap.warmup().expect("warmup");

  let labels = ["a dog barking", "rain", "music", "silence", "door creaking"];
  let embs: Vec<Embedding> = labels
    .iter()
    .map(|label| clap.text_mut().embed(label).expect("embed"))
    .collect();

  let golden = read_npy_f32("tests/fixtures/golden_text_embs.npy");
  assert_eq!(golden.len(), 5 * 512);

  for (i, label) in labels.iter().enumerate() {
    let golden_row = &golden[i * 512..(i + 1) * 512];
    let golden_emb = Embedding::try_from_unit_slice(golden_row).expect("golden row unit-norm");
    assert!(
      embs[i].is_close(&golden_emb, 1e-5),
      "text embedding drift exceeds 1e-5 for label {label:?}",
    );
    assert!(
      embs[i].is_close_cosine(&golden_emb, 5e-8),
      "text embedding cosine drift exceeds 5e-8 for label {label:?}",
    );
  }
}

/// Regression for the Codex finding: `embed_chunked` previously accepted any positive
/// window, including windows > TARGET_SAMPLES which `MelExtractor` silently head-truncated
/// to 10 s, dropping audio after the first window. The validation guard now rejects
/// `window_samples > 480_000`.
#[test]
fn embed_chunked_rejects_oversize_window_runtime() {
  let Some(dir) = models_dir() else {
    eprintln!("skipping: TEXTCLAP_MODELS_DIR not set");
    return;
  };
  let mut clap = Clap::from_files(
    dir.join("audio_model_quantized.onnx"),
    dir.join("text_model_quantized.onnx"),
    dir.join("tokenizer.json"),
    Options::new(),
  )
  .expect("Clap::from_files");
  let samples = vec![0.0f32; 480_000];
  let opts = ChunkingOptions::new()
    .with_window_samples(480_001)
    .with_hop_samples(480_001)
    .with_batch_size(1);
  let err = clap.audio_mut().embed_chunked(&samples, &opts).unwrap_err();
  // ChunkingConfig variant should fire.
  let msg = format!("{err}");
  assert!(
    msg.contains("invalid chunking options"),
    "unexpected error: {msg}"
  );
}

/// Regression for the Codex finding: `from_ort_session` preserves the caller's padding
/// config (or lack thereof). `embed_batch` must still produce a well-formed tensor when
/// the tokenizer doesn't pad — `ids_scratch` is right-padded to true `t_max` with `pad_id`.
#[test]
fn from_ort_session_uneven_lengths_no_padding() {
  let Some(dir) = models_dir() else {
    eprintln!("skipping: TEXTCLAP_MODELS_DIR not set");
    return;
  };

  // Build a Session manually and a Tokenizer with padding explicitly cleared.
  use ort::session::Session;
  let session = Session::builder()
    .expect("Session::builder")
    .commit_from_file(dir.join("text_model_quantized.onnx"))
    .expect("commit_from_file");

  let mut tokenizer =
    tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer load");
  // Drop any padding configuration to exercise the embed_batch pad-loop's robustness.
  tokenizer.with_padding(None);

  let mut text =
    textclap::TextEncoder::from_ort_session(session, tokenizer, textclap::Options::new())
      .expect("from_ort_session");

  // Intentionally uneven lengths: 1 word vs 5 words vs 2 words.
  let labels = ["rain", "the quick brown fox jumps", "a dog"];
  let embs = text.embed_batch(&labels).expect("embed_batch");
  assert_eq!(embs.len(), 3);
  for emb in &embs {
    assert_eq!(emb.dim(), 512);
  }
}

/// Regression for the Codex round-2 finding: embed_batch must produce embeddings that match
/// per-label embed exactly, so classify_* scores don't depend on batch composition. This is
/// the inverse of the round-1 from_ort_session_uneven_lengths_no_padding test (which verified
/// embed_batch tolerates uneven lengths) — here we verify embed_batch is *semantically*
/// equivalent to per-label embed.
#[test]
fn embed_batch_matches_per_label_embed() {
  let Some(dir) = models_dir() else {
    eprintln!("skipping: TEXTCLAP_MODELS_DIR not set");
    return;
  };
  let mut clap = textclap::Clap::from_files(
    dir.join("audio_model_quantized.onnx"),
    dir.join("text_model_quantized.onnx"),
    dir.join("tokenizer.json"),
    textclap::Options::new(),
  )
  .expect("Clap::from_files");
  clap.warmup().expect("warmup");

  let labels = ["a dog barking", "rain", "music", "silence", "door creaking"];
  let batched = clap.text_mut().embed_batch(&labels).expect("embed_batch");
  let per_label: Vec<textclap::Embedding> = labels
    .iter()
    .map(|t| clap.text_mut().embed(t).expect("embed"))
    .collect();
  assert_eq!(batched.len(), per_label.len());
  for (i, (b, p)) in batched.iter().zip(per_label.iter()).enumerate() {
    // Tolerance matches the per-label golden tolerance.
    assert!(
      b.is_close(p, 1e-5),
      "batch[{i}] != per-label embed for {label:?}",
      label = labels[i]
    );
    assert!(
      b.is_close_cosine(p, 5e-8),
      "batch[{i}] cosine drift for {:?}",
      labels[i]
    );
  }
}

#[test]
fn classify_discrimination_check() {
  let Some(dir) = models_dir() else {
    eprintln!("skipping: TEXTCLAP_MODELS_DIR not set");
    return;
  };
  let mut clap = Clap::from_files(
    dir.join("audio_model_quantized.onnx"),
    dir.join("text_model_quantized.onnx"),
    dir.join("tokenizer.json"),
    Options::new(),
  )
  .expect("Clap::from_files");
  clap.warmup().expect("warmup");

  let samples = read_wav_48k_mono("tests/fixtures/sample.wav");
  let labels = ["a dog barking", "rain", "music", "silence", "door creaking"];
  let scores = clap.classify_all(&samples, &labels).expect("classify_all");

  let score_of = |query: &str| -> f32 {
    scores
      .iter()
      .find(|s| s.label() == query)
      .expect("label present")
      .score()
  };
  let dog = score_of("a dog barking");
  let music = score_of("music");

  let top_2: Vec<&str> = scores.iter().take(2).map(|s| s.label()).collect();
  assert!(
    top_2.contains(&"a dog barking"),
    "expected dog-bark in top 2; top 2 = {top_2:?}",
  );
  assert!(
    dog - music > 0.05,
    "discrimination margin too small: dog {dog:.4} − music {music:.4} = {:.4}",
    dog - music,
  );
}
