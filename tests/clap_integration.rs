//! Integration test for textclap. Gated on TEXTCLAP_MODELS_DIR env var.
//!
//! When unset, all tests print a skip message and pass (so `cargo test` doesn't fail in
//! environments without the model files). When set, runs the §12.2 assertion battery.

use std::{env, path::PathBuf};

use textclap::{Clap, Embedding, Options};

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
