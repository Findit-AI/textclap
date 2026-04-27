use std::{env, path::PathBuf};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use textclap::{Options, TextEncoder};

fn bench_text_encode(c: &mut Criterion) {
  let Some(dir) = env::var_os("TEXTCLAP_MODELS_DIR").map(PathBuf::from) else {
    eprintln!("skipping bench_text_encode: TEXTCLAP_MODELS_DIR not set");
    return;
  };
  let mut text = TextEncoder::from_files(
    dir.join("text_model_quantized.onnx"),
    dir.join("tokenizer.json"),
    Options::new(),
  )
  .expect("TextEncoder::from_files");
  text.warmup().expect("warmup");

  let mut group = c.benchmark_group("text_encode");
  let queries = [
    "a dog barking",
    "rain on a metal roof",
    "applause in a stadium",
    "engine starting",
    "music with drums",
    "speech with crowd",
    "door creaking",
    "alarm sound",
    "water running",
    "wind through trees",
    "footsteps on gravel",
    "phone ringing",
    "glass breaking",
    "coffee shop ambience",
    "thunderstorm",
    "bird chirping",
    "traffic noise",
    "cat meowing",
    "typing on a keyboard",
    "fire crackling",
    "ocean waves",
    "helicopter overhead",
    "construction noise",
    "violin solo",
    "drum kit",
    "electronic beep",
    "child laughing",
    "lecture hall",
    "guitar strumming",
    "vacuum cleaner",
    "clock ticking",
    "wind chimes",
  ];

  for &n in &[1usize, 8, 32] {
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
      let texts: Vec<&str> = queries.iter().take(n).copied().collect();
      b.iter(|| text.embed_batch(&texts).expect("embed_batch"));
    });
  }
  group.finish();
}

criterion_group!(benches, bench_text_encode);
criterion_main!(benches);
