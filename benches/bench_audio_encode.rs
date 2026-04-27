use std::env;
use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use textclap::{AudioEncoder, Options};

fn bench_audio_encode(c: &mut Criterion) {
  let Some(dir) = env::var_os("TEXTCLAP_MODELS_DIR").map(PathBuf::from) else {
    eprintln!("skipping bench_audio_encode: TEXTCLAP_MODELS_DIR not set");
    return;
  };
  let mut audio = AudioEncoder::from_file(
    dir.join("audio_model_quantized.onnx"),
    Options::new(),
  )
  .expect("AudioEncoder::from_file");
  audio.warmup().expect("warmup");

  let mut group = c.benchmark_group("audio_encode");
  let samples = vec![0.0f32; 480_000];

  for &n in &[1usize, 4, 8] {
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
      let clips: Vec<&[f32]> = vec![&samples[..]; n];
      b.iter(|| audio.embed_batch(&clips).expect("embed_batch"));
    });
  }
  group.finish();
}

criterion_group!(benches, bench_audio_encode);
criterion_main!(benches);
