use criterion::{Criterion, criterion_group, criterion_main};

fn bench_mel(c: &mut Criterion) {
  // mel.rs is pub(crate); we exercise the full pipeline through the public AudioEncoder
  // in bench_audio_encode.rs. Placeholder to wire up the harness for future expansion.
  c.bench_function("mel_placeholder", |b| b.iter(|| 0_u32));
}

criterion_group!(benches, bench_mel);
criterion_main!(benches);
