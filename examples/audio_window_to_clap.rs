//! Indexing path demo: source frames at native rate (e.g. 44.1 kHz) -> rubato resample to 48 kHz
//! -> buffer 10 s -> AudioEncoder::embed -> 512-dim Embedding -> push to a stubbed lancedb writer.
//!
//! Run with TEXTCLAP_MODELS_DIR set to a directory containing audio_model_quantized.onnx.
//! See spec §1.1.

use std::env;
use std::path::PathBuf;

use rubato::{
  Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use textclap::{AudioEncoder, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dir = env::var_os("TEXTCLAP_MODELS_DIR")
    .map(PathBuf::from)
    .ok_or("set TEXTCLAP_MODELS_DIR to the directory containing audio_model_quantized.onnx")?;

  let mut audio = AudioEncoder::from_file(dir.join("audio_model_quantized.onnx"), Options::new())?;
  audio.warmup()?;

  // Simulate a decoder producing 44.1 kHz mono frames. In a real pipeline this comes from
  // ffmpeg / gstreamer / a microphone callback / etc.
  let source_rate = 44_100usize;
  let target_rate = 48_000usize;
  let chunk_size = 4_410usize; // 100 ms at 44.1 kHz
  let total_seconds = 10usize;

  // rubato 2.x async sinc resampler (44.1 -> 48 kHz, mono, fixed input chunk).
  let params = SincInterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    oversampling_factor: 256,
    interpolation: SincInterpolationType::Linear,
    window: WindowFunction::BlackmanHarris2,
  };
  let mut resampler = Async::<f32>::new_sinc(
    target_rate as f64 / source_rate as f64,
    2.0,
    &params,
    chunk_size,
    1, // mono
    FixedAsync::Input,
  )?;

  let mut buffer_48k: Vec<f32> = Vec::with_capacity(target_rate * total_seconds);
  let target_samples = target_rate * total_seconds; // 480 000

  // Pre-allocated output scratch sized to the resampler's worst case.
  let max_out = resampler.output_frames_max();
  let mut output_data: Vec<Vec<f32>> = vec![vec![0.0f32; max_out]; 1];

  let frames_per_chunk = source_rate * total_seconds / chunk_size; // ~100
  for i in 0..frames_per_chunk {
    // Synthesize a 1 kHz sine for demonstration; real pipeline reads from the decoder.
    let mut frame = vec![0.0f32; chunk_size];
    for k in 0..chunk_size {
      let t = (i * chunk_size + k) as f32 / source_rate as f32;
      frame[k] = (2.0 * std::f32::consts::PI * 1000.0 * t).sin();
    }
    let input_data = vec![frame];
    let input = SequentialSliceOfVecs::new(&input_data, 1, chunk_size)?;
    let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, max_out)?;
    let (_in_frames, out_frames) = resampler.process_into_buffer(&input, &mut output, None)?;
    buffer_48k.extend_from_slice(&output_data[0][..out_frames]);

    if buffer_48k.len() >= target_samples {
      let window: Vec<f32> = buffer_48k.drain(..target_samples).collect();
      let embedding = audio.embed(&window)?;
      // In a real pipeline: push embedding.as_slice() into a lancedb FixedSizeListBuilder.
      println!(
        "indexed 10 s window - embedding dim={}, head=[{:.4}, {:.4}, {:.4}, ...]",
        embedding.dim(),
        embedding.as_slice()[0],
        embedding.as_slice()[1],
        embedding.as_slice()[2],
      );
    }
  }
  Ok(())
}
