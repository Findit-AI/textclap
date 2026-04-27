//! Sequential pedagogical demo: index one fixed window first, then run a single query.
//!
//! In real deployments indexing is continuous and querying is on-demand; a single main()
//! cannot naturally show both as live, so this example is explicitly sequential.
//! See spec §1.1 / §13.

use std::env;
use std::path::PathBuf;

use textclap::{Clap, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dir = env::var_os("TEXTCLAP_MODELS_DIR")
    .map(PathBuf::from)
    .ok_or("set TEXTCLAP_MODELS_DIR")?;
  let mut clap = Clap::from_files(
    dir.join("audio_model_quantized.onnx"),
    dir.join("text_model_quantized.onnx"),
    dir.join("tokenizer.json"),
    Options::new(),
  )?;
  clap.warmup()?;

  // -- Indexing side: embed a single 10 s window --
  let pcm = vec![0.0f32; 480_000]; // silence; real pipeline reads PCM from a decoder
  let audio_emb = clap.audio_mut().embed(&pcm)?;
  println!(
    "indexed audio window: dim={}, first 3 = [{:.4}, {:.4}, {:.4}]",
    audio_emb.dim(),
    audio_emb.as_slice()[0],
    audio_emb.as_slice()[1],
    audio_emb.as_slice()[2],
  );

  // -- Query side: encode a search text and compute cosine similarity --
  let query = clap.text_mut().embed("dog barking near a door")?;
  let similarity = audio_emb.cosine(&query);
  println!("cosine similarity to 'dog barking near a door': {:.4}", similarity);

  // -- Read-back demo: stored vectors round-trip through try_from_unit_slice --
  let stored_bytes = audio_emb.to_vec();
  let restored = textclap::Embedding::try_from_unit_slice(&stored_bytes)?;
  let restored_sim = query.cosine(&restored);
  println!("restored embedding cosine: {:.4}", restored_sim);

  Ok(())
}
