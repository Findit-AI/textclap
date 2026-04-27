#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Error: TARGET is not provided"
  exit 1
fi

TARGET="$1"

# Install cross-compilation toolchain on Linux
if [ "$(uname)" = "Linux" ]; then
  case "$TARGET" in
    aarch64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-aarch64-linux-gnu
      ;;
  esac
fi

rustup toolchain install nightly --component miri
rustup override set nightly
cargo miri setup

export MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-disable-isolation -Zmiri-symbolic-alignment-check"

# Scope to the simd module only. Miri cannot simulate the FFI / native-lib
# dependencies pulled in by mel.rs (rustfft), text.rs (tokenizers /
# onig_sys), or audio.rs (ort / ONNX Runtime). The simd module is the one
# place in the crate that contains hand-written `unsafe` blocks worth
# verifying for UB (pointer arithmetic, target_feature dispatch, aliasing
# in the per-backend kernels), and its tests are pure compute.
cargo miri test --lib --target "$TARGET" simd::
