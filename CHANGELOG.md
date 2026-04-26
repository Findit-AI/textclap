# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

### Added
- Initial public release.
- `AudioEncoder` and `TextEncoder` for the LAION CLAP-HTSAT-unfused model.
- Top-level `Clap` with zero-shot classification helper.
- `Embedding` type with `is_close` / `is_close_cosine` similarity helpers.
- `serde` feature for `Options` / `ChunkingOptions` / `Embedding` (sequence form).
