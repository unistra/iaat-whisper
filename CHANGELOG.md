# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.6] - 2025-11-19

### Added
- Added tooltips to various elements.
- Added support for custom prompts.

### Changed
- Used full labels for translating options.
- Filtered the detected language for translating options.
- Enabled PDF download for summaries.
- Formatted code using Ruff.
- Added MIME type logging for debugging.

### Fixed
- Fixed M4A conversion and resampling for audio.
- Fixed MIME types for M4A in Docker.

## [1.0.5] - 2025-10-15

### Fixed
- Bugfix for the diarization checkbox.
- Hide some experimental features for production.

## [1.0.4] - 2025-10-14

### Changed
- Improved diarization performance.
- More lazy loading for libraries to improve startup time.
- Migrated the project to `uv` for faster dependency management.

### Fixed
- Fixed style and import issues.
- Cleaned unnecessary CSS.

## [1.0.3] - 2025-10-14

### Added
- Added interview prompt.

### Changed
- Upgraded `pyannote` and `transformers` libraries.
- Upgraded `whisper` library.
- Upgraded minor libraries.
- Updated `streamlit` version.
- Lazy loading for `whisper`, `torch`, and `pyannote`.

### Fixed
- Fixed style for the latest `streamlit` version.

## [1.0.2] - 2025-10-13

### Changed
- The summarization process now uses the speaker-annotated transcript when diarization is enabled, improving accuracy.

## [1.0.1] - 2025-10-13

### Added
- **Multiple Summary Formats**: Users can now choose from different types of summaries:
    - Meeting Report
    - Presentation Summary
    - Discussion Summary
    - Brainstorming / Quick Notes
- New prompt templates for each summary type.

### Changed
- The summarization prompts have been significantly improved to provide more distinct and higher-quality results for each use case.
- The model for transcription can now be sourced from LiteLLM.
- The user-facing model selection was removed to simplify the interface.
- Several technical parameters for the LLM (temperature, max tokens) and summarizer (default length) have been exposed or adjusted.

## [1.0.0] - 2025-10-13

- Initial release of the IAAT-Whisper application.
- Core features include audio transcription (local or API), speaker diarization, summary generation, and subtitle generation.