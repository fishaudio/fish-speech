# Upstream sync and opt-in CUDA 13 Implementation Plan

**Goal:** Merge upstream 5415669 (Fish Speech 3.0) while retaining the fork's API client and DGX Spark setup.
**Architecture:** Preserve root upstream dependency metadata and all existing extras. A separate uv project installs this repository editably with explicit CUDA 13 overrides, selected through its cu130 extra. This avoids incompatible mandatory and optional Torch pins.
**Tech Stack:** Python, uv, shell, upstream TypeScript WebUI.
**Spec:** User request and fishaudio/fish-speech PR #1348 comment 6095246512.

## Constraints and review focus
- Bare pip and uv root installs retain torch/torchaudio 2.8.0.
- CPU/CUDA 12/stable profiles retain upstream indexes and conflicts.
- Spark explicitly uses torch/torchaudio 2.10.0 and torchcodec 0.10.0.
- Drama-3 cannot use --compile; remove implicit compilation from shared launchers.
- Old Gradio entry point is removed upstream; serve built Awesome WebUI at /ui.
- Preserve standalone HTTP client and test its existing contracts.

## Tasks
- [x] Merge upstream and resolve dependency/docs/lock conflicts using upstream defaults.
- [x] Add isolated Spark uv project, installer, and update launchers/docs.
- [x] Regenerate both locks; check default and Spark resolution and conflicting extras.
- [x] Run client and launcher tests, Python syntax and shell syntax checks.
- [x] Commit and push a review branch and open a fork PR describing the migration and hardware limits.

## Validation record

- Root pyproject.toml and uv.lock exactly match upstream 5415669 after `uv lock`.
- Both projects pass `uv lock --check`.
- Root sync dry-run selects torch/torchaudio 2.8.0.
- Spark Linux ARM64 / Python 3.12 dry-run selects torch/torchaudio 2.10.0+cu130 and torchcodec 0.10.0+cu130.
- CPU + cu129 conflict still rejects activation as expected.
- Client and launcher tests: 3 passed. Python compileall and shell syntax checks passed.
- Full pytest collection blocked by missing ML/runtime dependencies: test_active_kv_attention, test_drama3_chunking, test_fish_omni_backbone, test_fish_omni_generate (torch); test_server_drama3 (soundfile); test_visualization (package not installed); tools/drama3_ref_test (requests). No GPU or checkpoint inference was performed.

- Awesome WebUI: `npm ci && npm run build` passed (TypeScript + Vite).
- Whitespace check against upstream passed; merge diff retains upstream BibTeX trailing whitespace.
