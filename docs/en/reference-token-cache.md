# Selectable reference encoding and persistent Token cache

The API and Gradio reference controls support:

- `reference_encoder_device`: `cpu` or `gpu` (default `gpu`, preserving the existing model-device route).
- `read_disk_cache`: allow persistent reads (default `true`).
- `save_disk_cache`: save successful encoding results (default `false`).
- `force_reencode`: bypass both memory and disk reads (default `false`). Saving remains independent.

These options affect uploaded references and saved reference IDs equally. The existing `use_memory_cache` option remains independent: disabling disk reads does not disable memory hits. To deliberately compare encoding results, enable `force_reencode` or disable all applicable cache reads.

`gpu` uses the existing Decoder on its configured device, including upstream CPU-only fallback. It does not promise that a GPU exists. Selecting `cpu` lazily loads a separate FP32 Codec encoder from the configured checkpoint, removes its unused Decoder, and retains it for subsequent requests. The original Decoder and LLM are not moved. This costs additional host RAM and first-request loading time. This branch does not introduce GPU workers, hardware-specific memory management, or a ROCm 7.14 port.

Both standard API startup and `tools/run_webui.py` configure the cache automatically. Integrations constructing `TTSInferenceEngine` themselves should call:

```python
engine.configure_reference_token_cache(codec_checkpoint, codec_config_name)
```

Existing integrations without checkpoint metadata retain the temporary/memory model-device path; persistent reads/writes are disabled to prevent incompatible reuse. Explicit CPU encoding requires checkpoint metadata.

## Identity and persistence

Cache identity uses MD5 of the actual input audio bytes, strengthened by SHA256, plus a Codec fingerprint. Voice IDs and filenames only locate source materials. Current reference text is always read from the current request or `.lab` file and is never cached with Tokens.

The Codec fingerprint covers checkpoint/configuration hashes, DAC and encoding source, preprocessing, and relevant library versions. Changing Codec weights or preprocessing isolates incompatible entries. Changing encoding device does not prevent a compatible cache hit.

The default root is `references/.token_cache/`; override it with `REFERENCE_TOKEN_CACHE_DIR`. With Docker, persist this root in a private writable volume. Use a cache root owned by the service account, outside web-served directories. POSIX permissions are 700 for directories and 600 for files; on Windows, privacy depends on the directory's inherited ACLs.

```text
<audio-md5>/<codec-fingerprint>/<audio-sha256>/<token-md5>-<token-sha256>/
    tokens.npy
    metadata.json
```

Tokens are canonical CPU int64 arrays. Loading forbids pickle and validates dtype, shape, per-codebook ranges, identity and checksums. Different CPU/GPU Token results are preserved as variants; the newest newly created valid variant is selected. Identical re-encodings do not replace existing metadata or promote an older variant. There is currently no manual variant selector or automatic eviction policy.

Process/thread locking and atomic directory publication prevent partial cache entries. Missing, invalid or unavailable disk entries fall back to ordinary encoding. Failed encoding never publishes valid Tokens. Disk errors do not prevent otherwise successful temporary encoding. Memory entries are bounded; disk usage is not yet bounded.

No models, audio, Tokens, credentials or private deployment configuration are included in this feature branch. The `.gitignore` additions retain example/template configuration files.

## Validation and limits

Run the model-free regression tests with the project's dependencies installed:

```bash
python -m unittest discover -s tests -p 'test_reference*.py' -v
```

The optional checkpoint test uses a generated sine tone, never a person's recording. To enable it, set `FISH_CODEC_TEST_CHECKPOINT` to an existing compatible Codec checkpoint; it does not download one.

This feature was verified with 18 cache tests, 5 model-device/inference/WebUI routing tests, and 1 real CPU Codec checkpoint test. They cover saving/loading, MD5 content changes, miss fallback, independent switches, fresh-loader persistence, saved/uploaded identity, current text, corrupt data, concurrent writers and unchanged default routing. A separate native Windows smoke test verified file locking, long paths, cross-device cache hits and corrupt repair.

The executed CPU tests used Python 3.12 and an existing Torch 2.11 container. Upstream's pinned Torch 2.8 environment has not been reproduced for this publication. GPU route tests use mocks and CPU-only upstream fallback; no NVIDIA GPU hardware run or complete LLM-to-WAV TTS run was performed on this official-upstream branch. Earlier tests of a separate private deployment are not presented as validation of this branch. GPU dtype/backend differences may produce different compatible Token variants.
