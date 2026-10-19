# Shared Hugging Face model cache

The API and text-to-semantic CLI now default to `fishaudio/s2-pro` in the shared
Hugging Face cache. Downloads happen once at startup; files already cached are
reused. They are not copied into the repository. Authentication uses `hf auth login`
or `HF_TOKEN`. Respect the model's access and license requirements.

```bash
./start-api.sh
./start-webui.sh --model fishaudio/drama-3 --device cuda:0
```

The WebUI uses the same API model resolver. Build its assets first as described
in the installation guide. Drama-3 does not support `--compile`.

For CLI inference, add `--model fishaudio/drama-3` to the existing
`fish_speech/models/text2semantic/inference.py` command. All prompt/output options
remain the same. Standalone DAC and other low-level utilities continue to take
explicit paths; use `hf download MODEL_ID` to print a cached snapshot path.

## Location, revisions, and offline use

Hugging Face chooses the cache directory: `HF_HUB_CACHE`, or the `hub` directory
under `HF_HOME` (normally `~/.cache/huggingface/hub`). Set these variables before
starting the process. For example:

```bash
export HF_HUB_CACHE=/data/shared/huggingface/hub
hf download fishaudio/s2-pro
./start-api.sh --offline
```

`--revision COMMIT_OR_TAG` selects the model revision; pin a commit to reproduce
an exact model. `--offline` (or `HF_HUB_OFFLINE=1`) prohibits Hub downloads and
requires the needed files already cached. Errors from missing cached files or
Hub authentication are reported at startup.

A codec is resolved from the model directory first, then the legacy sibling
`s2-pro/codec.pth`. If absent, only `codec.pth` is fetched from `fishaudio/s2-pro`
into the shared cache. Override that fallback with `--codec-model` and
`--codec-revision`. The model revision and fallback codec revision are independent.

Explicit local paths still work and take precedence over defaults:

```bash
./start-api.sh --llama-checkpoint-path /models/my-finetune --decoder-checkpoint-path /models/codec.pth
```

Do not combine `--model` with `--llama-checkpoint-path` (`--checkpoint-path` for
the inference CLI). A missing explicit local directory fails instead of being
treated as a model ID. No cache network calls are needed when all explicit local
model and codec files exist.

## Migrate existing weights without downloading them again

Run the standalone script from the repository root; uv installs only its small
Hub dependencies, not Torch. First verify the plan:

```bash
uv run tools/migrate_hf_cache.py checkpoints/s2-pro --model fishaudio/s2-pro --dry-run
```

Copy verified files into the shared cache, keeping the originals:

```bash
uv run tools/migrate_hf_cache.py checkpoints/s2-pro --model fishaudio/s2-pro
```

Or remove verified source files after the entire import succeeds:

```bash
uv run tools/migrate_hf_cache.py checkpoints/s2-pro --model fishaudio/s2-pro --move
```

For Drama-3, substitute `checkpoints/drama-3` and `fishaudio/drama-3`.
Use `--revision COMMIT` if the files came from an older revision. A mismatch
fails before any source deletion; do not relabel fine-tuned weights as official
Hub weights. Keep custom models as local paths.

The script fetches repository metadata over the network, not model weights.
It verifies file sizes and Git-blob SHA-1 / LFS SHA-256 hashes, makes independent
copies, verifies them, and publishes normal Hub blob/snapshot/ref entries.
It uses Hub-compatible blob locks and atomic file replacement. Source symlinks
are rejected; cache and source must be separate, non-nested directories.
Existing cache content must also verify; corrupt cache entries are not overwritten.

Missing repository files are reported: a partial migration is allowed, and the
next online startup downloads the missing files. Unknown files, local HF download
metadata, and directories remain untouched even with `--move`. The script needs
space for the cache copies before deleting any originals. Do not modify source
weights or prune the destination cache while migration is running.

To inspect the result:

```bash
hf cache list
HF_HUB_OFFLINE=1 hf download fishaudio/s2-pro
./start-api.sh --offline
```

## Docker

The Docker build defaults to uv 0.12.23, matching the regenerated lockfile.
Compose mounts `${HF_CACHE_DIR:-${HOME}/.cache/huggingface/hub}` at `/hf-cache` and
sets `HF_HUB_CACHE=/hf-cache`. Set `HF_CACHE_DIR` to your host's custom Hub cache
path if needed. Local checkpoint mounts remain available for custom models.
For the CUDA/CPU Compose server, `HF_MODEL` defaults to `fishaudio/s2-pro`;
`HF_REVISION` pins its revision and `HF_HUB_OFFLINE=1` enables offline mode.
Setting `LLAMA_CHECKPOINT_PATH` selects a local model instead.

For a manual Docker invocation, mount your cache and set `HF_HUB_CACHE` to its
container path. Ensure the container user has write permission for downloads.
