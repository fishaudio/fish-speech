# MCP server

`tools/mcp_server.py` exposes the self-hosted Fish Speech API as a
[Model Context Protocol](https://modelcontextprotocol.io/) server. It is a thin
HTTP client: it does not import Torch, CUDA, or the model code, so it can run on
the Spark or on another machine that can reach the API.

The MCP environment is separate from the CUDA model environment. Installing the
MCP SDK therefore cannot change the pinned Torch/Pydantic versions used for
inference.

## Start

On the Spark, start the TTS API first:

```bash
./start.sh
```

In another terminal, start the MCP server on stdio:

```bash
./start-mcp.sh
```

`start-mcp.sh` creates `environments/mcp/.venv` on first use and then runs the
server. For an HTTP transport instead of stdio:

```bash
./start-mcp.sh --transport streamable-http --host 127.0.0.1 --port 8765
# endpoint: http://127.0.0.1:8765/mcp
```

Keep HTTP transports on loopback unless you put authentication in front of them;
the MCP server itself does not add authentication.

## Client configuration

A stdio MCP client launches the script as a child process. This is a generic
configuration; the exact file location depends on the client.

```json
{
  "mcpServers": {
    "fish-speech": {
      "command": "/home/spark/opt/TTS/dgxspark.fish-speech/start-mcp.sh",
      "env": {
        "FISH_API_URL": "http://127.0.0.1:8080"
      }
    }
  }
}
```

To use an API on another machine, set `FISH_API_URL` to that host and keep the
MCP server on the machine where you want the generated files to be written. If
the API requires `--api-key`, also set `FISH_API_KEY` to the same value.

## Tools

| Tool | Purpose |
| --- | --- |
| `fish_health` | Check that the API is reachable. |
| `fish_defaults` | Show the loaded model family and sampling defaults. |
| `fish_voices_list` | List saved reference voice IDs. |
| `fish_voice_add` | Register a WAV and its exact transcript. |
| `fish_voice_delete` | Delete a saved voice. |
| `fish_tts` | Generate speech and save it to a file. |

`fish_tts` accepts `voice` (a saved ID) or `reference_audio` +
`reference_text` for a one-shot clone. The file path is read on the **MCP
server** host, base64-encoded, and sent to the API; the API host does not need
that path. The returned `path` is also on the MCP server host.

Generated files go to `outputs/mcp/` by default. Set
`FISH_MCP_OUTPUT_DIR=/some/directory` to change it. File names are timestamped
and include a short random suffix, so concurrent calls do not overwrite each
other. `include_audio_base64=true` additionally returns the audio bytes in the
tool result; this is convenient for small clips but inflates large responses.

### Example calls

```json
{"text": "Welcome to the DGX Spark.", "voice": "narrator", "seed": 42}
```

```json
{
  "text": "This is a one-shot clone.",
  "reference_audio": "/home/spark/voices/reference.wav",
  "reference_text": "The exact words spoken in the reference.",
  "speaker": 0,
  "temperature": 0.7
}
```

The result is JSON with `path`, `format`, `bytes`, `elapsed_s`, `rtf`,
`duration_s`, `sample_rate`, and `channels` (WAV only for the last four).

For interactive first-audio latency, pass `"chunk_length": 100`. The server then
splits untagged text into smaller generation chunks; the tradeoff is more chunk
boundaries. The API default is 200.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `FISH_API_URL` | `http://127.0.0.1:8080` | Fish Speech API base URL. |
| `FISH_API_KEY` | unset | Bearer token if the API was started with `--api-key`. |
| `FISH_API_TIMEOUT` | `600` | HTTP timeout in seconds. |
| `FISH_MCP_OUTPUT_DIR` | `outputs/mcp` | Where generated audio is written. |
| `FISH_MCP_TRANSPORT` | `stdio` | `stdio`, `sse`, or `streamable-http`. |
| `FISH_MCP_HOST` | `127.0.0.1` | Bind host for HTTP transports. |
| `FISH_MCP_PORT` | `8765` | Bind port for HTTP transports. |

## Tests

The MCP tests use a fake API and do not load the model:

```bash
uv run --project environments/mcp --locked python -m unittest tests.test_mcp_server -v
```
