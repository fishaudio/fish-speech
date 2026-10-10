# English voiceover — start, stop, and MCP setup

Your current model is **OpenAudio S1 Mini**. It generates English voiceovers
locally using your Mac's Apple GPU. The model weights take about **3.6 GB**;
you do not need to download or install them again each time you start it.

Project folder:

```text
/Volumes/Files/Voice Model/fish-speech
```

## 1. Start the voice server

1. Keep the external drive **Files** connected and open the project folder in Finder.
2. Double-click **Start Fish Speech.command**.
3. Keep its Terminal window open. The browser opens after model loading and the startup test finish.
4. Open **http://127.0.0.1:7860**. The page should show **Fish Speech · S1 Mini**.

If the server is already running, the shortcut opens its existing page. Allow
time for the model to load before generating audio.

Alternatively, start it from Terminal:

```sh
cd "/Volumes/Files/Voice Model/fish-speech"
./"Start Fish Speech.command"
```

## 2. Stop the voice server

Double-click **Stop Fish Speech.command**, or press **Control-C** in the
Terminal window running the server. Closing the browser tab does not stop it.
Stopping the model process releases its memory; the model weights and generated
WAV files remain on disk.

From Terminal:

```sh
cd "/Volumes/Files/Voice Model/fish-speech"
./"Stop Fish Speech.command"
```

## 3. Use your voice in the browser

1. Enter your English video script in **Input Text**.
2. Open **Reference Audio** and upload a clear recording.
3. Enter the exact words spoken in the recording under **Reference Text**.
4. Leave **Reference ID** empty. Click **Generate** and listen to the result.
5. Download the WAV from the audio player and import it into your video editor.

Your tested reference:

```text
Audio: /Users/hadi/Downloads/New Recording.m4a
Transcript: My name is hadiuzzaman, im 30 years old
```

WAV and your M4A recording have been tested. M4A is decoded locally with FFmpeg.
The decoder also supports common formats such as MP3, FLAC, OGG, AAC, and AIFF.
A clear 5–10 second recording is a good starting point; keep music and background
noise low. Output is WAV; the tested files are **44,100 Hz mono**.

For longer videos, generate narration in sections of 2–3 sentences. This lets
you regenerate individual sections when needed. Listen to each section before
placing it in the editor.

## 4. How the MCP connection works

```text
Codex / Claude Code
  → local-voice MCP adapter
  → running Fish Speech server (127.0.0.1:7860)
  → WAV file inside your video project
  → your video editing MCP imports it and places it on the timeline
```

The adapter provides two tools:

| Tool | Purpose |
| --- | --- |
| `voice_server_status` | Checks whether the voice server is ready |
| `generate_voiceover` | Generates a WAV from text and an optional reference, then returns its path and duration |

The adapter and its separate Python environment, **`.venv-mcp`**, are installed
in this project. Fish Speech's Python packages remain in `venv`. Start the
model server with its shortcut; your MCP client starts the adapter.

**The connection commands below have not yet been registered in your Codex or
Claude Code settings. Register once; you do not need to repeat registration
every time.** Your video editing MCP must be connected separately. Its editor
has not been specified, so this guide does not include its registration settings.

## 5. Connect Codex once

Run in Terminal:

```sh
codex mcp add local-voice -- \
  "/Volumes/Files/Voice Model/fish-speech/.venv-mcp/bin/python" \
  "/Volumes/Files/Voice Model/fish-speech/local_voice_mcp.py"

codex mcp get local-voice
```

Voice generation can take more than a minute. Set **`tool_timeout_sec = 900`**
in this server's entry in `~/.codex/config.toml`. The complete entry should be:

```toml
[mcp_servers.local-voice]
command = "/Volumes/Files/Voice Model/fish-speech/.venv-mcp/bin/python"
args = ["/Volumes/Files/Voice Model/fish-speech/local_voice_mcp.py"]
tool_timeout_sec = 900
```

After the command creates the entry, add the timeout to **that existing entry**.
Do not add a second table with the same name. Preserve your other settings.
Restart Codex or open a new session. Check `local-voice` using `/mcp` in the
CLI or the app / extension's MCP settings.
[Official Codex MCP instructions](https://learn.chatgpt.com/docs/extend/mcp).

If the `codex` command is unavailable, add a STDIO server through Codex's MCP
settings: name it `local-voice`, use the Python path above as the command, and
provide the full `local_voice_mcp.py` path as its argument. Set the timeout in
the configuration as well.

## 6. Connect Claude Code once

Run in Terminal:

```sh
claude mcp add --transport stdio --scope user local-voice -- \
  "/Volumes/Files/Voice Model/fish-speech/.venv-mcp/bin/python" \
  "/Volumes/Files/Voice Model/fish-speech/local_voice_mcp.py"

claude mcp get local-voice
```

Using `--scope user` makes the adapter available across your video projects.
Open a new Claude Code session. To allow longer generations, launch with:

```sh
MCP_TOOL_TIMEOUT=900000 claude
```

Inside Claude Code, use `/mcp` to check the connection. If the client requests
approval to use a tool, approve it there.
[Official Claude Code MCP instructions](https://code.claude.com/docs/en/mcp).

## 7. Your workflow each time

1. Start the voice server with **Start Fish Speech.command** and wait for it to load.
2. Open Codex or Claude Code. Once registered, the adapter starts with the client.
   Keep your video editing MCP connected too.
3. Ask the agent to call `voice_server_status` first. Generate narration when it returns `ready: true`.
4. Set `generate_voiceover`'s output path inside your video project so the editor
   can read it. Existing WAV files are not overwritten; use a new filename.
5. When finished, run **Stop Fish Speech.command**.

Copy this prompt and adjust the output path to your video folder:

```text
Check local-voice's voice_server_status. Generate an English voiceover with
generate_voiceover using this text: "Welcome to my channel. Let's get started."
Use reference_audio_path: /Users/hadi/Downloads/New Recording.m4a
Use reference_text: My name is hadiuzzaman, im 30 years old
Save output_path: /Volumes/Files/Voice Model/fish-speech/results/intro-01.wav
Return its path and duration. Then use my connected video editing MCP to import
that WAV into the current project as narration and align it with the opening.
```

The video editing step depends on the tools provided by your connected editor.
Codex and Claude Code can use the same running voice server. Let one generation
finish before starting the next. Voice inference runs locally without an API
key; Codex and Claude use their own model connections as usual.

## 8. Generate from Terminal or an agent without MCP

This command uses the same running server's API:

```sh
cd "/Volumes/Files/Voice Model/fish-speech"
./venv/bin/python generate_voiceover.py \
  --text "Welcome to my channel. Let's get started." \
  --reference-audio "/Users/hadi/Downloads/New Recording.m4a" \
  --reference-text "My name is hadiuzzaman, im 30 years old" \
  --output "results/intro-01.wav"
```

For a general voice without your reference, omit both `--reference-...` options.
To use a script file, replace `--text` with `--text-file script.txt`.
The command returns JSON containing the absolute WAV path, duration, sample
rate, and channel count.

## 9. Troubleshooting

| Problem | What to do |
| --- | --- |
| `ready: false` / cannot reach server | Run the Start shortcut and wait for model loading to finish |
| MCP is connected but voice generation fails | The adapter and model server are separate; the model server must also be running |
| Output already exists | Choose a new WAV filename |
| M4A reference error | Refresh the page, upload again, enter the exact transcript, and leave Reference ID empty |
| Codex tool timeout | Set `tool_timeout_sec = 900` in the `local-voice` entry, then open a new session |
| MCP executable not found | Check that the drive is connected; run `Install Voice MCP.command` if the adapter environment needs reinstalling |
| Video editor cannot read the WAV | Use a new output path in a project folder the editor can access |

If you change port **7860**, update the URLs in the helper and MCP adapter too.
This guide uses the default port. If you move the project folder, update the
registered absolute paths in both clients.

The server launched during setup writes to `logs/webui.log`. When you launch
through the shortcut, errors appear in its Terminal window. See
[LOCAL_SETUP.md](LOCAL_SETUP.md) for model details and the audio-format fix.
The earlier voice-clone test is `results/reference-clone-test.wav`; the MCP
integration test is `results/mcp-voiceover-test.wav`.

**Verified:** STDIO MCP initialization, tool discovery, server status, and voice
generation with your M4A reference succeeded using an SDK client. The MCP test
WAV is **2.694 seconds, 44,100 Hz mono**. Attempts to overwrite an existing output
return an error. After registration in Codex or Claude Code, use `/mcp` to verify
the connection in that client.
