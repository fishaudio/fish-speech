"""HTTP contract tests; no model, Torch, or GPU required."""

import importlib.util
import json
import tempfile
import threading
import unittest
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ormsgpack

ROOT = Path(__file__).resolve().parents[1]


class ClientTest(unittest.TestCase):
    def test_client_contract(self):
        self.assertTrue((ROOT / "fishapi.py").exists(), "Standalone client is missing")
        spec = importlib.util.spec_from_file_location("fishapi", ROOT / "fishapi.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def handle_request(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                calls.append((self.command, self.path, self.headers, body))
                status, content_type = 200, "application/json"
                if self.path == "/v1/tts":
                    payload = json.loads(body)
                    if payload["text"] == "error":
                        status, data = 500, b'{"detail":"generation failed"}'
                    else:
                        content_type, data = "audio/wav", b"RIFF-test-audio"
                elif self.path == "/ui":
                    content_type, data = "text/html", b"<html>Fish</html>"
                elif self.path == "/v1/vqgan/encode":
                    data = ormsgpack.packb({"tokens": [[[1, 2]]]})
                    content_type = "application/msgpack"
                elif self.path == "/v1/vqgan/decode":
                    data = ormsgpack.packb({"audios": [b"\x00\x00"]})
                    content_type = "application/msgpack"
                elif self.path == "/v1/references/list":
                    data = b'{"success":true,"reference_ids":["alice"]}'
                elif self.path == "/v1/health":
                    data = b'{"status":"ok"}'
                else:
                    data = b'{"success":true}'
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_DELETE = handle_request

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            api = module.FishAPI(f"127.0.0.1:{server.server_port}", api_key="test-key")
            self.assertEqual(api.health(), {"status": "ok"})
            self.assertEqual(api.list_voices(), ["alice"])
            self.assertEqual(api.health(method="POST"), {"status": "ok"})
            self.assertEqual(api.webui(), "<html>Fish</html>")
            self.assertEqual(api.encode([b"input"]), [[[1, 2]]])
            self.assertEqual(ormsgpack.unpackb(calls[-1][3]), {"audios": [b"input"]})
            self.assertEqual(api.decode([[[1, 2]]]), [b"\x00\x00"])
            self.assertEqual(ormsgpack.unpackb(calls[-1][3]), {"tokens": [[[1, 2]]]})
            self.assertEqual(
                b"".join(api.generate_stream("hi", chunk_size=3)), b"RIFF-test-audio"
            )
            self.assertTrue(json.loads(calls[-1][3])["streaming"])
            self.assertEqual(
                module.FishAPI("spark.local").base_url, "http://spark.local:8080"
            )
            with tempfile.TemporaryDirectory() as directory:
                audio = Path(directory) / "reference.wav"
                audio.write_bytes(b"RIFF-reference")
                api.addvoice("alice", audio, "Hola, ¿cómo estás?")
                headers, body = calls[-1][2:]
                message = BytesParser(policy=default).parsebytes(
                    ("Content-Type: " + headers["Content-Type"] + "\r\n\r\n").encode()
                    + body
                )
                fields = {
                    part.get_param(
                        "name", header="content-disposition"
                    ): part.get_payload(decode=True)
                    for part in message.iter_parts()
                }
                self.assertEqual(
                    fields,
                    {
                        "id": b"alice",
                        "text": "Hola, ¿cómo estás?".encode(),
                        "audio": b"RIFF-reference",
                    },
                )
                output = Path(directory) / "turn.wav"
                result = api.generate("Agreed.", voice="alice", output=output, seed=42)
                self.assertEqual(result, output.read_bytes())
                self.assertEqual(json.loads(calls[-1][3])["reference_id"], "alice")
                self.assertFalse(json.loads(calls[-1][3])["streaming"])
                output.write_bytes(b"keep me")
                with self.assertRaisesRegex(
                    module.FishAPIError, "500.*generation failed"
                ):
                    api.generate("error", output=output)
                self.assertEqual(output.read_bytes(), b"keep me")
            api.rename_voice("alice", "bob")
            self.assertEqual(
                json.loads(calls[-1][3]),
                {"old_reference_id": "alice", "new_reference_id": "bob"},
            )
            api.delete_voice("bob")
            self.assertEqual(calls[-1][:2], ("DELETE", "/v1/references/delete"))
            self.assertEqual(json.loads(calls[-1][3]), {"reference_id": "bob"})
            self.assertTrue(
                all(c[2]["Authorization"] == "Bearer test-key" for c in calls)
            )
            self.assertTrue(
                all(
                    c[2]["Accept"] == "application/json"
                    for c in calls
                    if c[1].startswith("/v1/references")
                )
            )
            with self.assertRaises(ValueError):
                api.generate("hello", streaming=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
