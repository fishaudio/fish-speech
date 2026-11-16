import base64
import io
import json
import unittest
import wave

from pydantic import ValidationError

from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


def make_wav(samples: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(b"\x00\x00" * samples)
    return buffer.getvalue()


class ReferenceAudioSchemaTest(unittest.TestCase):
    def test_json_base64_audio_is_decoded_regardless_of_length(self):
        for samples in (8, 400):
            with self.subTest(samples=samples):
                audio = make_wav(samples)
                encoded = base64.b64encode(audio).decode("ascii")
                request = ServeTTSRequest.model_validate_json(
                    json.dumps(
                        {
                            "text": "Hello",
                            "references": [{"audio": encoded, "text": "Hi"}],
                        }
                    )
                )
                self.assertEqual(request.references[0].audio, audio)
                with wave.open(io.BytesIO(request.references[0].audio)) as decoded:
                    self.assertEqual(decoded.getnframes(), samples)

    def test_line_wrapped_base64_remains_supported(self):
        audio = make_wav(400)
        encoded = base64.encodebytes(audio).decode("ascii")
        reference = ServeReferenceAudio(audio=encoded, text="Hi")
        self.assertEqual(reference.audio, audio)

    def test_binary_audio_is_not_base64_decoded(self):
        for audio in (make_wav(400), b"YWJj"):
            with self.subTest(audio_size=len(audio)):
                reference = ServeReferenceAudio(audio=audio, text="Hi")
                self.assertEqual(reference.audio, audio)

    def test_validation_does_not_modify_the_input_dictionary(self):
        encoded = base64.b64encode(make_wav(400)).decode("ascii")
        values = {"audio": encoded, "text": "Hi"}
        reference = ServeReferenceAudio.model_validate(values)
        self.assertEqual(values["audio"], encoded)
        self.assertEqual(reference.audio, base64.b64decode(encoded))

    def test_invalid_base64_is_a_validation_error(self):
        for encoded in ("not base64!", "a", "é", "!" * 300):
            with self.subTest(encoded=encoded):
                with self.assertRaises(ValidationError) as raised:
                    ServeReferenceAudio(audio=encoded, text="Hi")
                self.assertEqual(raised.exception.errors()[0]["loc"], ("audio",))

    def test_non_object_references_are_validation_errors(self):
        for reference in (None, [], "not a reference"):
            with self.subTest(reference=reference):
                with self.assertRaises(ValidationError):
                    ServeTTSRequest(text="Hello", references=[reference])


if __name__ == "__main__":
    unittest.main()
