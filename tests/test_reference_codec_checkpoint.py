"""Opt-in real Codec test: uses generated tone bytes, no stored user audio."""
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import soundfile as sf

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.utils.schema import ServeReferenceAudio


@unittest.skipUnless(os.environ.get('FISH_CODEC_TEST_CHECKPOINT'), 'requires an existing Codec checkpoint')
class CheckpointTests(unittest.TestCase):
    def test_real_cpu_encoding_saved_voice_content_cache_and_fresh_loader(self):
        class Loader(ReferenceLoader, VQManager):
            pass

        def loader():
            item = Loader()
            item.decoder_model = SimpleNamespace(sample_rate=44100, frame_length=2048,
                quantizer=SimpleNamespace(semantic_quantizer=SimpleNamespace(codebook_size=4096),
                    quantizer=SimpleNamespace(codebook_size=1024, quantizers=[None] * 9)))
            item.configure_reference_token_cache(os.environ['FISH_CODEC_TEST_CHECKPOINT'], 'modded_dac_vq')
            return item

        audio = io.BytesIO()
        tone = (.1 * np.sin(2 * np.pi * 440 * np.arange(22050) / 44100)).astype(np.float32)
        sf.write(audio, tone, 44100, format='WAV', subtype='PCM_16')
        raw = audio.getvalue()
        previous = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                first = loader()
                refs = [ServeReferenceAudio(audio=raw, text='synthetic tone fixture')]
                tokens, _ = first.load_by_hash(refs, 'off', encoder_device='cpu', save_disk_cache=True)
                event = first.reference_cache_events[-1]
                self.assertEqual(event['source'], 'encoding')
                self.assertTrue(event['disk_saved'])
                self.assertEqual(tokens[0].shape[0], 10)
                # Cross-device cache hit must not attempt native GPU encoding.
                first.load_by_hash(refs, 'off', encoder_device='gpu')
                self.assertEqual(first.reference_cache_events[-1]['source'], 'disk')
                self.assertEqual(first.reference_cache_events[-1]['token_md5'], event['token_md5'])
                folder = Path('references/fixture'); folder.mkdir()
                (folder/'renamed.wav').write_bytes(raw)
                (folder/'renamed.lab').write_text('updated saved text')
                fresh = loader()
                _, texts = fresh.load_by_id('fixture', 'off', encoder_device='cpu')
                self.assertEqual(texts, ['updated saved text'])
                self.assertEqual(fresh.reference_cache_events[-1]['source'], 'disk')
                self.assertFalse(hasattr(fresh, '_cpu_reference_encoder'))
                changed = bytearray(raw); changed[-100] ^= 1
                first.load_by_hash([ServeReferenceAudio(audio=bytes(changed), text='changed')], 'off', encoder_device='cpu')
                self.assertEqual(first.reference_cache_events[-1]['source'], 'encoding')
                self.assertNotEqual(first.reference_cache_events[-1]['audio_md5'], event['audio_md5'])
                print('Real CPU Codec encoding, persistent/cache hits, fresh loader and content-change fallback passed.')
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main(verbosity=2)
