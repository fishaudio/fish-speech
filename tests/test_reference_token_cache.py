import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import os

import numpy as np
import torch
from fish_speech.inference_engine.reference_token_cache import ReferenceTokenCache
from fish_speech.inference_engine.reference_loader import ReferenceLoader


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'cache'
        self.cache = ReferenceTokenCache(self.root, 'f'*64, [4096]+[1024]*9)
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def encode(self, audio, device):
        self.calls.append((audio,device))
        return np.full((10,3),1 if device=='cpu' else 2,dtype=np.int64)

    def resolve(self, **options):
        return self.cache.resolve(b'owned synthetic audio',self.encode,use_memory_cache=False,**options)

    def test_save_and_cross_device_read_without_reencoding(self):
        _,a=self.resolve(save_disk_cache=True)
        _,b=self.resolve(encoder_device='gpu')
        self.assertEqual(a['token_md5'],b['token_md5']);self.assertEqual(b['source'],'disk')
        self.assertEqual(len(self.calls),1)

    def test_gpu_variant_is_read_by_cpu_and_prior_result_preserved(self):
        _,a=self.resolve(save_disk_cache=True)
        _,b=self.resolve(encoder_device='gpu',save_disk_cache=True,force_reencode=True)
        self.assertNotEqual(a['token_md5'],b['token_md5'])
        _,c=self.resolve()
        self.assertEqual(c['encoded_on'],'gpu');self.assertEqual(c['source'],'disk')
        self.assertEqual(len(list(self.root.rglob('tokens.npy'))),2)

    def test_four_disk_combinations(self):
        self.resolve(save_disk_cache=True)
        for read,save in [(True,True),(True,False),(False,True),(False,False)]:
            _,event=self.resolve(read_disk_cache=read,save_disk_cache=save)
            self.assertEqual(event['source'],'disk' if read else 'encoding')
        self.assertEqual(len(self.calls),3)

    def test_no_disk_operations_when_both_disabled(self):
        self.resolve(read_disk_cache=False,save_disk_cache=False)
        self.assertFalse(self.root.exists())

    def test_read_only_miss_does_not_save_tokens(self):
        self.resolve(read_disk_cache=True,save_disk_cache=False)
        self.assertEqual(list(self.root.rglob('tokens.npy')),[])

    def test_memory_cache_is_independent_and_force_bypasses_both(self):
        args=dict(use_memory_cache=True,read_disk_cache=False,save_disk_cache=False)
        _,a=self.cache.resolve(b'x',self.encode,**args)
        _,b=self.cache.resolve(b'x',self.encode,encoder_device='gpu',**args)
        _,c=self.cache.resolve(b'x',self.encode,encoder_device='gpu',force_reencode=True,**args)
        self.assertEqual(b['source'],'memory');self.assertEqual(c['source'],'encoding')
        self.assertEqual(len(self.calls),2)

    def test_memory_hit_can_be_saved_later(self):
        self.cache.resolve(b'x',self.encode,use_memory_cache=True,read_disk_cache=False)
        _,event=self.cache.resolve(b'x',self.encode,use_memory_cache=True,save_disk_cache=True)
        self.assertEqual(event['source'],'memory');self.assertTrue(event['disk_saved'])

    def test_audio_bytes_change_causes_miss(self):
        self.resolve(save_disk_cache=True)
        _,event=self.cache.resolve(b'changed actual content',self.encode,use_memory_cache=False)
        self.assertEqual(event['source'],'encoding');self.assertEqual(len(self.calls),2)

    def test_codec_change_invalidates_disk_cache(self):
        self.resolve(save_disk_cache=True)
        other=ReferenceTokenCache(self.root,'a'*64,[4096]+[1024]*9)
        _,event=other.resolve(b'owned synthetic audio',self.encode,use_memory_cache=False)
        self.assertEqual(event['source'],'encoding')

    def test_corrupt_array_reencodes_and_republishes_atomically(self):
        self.resolve(save_disk_cache=True)
        path=next(self.root.rglob('tokens.npy'));path.write_bytes(b'truncated')
        _,event=self.resolve(save_disk_cache=True)
        self.assertEqual(event['source'],'encoding');self.assertTrue(event['disk_saved'])
        _,again=self.resolve();self.assertEqual(again['source'],'disk')
        self.assertEqual(len(self.calls),2)

    def test_modified_metadata_rejected(self):
        self.resolve(save_disk_cache=True)
        path=next(self.root.rglob('metadata.json'))
        meta=json.loads(path.read_text());meta['audio_sha256']='0'*64;path.write_text(json.dumps(meta))
        _,event=self.resolve();self.assertEqual(event['source'],'encoding')

    def test_object_pickle_cache_is_never_loaded(self):
        self.resolve(save_disk_cache=True)
        path=next(self.root.rglob('tokens.npy'))
        np.save(path,np.array([{'not':'numeric tokens'}],dtype=object),allow_pickle=True)
        _,event=self.resolve();self.assertEqual(event['source'],'encoding')

    def test_bad_dtype_shape_and_each_codebook_range_rejected_without_cache(self):
        bad=[np.ones((10,3),dtype=np.float32),np.ones((9,3),dtype=np.int64),np.empty((10,0),dtype=np.int64)]
        x=np.zeros((10,3),dtype=np.int64);x[0,0]=4096;bad.append(x)
        x=np.zeros((10,3),dtype=np.int64);x[1,0]=1024;bad.append(x)
        for value in bad:
            with self.assertRaises(ValueError):
                self.cache.resolve(b'bad',lambda *_:value,use_memory_cache=False,save_disk_cache=True)
        self.assertEqual(list(self.root.rglob('tokens.npy')),[])

    def test_encoding_failure_does_not_save(self):
        def fail(*_):raise torch.OutOfMemoryError('simulated GPU OOM')
        with self.assertRaises(torch.OutOfMemoryError):
            self.cache.resolve(b'bad',fail,use_memory_cache=False,save_disk_cache=True)
        self.assertEqual(list(self.root.rglob('tokens.npy')),[])

    def test_two_writers_share_one_published_encoding(self):
        second=ReferenceTokenCache(self.root,'f'*64,[4096]+[1024]*9)
        def slow(audio,device):
            time.sleep(0.05);return self.encode(audio,device)
        def run(cache):return cache.resolve(b'concurrent',slow,use_memory_cache=False,save_disk_cache=True)[1]
        with ThreadPoolExecutor(2) as pool:events=list(pool.map(run,[self.cache,second]))
        self.assertEqual(sorted(x['source'] for x in events),['disk','encoding'])
        self.assertEqual(len(list(self.root.rglob('tokens.npy'))),1)

    @unittest.skipIf(os.name == 'nt', 'Windows privacy uses filesystem ACLs, not POSIX modes')
    def test_private_permissions(self):
        self.resolve(save_disk_cache=True)
        self.assertEqual(self.root.stat().st_mode&0o777,0o700)
        for path in self.root.rglob('tokens.npy'):
            self.assertEqual(path.stat().st_mode&0o777,0o600)

    def test_unavailable_disk_cache_does_not_break_encoding(self):
        unavailable=Path(self.temp.name)/'not-a-directory';unavailable.write_text('file')
        cache=ReferenceTokenCache(unavailable/'cache','f'*64,[4096]+[1024]*9)
        _,event=cache.resolve(b'x',self.encode,save_disk_cache=True)
        self.assertEqual(event['source'],'encoding');self.assertFalse(event['disk_saved'])

    def test_clone_and_saved_voice_share_identity_and_latest_text(self):
        class Loader(ReferenceLoader):
            def encode_reference(inner,audio,enabled,encoder_device=None):
                return self.encode(audio,encoder_device)
        previous=Path.cwd()
        try:
            os.chdir(self.temp.name)
            loader=Loader();loader.reference_token_cache=self.cache;loader._reference_codec_config=("fixture", "fixture")
            folder=Path('references')/'voice-one';folder.mkdir(parents=True)
            (folder/'original.wav').write_bytes(b'owned synthetic audio')
            (folder/'original.lab').write_text('saved text one')
            options=dict(save_disk_cache=True)
            _,text=loader.load_by_hash([SimpleNamespace(audio=b'owned synthetic audio',text='upload text one')],'off',**options)
            self.assertEqual(text,['upload text one'])
            _,text=loader.load_by_id('voice-one','off',**options)
            self.assertEqual(text,['saved text one']);self.assertEqual(loader.reference_cache_events[-1]['source'],'disk')
            (folder/'original.wav').rename(folder/'renamed.wav')
            (folder/'original.lab').rename(folder/'renamed.lab')
            (folder/'renamed.lab').write_text('saved text updated')
            folder.rename(Path('references')/'voice-two')
            _,text=loader.load_by_id('voice-two','off',**options)
            self.assertEqual(text,['saved text updated'])
            _,text=loader.load_by_hash([SimpleNamespace(audio=b'owned synthetic audio',text='upload text updated')],'off',**options)
            self.assertEqual(text,['upload text updated']);self.assertEqual(len(self.calls),1)
        finally:os.chdir(previous)


if __name__=='__main__':unittest.main(verbosity=2)
