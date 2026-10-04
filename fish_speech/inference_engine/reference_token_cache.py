"""Content-addressed reference tokens; no transcript or voice ID is cached."""
from collections import OrderedDict
from contextlib import contextmanager
try:
    import fcntl
except ImportError:  # Windows: keep cache imports usable without POSIX APIs.
    fcntl = None
    import msvcrt
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import uuid

import numpy as np
from loguru import logger


FORMAT_VERSION = 1
MAX_TOKEN_BYTES = 2 * 1024 * 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def codec_fingerprint(checkpoint, config_name, preprocessing_source, model, backend):
    import importlib.metadata
    import soundfile
    import torch
    import torchaudio
    package = Path(__file__).resolve().parents[1]
    sources = {str(path.relative_to(package)): file_sha256(path)
               for path in sorted((package / 'models' / 'dac').glob('*.py'))}
    sources['inference_engine/vq_manager.py'] = file_sha256(package / 'inference_engine' / 'vq_manager.py')
    dependencies = {}
    for name in ('descript-audio-codec', 'descript-audiotools', 'torchcodec'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = 'unavailable'
    identity = {
        'format_version': FORMAT_VERSION,
        'checkpoint_sha256': file_sha256(checkpoint),
        'config_sha256': file_sha256(package / 'configs' / (config_name + '.yaml')),
        'codec_sources': sources,
        'codec_dependencies': dependencies,
        'preprocessing_sha256': hashlib.sha256(preprocessing_source.encode()).hexdigest(),
        'preprocessing': {'backend': backend, 'channel_mix': 'mean',
                          'sample_rate': model.sample_rate, 'frame_length': model.frame_length},
        'versions': {'numpy': np.__version__, 'torch': torch.__version__,
                     'torchaudio': torchaudio.__version__, 'soundfile': soundfile.__version__},
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest(), identity


class ReferenceTokenCache:
    def __init__(self, root, fingerprint, token_limits, codec_identity=None, memory_entries=32):
        self.root = Path(root)
        if os.name == 'nt':
            absolute = str(self.root.absolute())
            if not absolute.startswith('\\\\?\\'):
                absolute = ('\\\\?\\UNC\\' + absolute[2:]
                            if absolute.startswith('\\\\') else '\\\\?\\' + absolute)
            self.root = Path(absolute)
        self.fingerprint = fingerprint
        self.token_limits = tuple(token_limits)
        self.codec_identity = codec_identity or {}
        self.memory_entries = memory_entries
        self.memory = OrderedDict()
        self.lock = threading.RLock()

    def _array(self, tokens):
        if hasattr(tokens, 'detach'):
            tokens = tokens.detach().cpu().numpy()
        raw = np.asarray(tokens)
        if raw.dtype.kind not in 'iu' or raw.dtype.kind == 'b':
            raise ValueError('Reference tokens must contain integers')
        if raw.ndim != 2 or raw.shape[0] != len(self.token_limits) or not 0 < raw.shape[1] <= 16384:
            raise ValueError('Invalid reference token shape')
        if raw.nbytes > MAX_TOKEN_BYTES:
            raise ValueError('Reference token array is too large')
        for row, limit in zip(raw, self.token_limits):
            if np.min(row) < 0 or np.max(row) >= limit:
                raise ValueError('Reference token value is outside its codebook')
        return np.ascontiguousarray(raw, dtype='<i8')

    def _key(self, audio):
        return {'audio_md5': hashlib.md5(audio).hexdigest(),
                'audio_sha256': hashlib.sha256(audio).hexdigest(),
                'codec_fingerprint': self.fingerprint}

    def _directory(self, key):
        return self.root / key['audio_md5'] / key['codec_fingerprint'] / key['audio_sha256']

    @contextmanager
    def _disk_lock(self, key, enabled):
        if not enabled:
            yield False
            return
        fd = None
        try:
            if self.root.is_symlink():
                raise OSError('Token cache root must not be a symbolic link')
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.root, 0o700)
            locks = self.root / '.locks'
            locks.mkdir(mode=0o700, exist_ok=True)
            if locks.is_symlink():
                raise OSError('Token lock directory must not be a symbolic link')
            name = key['audio_md5'] + '-' + key['audio_sha256'] + '-' + key['codec_fingerprint'] + '.lock'
            lock_path = locks / name
            if lock_path.is_symlink():
                raise OSError('Symlinked token lock file')
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b'0')
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        except OSError as exc:
            logger.warning('Token disk cache unavailable; temporary encoding remains usable: {}', exc)
            if fd is not None:
                os.close(fd)
            yield False
            return
        try:
            yield True
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            else:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)

    def _read_variant(self, directory, key):
        if directory.is_symlink():
            raise ValueError('Symlinked token cache variant')
        meta_path, token_path = directory / 'metadata.json', directory / 'tokens.npy'
        if meta_path.is_symlink() or token_path.is_symlink():
            raise ValueError('Symlinked token cache file')
        if meta_path.stat().st_size > 65536 or token_path.stat().st_size > MAX_TOKEN_BYTES + 4096:
            raise ValueError('Oversized token cache file')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if not isinstance(meta, dict) or any(meta.get(k) != v for k, v in key.items()):
            raise ValueError('Incompatible audio or codec identity')
        if meta.get('format_version') != FORMAT_VERSION or meta.get('dtype') != 'int64-le':
            raise ValueError('Unsupported token cache format')
        if meta.get('encoder_device') not in ('cpu', 'gpu') or meta.get('token_limits') != list(self.token_limits):
            raise ValueError('Invalid encoder metadata or codebook sizes')
        if meta.get('codec_identity') != self.codec_identity:
            raise ValueError('Codec implementation metadata changed')
        created = meta.get('created_ns')
        if type(created) is not int or created <= 0:
            raise ValueError('Invalid cache timestamp')
        array = np.load(token_path, allow_pickle=False, max_header_size=4096)
        if not isinstance(array, np.ndarray):
            if hasattr(array, 'close'):
                array.close()
            raise ValueError('Cache file must contain one numeric NPY array')
        if array.dtype != np.dtype('<i8'):
            raise ValueError('Stored token dtype is not canonical int64')
        array = self._array(array)
        if list(array.shape) != meta.get('shape'):
            raise ValueError('Token shape disagrees with metadata')
        data = array.tobytes()
        md5, sha256 = hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()
        if md5 != meta.get('token_md5') or sha256 != meta.get('token_sha256'):
            raise ValueError('Token content checksum failed')
        if directory.name != md5 + '-' + sha256:
            raise ValueError('Token variant path disagrees with content')
        return array, meta

    def _read_disk(self, key):
        parent = self._directory(key)
        if not parent.exists():
            return None
        if any(p.is_symlink() for p in (parent, parent.parent, parent.parent.parent)):
            logger.warning('Invalid Cache / Re-encoding: symlinked identity directory')
            return None
        valid = []
        for path in sorted(parent.iterdir()):
            if path.name.startswith('.') or not path.is_dir():
                continue
            try:
                valid.append(self._read_variant(path, key))
            except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
                logger.warning('Invalid Cache / Re-encoding audio_md5={} variant={}: {}',
                               key['audio_md5'], path.name, exc)
        if not valid:
            return None
        # Newest valid result is selected, regardless of its encoding device.
        return max(valid, key=lambda item: (item[1]['created_ns'], item[1]['token_md5']))

    def _save_disk(self, key, array, metadata):
        parent = self._directory(key)
        for path in (parent.parent.parent, parent.parent, parent):
            if path.is_symlink():
                raise OSError('Symlinked token identity directory')
            path.mkdir(mode=0o700, exist_ok=True)
        target = parent / (metadata['token_md5'] + '-' + metadata['token_sha256'])
        if target.exists():
            try:
                self._read_variant(target, key)
                return False
            except (OSError, ValueError, TypeError, KeyError, EOFError):
                target.rename(parent / ('.invalid-' + uuid.uuid4().hex))
        temp = Path(tempfile.mkdtemp(prefix='.tmp-', dir=parent))
        try:
            for name, binary in [('tokens.npy', True), ('metadata.json', False)]:
                fd = os.open(temp/name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, 'wb' if binary else 'w', **({} if binary else {'encoding':'utf-8'})) as stream:
                    if binary:
                        np.save(stream, array, allow_pickle=False)
                    else:
                        json.dump(metadata, stream, sort_keys=True, ensure_ascii=False, indent=2)
                        stream.write('\n')
                    stream.flush();os.fsync(stream.fileno())
            temp.rename(target)
            if os.name != 'nt':
                dir_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                try:os.fsync(dir_fd)
                finally:os.close(dir_fd)
            return True
        finally:
            if temp.exists():shutil.rmtree(temp)

    def resolve(self, audio, encode, encoder_device='cpu', use_memory_cache=True,
                read_disk_cache=True, save_disk_cache=False, force_reencode=False):
        if not isinstance(audio, bytes) or not audio:
            raise ValueError('Reference audio bytes are empty')
        if encoder_device not in ('cpu', 'gpu'):
            raise ValueError('Unknown reference encoder device')
        key = self._key(audio)
        memory_key = tuple(key.values())
        start = time.perf_counter()
        with self.lock, self._disk_lock(key, save_disk_cache or (read_disk_cache and not force_reencode)) as disk:
            hit, source = None, 'encoding'
            if use_memory_cache and not force_reencode and memory_key in self.memory:
                hit = self.memory[memory_key]
                self.memory.move_to_end(memory_key)
                source = 'memory'
                logger.info('Memory Cache Hit audio_md5={}', key['audio_md5'])
            elif read_disk_cache and not force_reencode and disk:
                try:hit = self._read_disk(key)
                except OSError as exc:logger.warning('Invalid Cache / Re-encoding: {}', exc)
                if hit is not None:
                    source = 'disk'
                    logger.info('Disk Cache Hit audio_md5={} token_md5={} encoded_on={}',
                                key['audio_md5'], hit[1]['token_md5'], hit[1]['encoder_device'])
            encode_seconds = 0.0
            if hit is None:
                logger.info('Cache Miss audio_md5={} force_reencode={}', key['audio_md5'], force_reencode)
                logger.info('{} Encoding audio_md5={}', encoder_device.upper(), key['audio_md5'])
                t = time.perf_counter()
                array = self._array(encode(audio, encoder_device))
                encode_seconds = time.perf_counter()-t
                data = array.tobytes()
                metadata = dict(key, format_version=FORMAT_VERSION, dtype='int64-le',
                                shape=list(array.shape), token_limits=list(self.token_limits),
                                token_md5=hashlib.md5(data).hexdigest(),
                                token_sha256=hashlib.sha256(data).hexdigest(),
                                encoder_device=encoder_device, encoder_precision='fp32' if encoder_device=='cpu' else 'model-native',
                                created_ns=time.time_ns(), codec_identity=self.codec_identity)
            else:
                array, metadata = hit
            saved = False
            if save_disk_cache and disk:
                try:
                    saved = self._save_disk(key, array, metadata)
                    if saved:
                        logger.info('Disk Cache Saved audio_md5={} token_md5={} encoded_on={}',
                                    key['audio_md5'], metadata['token_md5'], metadata['encoder_device'])
                except (OSError, ValueError) as exc:
                    logger.warning('Token cache save failed; generated tokens remain usable: {}', exc)
            if use_memory_cache:
                immutable = array.copy();immutable.setflags(write=False)
                self.memory[memory_key] = (immutable, dict(metadata))
                self.memory.move_to_end(memory_key)
                while len(self.memory)>self.memory_entries:self.memory.popitem(last=False)
            event = {**key, 'token_md5':metadata['token_md5'], 'token_sha256':metadata['token_sha256'],
                     'source':source, 'requested_device':encoder_device, 'encoded_on':metadata['encoder_device'],
                     'encode_seconds':encode_seconds, 'resolve_seconds':time.perf_counter()-start,
                     'disk_saved':saved, 'force_reencode':force_reencode}
            return array.copy(), event
