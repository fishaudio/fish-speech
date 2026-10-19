import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from huggingface_hub import hf_hub_download
from tools.migrate_hf_cache import migrate


def metadata(data, lfs=False, name='model.pth'):
    digest = hashlib.sha256(data).hexdigest() if lfs else hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    return NS(rfilename=name, size=len(data), blob_id=digest,
              lfs=NS(sha256=digest) if lfs else None)


def run(source, cache, files, **kwargs):
    with patch('tools.migrate_hf_cache.HfApi') as api:
        api.return_value.model_info.return_value = NS(sha='a' * 40, siblings=files)
        return migrate(source, 'test/model', cache_dir=cache, **kwargs)


def test_copy_is_reusable_by_hub_without_network(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'model.pth').write_bytes(b'weights')
    cache = tmp_path / 'cache'
    run(source, cache, [metadata(b'weights', lfs=True)])
    cached = Path(hf_hub_download('test/model', 'model.pth', cache_dir=cache, local_files_only=True))
    assert cached.read_bytes() == b'weights'
    assert (source / 'model.pth').exists()
    run(source, cache, [metadata(b'weights', lfs=True)], move=True)
    assert not (source / 'model.pth').exists()
    assert cached.read_bytes() == b'weights'


def test_mismatch_preserves_every_source_file(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'config.json').write_bytes(b'good')
    (source / 'model.pth').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='mismatch'):
        run(source, tmp_path / 'cache', [metadata(b'good', name='config.json'), metadata(b'right', lfs=True)], move=True)
    assert (source / 'config.json').exists()
    assert (source / 'model.pth').read_bytes() == b'wrong'


def test_dry_run_and_unknown_files(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'model.pth').write_bytes(b'weights')
    (source / 'notes.txt').write_text('keep')
    cache = tmp_path / 'cache'
    run(source, cache, [metadata(b'weights')], dry_run=True, move=True)
    assert not cache.exists()
    run(source, cache, [metadata(b'weights')], move=True)
    assert (source / 'notes.txt').read_text() == 'keep'


def test_corrupt_existing_blob_never_deletes_source(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'model.pth').write_bytes(b'weights')
    cache = tmp_path / 'cache'; info = metadata(b'weights', lfs=True)
    blob = cache / 'models--test--model' / 'blobs' / info.lfs.sha256
    blob.parent.mkdir(parents=True); blob.write_bytes(b'bad')
    with pytest.raises(ValueError, match='mismatch'):
        run(source, cache, [info], move=True)
    assert (source / 'model.pth').exists()


def test_reject_source_inside_cache(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, tmp_path, [])


def test_rejects_symlink_sources_and_traversal(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    outside = tmp_path / 'outside'; outside.write_bytes(b'weights')
    (source / 'model.pth').symlink_to(outside)
    with pytest.raises(ValueError, match='regular file'):
        run(source, tmp_path / 'cache', [metadata(b'weights')], move=True)
    assert outside.exists()
    with pytest.raises(ValueError, match='Unsafe'):
        run(source, tmp_path / 'cache', [metadata(b'weights', name='../outside')])


def test_failed_later_cache_file_preserves_all_sources(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'config.json').write_bytes(b'config')
    (source / 'model.pth').write_bytes(b'weights')
    info = metadata(b'weights')
    cache = tmp_path / 'cache'
    blob = cache / 'models--test--model' / 'blobs' / info.blob_id
    blob.parent.mkdir(parents=True); blob.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='mismatch'):
        run(source, cache, [metadata(b'config', name='config.json'), info], move=True)
    assert (source / 'config.json').read_bytes() == b'config'
    assert (source / 'model.pth').read_bytes() == b'weights'
