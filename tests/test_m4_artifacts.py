import hashlib
import zipfile

import pytest

from m4_whisper.download import install_archive


def build_zip(path, names):
    with zipfile.ZipFile(path, 'w') as archive:
        for name in names:
            entry = zipfile.ZipInfo('placeholder')
            entry.filename = name
            archive.writestr(entry, b'model-test')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_is_verified_then_installed_without_overwriting(tmp_path):
    source = tmp_path / 'model.zip'
    digest = build_zip(source, ['model.bin', 'config.json'])
    target = tmp_path / 'installed'
    install_archive(source, target, digest)
    assert (target / 'model.bin').read_bytes() == b'model-test'
    with pytest.raises(FileExistsError):
        install_archive(source, target, digest)


def test_bad_checksum_never_installs_model(tmp_path):
    source = tmp_path / 'model.zip'
    build_zip(source, ['model.bin'])
    with pytest.raises(ValueError, match='SHA-256'):
        install_archive(source, tmp_path / 'target', '0' * 64)
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/drive', 'a\\escape'])
def test_archive_path_escape_is_rejected(tmp_path, name):
    source = tmp_path / 'model.zip'
    digest = build_zip(source, [name])
    with pytest.raises(ValueError, match='path'):
        install_archive(source, tmp_path / 'target', digest)
    assert not (tmp_path / 'target').exists()
