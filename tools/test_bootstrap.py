import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('bootstrap', Path(__file__).with_name('bootstrap.py'))
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def test_anonymous_download_verifies_and_reuses_complete_file(self):
        payload = b'public release asset\x00\xff'
        expected = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        specification = dict(repository='owner/project', tag='release-v1')
        url = bootstrap.asset_url(specification, 'part-001')
        self.assertEqual(url, 'https://github.com/owner/project/releases/download/release-v1/part-001')
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'part-001'
            with patch.object(bootstrap, 'urlopen', return_value=io.BytesIO(payload)) as opening:
                bootstrap.download_asset(url, destination, expected)
                request = opening.call_args.args[0]
                self.assertIsNone(request.get_header('Authorization'))
                self.assertEqual(request.full_url, url)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_name('part-001.downloading').exists())
            with patch.object(bootstrap, 'urlopen', side_effect=AssertionError('Verified file was downloaded again')):
                bootstrap.download_asset(url, destination, expected)

    def test_bad_public_download_preserves_existing_file(self):
        payload = b'correct contents'
        expected = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        for invalid in (b'x' * len(payload), payload[:-1], payload + b'excess'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as folder:
                destination = Path(folder) / 'part-001'
                destination.write_bytes(b'previous file')
                with patch.object(bootstrap, 'urlopen', return_value=io.BytesIO(invalid)):
                    with self.assertRaises(ValueError):
                        bootstrap.download_asset('https://example.invalid/asset', destination, expected)
                self.assertEqual(destination.read_bytes(), b'previous file')
                self.assertFalse(destination.with_name('part-001.downloading').exists())

    def test_interrupted_public_download_removes_partial_file(self):
        payload = b'correct contents'
        expected = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'part-001'
            with patch.object(bootstrap, 'urlopen') as opening:
                opening.return_value.__enter__.return_value.read.side_effect = [payload[:4], OSError('connection lost')]
                with self.assertRaisesRegex(OSError, 'connection lost'):
                    bootstrap.download_asset('https://example.invalid/asset', destination, expected)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name('part-001.downloading').exists())

    def test_assembly_preserves_bytes_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parts = []
            for index, content in enumerate((b'first', b'\x00second\xff')):
                path = root / f'part{index}'
                path.write_bytes(content)
                parts.append(dict(name=path.name, bytes=len(content), sha256=bootstrap.sha(path)))
            combined = root / 'reference'
            combined.write_bytes(b'first\x00second\xff')
            manifest = dict(parts=parts, archive=dict(name='bundle.tar', bytes=combined.stat().st_size, sha256=bootstrap.sha(combined)))
            target = bootstrap.assemble(root, manifest)
            self.assertEqual(target.read_bytes(), combined.read_bytes())
            target.write_bytes(b'bad')
            (root / 'part1').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                bootstrap.assemble(root, manifest)

    def test_tar_rejects_escape_before_extracting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'unsafe.tar'
            with tarfile.open(archive, 'w') as handle:
                entry = tarfile.TarInfo('fight_fusion_linux/../../outside')
                entry.size = 3
                handle.addfile(entry, io.BytesIO(b'bad'))
            with self.assertRaises(ValueError):
                bootstrap.extract(archive, root / 'work')
            self.assertFalse((root / 'outside').exists())

    def test_tar_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'unsafe.tar'
            with tarfile.open(archive, 'w') as handle:
                entry = tarfile.TarInfo('fight_fusion_linux/link')
                entry.type = tarfile.SYMTYPE
                entry.linkname = '/etc/passwd'
                handle.addfile(entry)
            with self.assertRaises(ValueError):
                bootstrap.extract(archive, root / 'work')

    def test_missing_status_is_empty_without_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'absent'
            self.assertEqual(bootstrap.status(path), {'branches': {}})
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
