import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('bootstrap', Path(__file__).with_name('bootstrap.py'))
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
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
