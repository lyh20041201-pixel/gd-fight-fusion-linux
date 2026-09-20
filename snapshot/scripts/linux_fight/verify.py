"""Portable, standard-library-only verification for an exported training bundle."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes bundle: ' + name)
    return path


def verify(root=ROOT, full=False):
    inventory = read(root / 'migration/inventory.json')
    checked = 0
    for entry in inventory['files']:
        if not full and not entry['immutable']:
            continue
        path = contained(root, entry['path'])
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Missing or changed bundle file: ' + entry['path'])
        checked += 1
        if checked % 500 == 0:
            print('VERIFIED', checked, flush=True)
    manifest_path = root / 'results/fight_fusion_v1/data/manifest.json'
    manifest = read(manifest_path)
    seal = read(manifest_path.with_suffix('.seal.json'))
    if sha(manifest_path) != seal['manifest_sha256']:
        raise ValueError('Manifest seal mismatch')
    for row in manifest['rows']:
        for key, digest_key in [('path', 'sha256'), ('annotation_path', 'annotation_sha256')]:
            if row.get(key):
                path = contained(root, row[key])
                # The full inventory already hashes all videos and annotations.
                if not path.is_file():
                    raise ValueError('Missing source: ' + row[key])
    return dict(status='verified', files=checked, source_rows=len(manifest['rows']), full=full)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Verify original snapshot, before first training only')
    args = parser.parse_args()
    print(json.dumps(verify(full=args.full), indent=2))
