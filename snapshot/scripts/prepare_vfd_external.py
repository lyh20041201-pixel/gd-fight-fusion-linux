"""Audit official VFD labels, quarantine conflicts, and seal the external set."""
from collections import Counter
from pathlib import Path
import hashlib
import json
from prepare_rebuild_manifests import video_info

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'datasets/vfd2000/new_youtube'
    output = ROOT / 'datasets/video_events/rebuild'
    by_hash = {}
    errors = []
    for label, filename in [(1, 'fight_vid.txt'), (0, 'nofight_vid.txt')]:
        for line in (source / filename).read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            paths = [source / folder / fields[0] for folder in ('train_data', 'test_data')]
            paths = [p for p in paths if p.is_file()]
            if len(paths) != 1:
                errors.append(dict(filename=fields[0],error='Expected one source video'))
                continue
            path = paths[0]
            try:
                info = video_info(path)
            except Exception as exc:
                errors.append(dict(path=str(path),error=str(exc)))
                continue
            row = dict(path=str(path),label=label,group=fields[-1],split='external_test',
                       category=fields[1],duration=info['duration'],sha256=info['sha256'])
            by_hash.setdefault(row['sha256'], []).append(row)
    conflicts = {sha: rows for sha, rows in by_hash.items() if len({r['label'] for r in rows}) > 1}
    rows = [items[0] for sha, items in by_hash.items() if sha not in conflicts]
    exclusions_path=output / 'vfd_source_exclusions.json'
    exclusions=json.loads(exclusions_path.read_text()) if exclusions_path.exists() else dict(groups=[])
    rows=[row for row in rows if row['group'] not in exclusions['groups']]
    tnue = json.loads((output / 'tnue_audit.json').read_text())
    tnue_hashes = {r['sha256'] for r in tnue['videos']}
    overlap = sorted(set(by_hash) & tnue_hashes)
    if overlap:
        raise ValueError('Exact TNUE/VFD overlap: ' + str(overlap))
    manifest = dict(dataset='VFD-2000',labels=['normal','fight'],rows=rows,errors=errors,
                    unique_samples=len(rows),class_counts=dict(Counter(r['label'] for r in rows)),
                    duplicates=[items for items in by_hash.values() if len(items)>1],
                    conflicting_label_hashes_excluded=sorted(conflicts),
                    protocol='Entire available labelled release reserved for external evaluation; no training or threshold selection',
                    exact_tnue_overlap=overlap,
                    perceptual_source_exclusions=exclusions,
                    overlap_limitation='Exact hashes only; edited copies of the same incident require visual/source review')
    (output / 'vfd_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(dict(samples=len(rows),errors=len(errors),conflicts=len(conflicts),classes=manifest['class_counts']))


if __name__ == '__main__':
    main()
