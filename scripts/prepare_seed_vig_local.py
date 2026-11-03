"""Prepare a local SEED-VIG release and audit MAT pairs without inventing labels/IDs.

Run with the project's Python environment (SciPy required). Original ZIP is read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import loadmat, whosmat


FEATURES = {
    'EEG_Feature_5Bands': ('psd_movingAve', 1, (17, 885, 5)),
    'Forehead_EEG/EEG_Feature_5Bands': ('psd_movingAve', 1, (4, 885, 5)),
    'EOG_Feature': ('features_table_ica', 0, (885, 36)),
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(archive: Path, destination: Path, output: Path):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        selected = []
        for member in z.infolist():
            if member.is_dir():
                continue
            name = member.filename
            wanted = (name in {'Readme_English.txt', 'Readme_Chinese.txt', 'channel_62_pos.locs'}
                      or any(name.startswith(prefix + '/') for prefix in FEATURES)
                      or name.startswith('perclos_labels/'))
            if not wanted:
                continue
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Unsafe archive member: {name}')
            selected.append((member, target))
        if not selected:
            raise ValueError('No supported SEED-VIG release files in ZIP')
        for member, target in selected:
            data = z.read(member)  # zipfile validates this member's CRC.
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f'Existing local file differs; refusing overwrite: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(data)
    labels = {p.stem: p for p in (destination / 'perclos_labels').glob('*.mat')}
    if not labels:
        raise ValueError('No PERCLOS labels found')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    layouts = {}
    for folder, (key, axis, shape) in FEATURES.items():
        files = {p.stem: p for p in (destination / folder).glob('*.mat')}
        if files.keys() != labels.keys():
            raise ValueError(f'{folder}: feature/label session sets differ')
        sessions = []
        for stem in sorted(files):
            feature_path, label_path = files[stem], labels[stem]
            features, lab = loadmat(feature_path), loadmat(label_path)
            values = np.asarray(features[key], dtype=float)
            y = np.asarray(lab['perclos'], dtype=float).reshape(-1)
            if values.shape != shape or len(y) != values.shape[axis]:
                raise ValueError(f'{stem}: unexpected shape or row mismatch')
            if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
                raise ValueError(f'{stem}: invalid PERCLOS')
            for var, value in features.items():
                if not var.startswith('__') and (value.shape != shape or not np.isfinite(value).all()):
                    raise ValueError(f'{stem}/{var}: wrong shape/non-finite feature')
            prefix = stem.split('_', 1)[0]
            if not prefix.isdigit():
                raise ValueError(f'{stem}: expected numeric release prefix')
            record = dict(session=stem, filename_prefix=prefix, modality=folder,
                          feature_shape=list(values.shape), feature_key=key, sample_axis=axis,
                          label_key='perclos', rows=len(y), label_min=float(y.min()),
                          label_max=float(y.max()), label_mean=float(y.mean()),
                          feature_sha256=digest(feature_path), label_sha256=digest(label_path),
                          variables=[dict(key=k, shape=list(s), type=t) for k, s, t in whosmat(feature_path)])
            records.append(record)
            sessions.append(dict(subject_id=f'REPLACE_WITH_VERIFIED_PERSON_FOR_PREFIX_{int(prefix):02d}',
                                 candidate_group=f'seed_prefix_{int(prefix):02d}',
                                 session_id=stem, feature_file=str(feature_path), feature_key=key,
                                 sample_axis=axis, label_file=str(label_path), label_key='perclos',
                                 alignment_verified=False,
                                 pairing_evidence='Identical release filename; README describes 885 ordered rows; numeric length verified',
                                 person_identity_status='UNVERIFIED: do not count session filenames as people'))
        manifest = dict(dataset='seed-vig', source_is_synthetic=False,
                        status='DRAFT_REQUIRES_PERSON_ID_AND_ALIGNMENT_REVIEW', sessions=sessions)
        short = {'EEG_Feature_5Bands': 'eeg_5bands',
                 'Forehead_EEG/EEG_Feature_5Bands': 'forehead_5bands', 'EOG_Feature': 'eog_ica'}[folder]
        (output / f'manifest_{short}.draft.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        layouts[short] = dict(sessions=len(sessions), rows=sum(r['rows'] for r in records if r['modality']==folder),
                             feature_shape=list(shape), flattened_features=int(np.prod(shape)/885))
    prefixes = Counter(stem.split('_', 1)[0] for stem in labels)
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), archive=str(archive.resolve()),
                  archive_size_bytes=archive.stat().st_size, destination=str(destination),
                  extracted_files=len(selected), extracted_bytes=sum(m.file_size for m, _ in selected),
                  label_sessions=len(labels), filename_prefix_groups=len(prefixes),
                  repeated_prefixes={k:v for k,v in prefixes.items() if v>1}, layouts=layouts,
                  person_identity_verified=False, training_run=False, thresholds_chosen=False,
                  warning='23 experiments are not automatically 23 people. Prefix grouping is provisional. No KSS conversion.',
                  records=records)
    (output / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),encoding='utf-8')
    (destination / '_n16_inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--destination', type=Path, default=Path('data/external/seed-vig'))
    p.add_argument('--output', type=Path, default=Path('output/seed_vig_local_20261028'))
    args=p.parse_args()
    prepare(args.archive,args.destination,args.output)
