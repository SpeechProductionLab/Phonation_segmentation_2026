"""Run grouped five-fold parameter selection and held-out evaluation."""
import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

import numpy as np
import soundfile as sf

from evaluate import aggregate, evaluate
from methods import METHODS


# Small, explicit starting grids, not claims of optimal settings.
GRIDS = {
    'fixed_rms': {'threshold_db': [-50, -40, -30]},
    'adaptive_rms': {'margin_db': [6, 10, 14]},
    'hysteresis_rms': {'margin_db': [4, 8], 'hysteresis_db': [4, 8]},
    'energy_periodicity': {'margin_db': [6, 10], 'periodicity': [.35, .55]},
    'spectral_similarity': {'margin_db': [6, 10], 'similarity': [.75, .9]},
    'rvad_fast': {'vad_threshold': [.3, .4, .5]},
}
THRESHOLDS = (.25, .5, .75)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_annotations(path):
    recordings = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'audio_filename', 'start_seconds', 'end_seconds'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV requires columns: {sorted(required)}')
        for line, row in enumerate(reader, 2):
            name = row['audio_filename']
            if not name or Path(name).name != name or name in ('.', '..'):
                raise ValueError(f'CSV line {line}: expected a filename, not a path')
            a, b = float(row['start_seconds']), float(row['end_seconds'])
            if not np.isfinite([a, b]).all() or not 0 <= a < b:
                raise ValueError(f'CSV line {line}: invalid interval [{a}, {b}]')
            recordings.setdefault(name, []).append((a, b))
    if not recordings:
        raise ValueError('CSV contains no annotated attempts')
    for name, intervals in recordings.items():
        intervals.sort()
        if len(set(intervals)) != len(intervals):
            raise ValueError(f'{name}: duplicate annotation intervals')
    return recordings


def participant(name):
    stem = Path(name).stem
    if '_' not in stem or not stem.rsplit('_', 1)[1].isdigit():
        raise ValueError(f'Cannot infer participant from {name!r}; expected participant_number.wav')
    return stem.rsplit('_', 1)[0]


def get_folds(names, path, seed):
    names = sorted(names)
    groups = {name: participant(name) for name in names}
    if len(set(groups.values())) < 5:
        raise ValueError('Five-fold evaluation requires at least five distinct participants')
    if path.exists():
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(names) or {r['filename'] for r in rows} != set(names):
            raise ValueError('Saved folds do not match CSV filenames; use a new --folds path')
        assignments = {r['filename']: int(r['fold']) for r in rows}
        if set(assignments.values()) != set(range(1, 6)):
            raise ValueError('Saved folds must contain folds 1 through 5')
        for group in set(groups.values()):
            members = [n for n in names if groups[n] == group]
            if len({assignments[n] for n in members}) != 1:
                raise ValueError(f'Participant {group} occurs in multiple saved folds')
        if any(r['participant_id'] != groups[r['filename']] for r in rows):
            raise ValueError('Saved participant IDs disagree with filenames')
        return assignments, groups
    group_names = sorted(set(groups.values()))
    random.Random(seed).shuffle(group_names)
    # Balance recording counts, with seeded ordering to break equal-size ties.
    group_names.sort(key=lambda g: -sum(v == g for v in groups.values()))
    loads = [0] * 5
    assignment = {}
    for group in group_names:
        fold = min(range(5), key=lambda i: loads[i])
        members = [n for n in names if groups[n] == group]
        assignment.update({n: fold + 1 for n in members})
        loads[fold] += len(members)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, (dict(filename=n, participant_id=groups[n], fold=assignment[n]) for n in names))
    return assignment, groups


def configurations(method):
    grid = {**GRIDS[method], 'gap': [0., .025, .075], 'min_duration': [.030]}
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, default=Path('say_a_1_time_clean'))
    parser.add_argument('--csv', type=Path, default=Path('phonation-collection.csv'))
    parser.add_argument('--method', choices=['all', *METHODS], default='all')
    parser.add_argument('--folds', type=Path, default=Path('results/folds.csv'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--diagnostics', action='store_true', help='Save held-out frame traces as NPZ')
    args = parser.parse_args(argv)
    reference = read_annotations(args.csv)
    for name in reference:
        if not (args.audio_dir / name).is_file():
            raise ValueError(f'Missing annotated audio: {args.audio_dir / name}')
    folds, groups = get_folds(reference, args.folds, args.seed)
    selected = list(METHODS) if args.method == 'all' else [args.method]
    if 'rvad_fast' in selected:
        import rVADfast  # Fail clearly before starting an expensive experiment.
    output = args.output or Path('results') / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'folds.csv', (dict(filename=n, participant_id=groups[n], fold=folds[n])
                                     for n in sorted(reference)))
    manifest = dict(csv=str(args.csv.resolve()), audio_dir=str(args.audio_dir.resolve()),
                    csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                    seed=args.seed, folds=str(args.folds.resolve()), thresholds=THRESHOLDS,
                    primary_metric='pooled attempt F1 at IoU >= 0.50',
                    python=sys.version, versions={p: importlib.metadata.version(p)
                    for p in ('numpy', 'scipy', 'soundfile')}, methods=selected,
                    grids={m: configurations(m) for m in selected}, audio_sha256={},
                    source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                   for n in ('run.py', 'methods.py', 'evaluate.py')})
    if 'rvad_fast' in selected:
        manifest['versions']['rVADfast'] = importlib.metadata.version('rVADfast')
    def save_manifest():
        (output / 'config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_manifest()
    metrics, parameters, predictions, summary = [], [], [], []
    for method in selected:
        configs = configurations(method)
        cache = {}
        for index, name in enumerate(sorted(reference), 1):
            audio_path = args.audio_dir / name
            audio, sr = sf.read(audio_path, dtype='float64', always_2d=True)
            audio = audio.mean(axis=1)
            if not len(audio) or not np.isfinite(audio).all() or sr < 8000:
                raise ValueError(f'{name}: expected finite nonempty audio with sample rate >= 8000 Hz')
            duration = len(audio) / sr
            if any(b > duration + 1e-6 for _, b in reference[name]):
                raise ValueError(f'{name}: annotation extends past audio duration {duration:.6f}')
            manifest['audio_sha256'][name] = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            cache[name] = [METHODS[method](audio, sr, **p)[0] for p in configs]
            print(f'{method}: {index}/{len(reference)} recordings', flush=True)
        tuning = {name: [evaluate(reference[name], pred, .5) for pred in candidates]
                  for name, candidates in cache.items()}
        method_rows = []
        for fold in range(1, 6):
            train = [name for name in reference if folds[name] != fold]
            values = [aggregate([tuning[n][i] for n in train])['f1'] for i in range(len(configs))]
            # Deterministic ties retain the first configuration in the documented grid.
            best = max(range(len(values)), key=lambda i: values[i])
            parameters.append(dict(method=method, fold=fold, tuning_f1_iou_050=values[best],
                                   parameters=json.dumps(configs[best], sort_keys=True)))
            for name in sorted(n for n in reference if folds[n] == fold):
                pred = cache[name][best]
                for attempt, (a, b) in enumerate(pred, 1):
                    predictions.append(dict(method=method, fold=fold, audio_filename=name,
                                            attempt=attempt, start_seconds=a, end_seconds=b))
                for threshold in THRESHOLDS:
                    row = dict(level='recording', method=method, fold=fold, filename=name,
                               iou_threshold=threshold, **evaluate(reference[name], pred, threshold))
                    metrics.append(row)
                    method_rows.append(row)
                if args.diagnostics:
                    audio, sr = sf.read(args.audio_dir / name, always_2d=True)
                    _, traces = METHODS[method](audio.mean(axis=1), sr, **configs[best])
                    folder = output / 'diagnostics' / method
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / (name + '.npz'), **traces)
        for threshold in THRESHOLDS:
            rows = [r for r in method_rows if r['iou_threshold'] == threshold]
            fold_values = []
            for fold in range(1, 6):
                stats = aggregate([r for r in rows if r['fold'] == fold])
                fold_values.append(stats)
                metrics.append(dict(level='fold', method=method, fold=fold, filename='',
                                    iou_threshold=threshold, **stats))
            summary.append(dict(method=method, iou_threshold=threshold, **aggregate(rows),
                                **{f'{key}_fold_mean': float(np.mean([s[key] for s in fold_values]))
                                   for key in ('precision', 'recall', 'f1')},
                                **{f'{key}_fold_std': float(np.std([s[key] for s in fold_values], ddof=1))
                                   for key in ('precision', 'recall', 'f1')}))
        # Persist completed methods, even if a later detector fails.
        write_csv(output / 'parameters.csv', parameters)
        write_csv(output / 'predictions.csv', predictions,
                  ['method', 'fold', 'audio_filename', 'attempt', 'start_seconds', 'end_seconds'])
        fields = list(dict.fromkeys(k for row in metrics for k in row))
        write_csv(output / 'metrics.csv', metrics, fields)
        write_csv(output / 'summary.csv', summary)
        save_manifest()
    print(f'Completed. Results: {output.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit(f'Error: {error}') from error
