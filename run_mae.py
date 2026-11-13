"""MAE-only parameter search, grouped development folds, and separate hidden testing."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from endpoint_methods import EndpointRecording, GRIDS, SHARED_START
from evaluate import aggregate, evaluate
from methods import fixed_rms, adaptive_rms, hysteresis_rms, energy_periodicity, local_rms_energy_drop, rvad_fast
from comparison_methods import (COMPARISON_GRIDS, ComparisonRecording, require_dependencies,
                                praat_intensity, librosa_split)
from rms_variants import (RMS_VARIANT_GRIDS, RMSRecording, percentile_rms, background_rms, phonation_rms)
from run import get_folds, participant, read_annotations, write_csv


# Add a detector and its grid here. Detectors return (intervals_seconds, diagnostics).
# Entry order also defines deterministic ties; no F1/recall tie-breaking is used.
REGISTRY = {
    'fixed_rms': (fixed_rms, {'threshold_db': [-50, -40, -30]}),
    'adaptive_rms': (adaptive_rms, COMPARISON_GRIDS['adaptive_rms']),
    'local_rms_energy_drop': (local_rms_energy_drop, {'margin_db': [6, 10, 14], 'drop_db': [6, 9, 12]}),
    'hysteresis_rms': (hysteresis_rms, COMPARISON_GRIDS['hysteresis_rms']),
    'rvad_fast': (rvad_fast, COMPARISON_GRIDS['rvad_fast']),
    'praat_intensity': (praat_intensity, COMPARISON_GRIDS['praat_intensity']),
    'librosa_split': (librosa_split, COMPARISON_GRIDS['librosa_split']),
    'energy_periodicity': (energy_periodicity, {'margin_db': [6, 10], 'periodicity': [.35, .55]}),
}
REGISTRY.update({name: (function, RMS_VARIANT_GRIDS[name]) for name, function in
                 [('percentile_rms', percentile_rms), ('background_rms', background_rms),
                  ('phonation_rms', phonation_rms)]})
COMPARISON_FUNCTIONS = {name: REGISTRY[name][0] for name in COMPARISON_GRIDS}
SUM_KEYS = ('tp', 'fp', 'fn', 'matched_iou_sum', 'onset_error_sum_seconds',
            'offset_error_sum_seconds', 'split_reference_count', 'merged_prediction_count',
            'short_reference_count', 'short_matched_count', 'late_end_count', 'early_end_count')


def summed_score(values):
    return {k: float(v) if 'sum' in k else int(v) for k, v in zip(SUM_KEYS, values)}


def predict(name, audio, sr, params, comparison, endpoint):
    if name in RMS_VARIANT_GRIDS:
        if not hasattr(comparison, 'rms_variants'):
            comparison.rms_variants = RMSRecording(audio, sr)
        return comparison.rms_variants.predict(name, params)
    if name in GRIDS:
        return endpoint.refine(name, params)
    if name in COMPARISON_FUNCTIONS and REGISTRY[name][0] is COMPARISON_FUNCTIONS[name]:
        return comparison.predict(name, params)
    return REGISTRY[name][0](audio, sr, **params)[0]


IOU_THRESHOLD = .25
UNAVAILABLE = 'MAE unavailable'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('methods.py', 'evaluate.py', 'run.py', 'run_mae.py', 'endpoint_methods.py', 'comparison_methods.py', 'rms_variants.py')}


def split_data(names, path, seed=42):
    """Subset-sum holdout nearest 20% by recording count; never split a participant."""
    names = sorted(names)
    groups = {n: participant(n) for n in names}
    if path.exists():
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        required = {'filename', 'participant_id', 'set', 'fold'}
        if not rows or not required.issubset(rows[0]):
            raise ValueError('Invalid MAE split file schema')
        if len(rows) != len(names) or {r['filename'] for r in rows} != set(names):
            raise ValueError('Saved splits differ from CSV filenames; use a new --splits path')
    else:
        ids = sorted(set(groups.values()))
        if len(ids) < 6:
            raise ValueError('Need at least six participants: five development and one hidden')
        random.Random(seed).shuffle(ids)
        counts = {g: sum(v == g for v in groups.values()) for g in ids}
        # For each achievable size, keep a subset using the fewest participants,
        # ensuring at least five participants remain available for development.
        subsets = {0: ()}
        for g in ids:
            for size, subset in list(subsets.items()):
                candidate = subset + (g,)
                total = size + counts[g]
                if len(candidate) <= len(ids) - 5:
                    if total not in subsets or len(candidate) < len(subsets[total]):
                        subsets[total] = candidate
        target = .2 * len(names)
        size = min((n for n in subsets if n > 0), key=lambda n: (abs(n - target), n))
        hidden = set(subsets[size])
        dev = [n for n in names if groups[n] not in hidden]
        # Reuse existing balancing logic, but save everything in one MAE split CSV.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            folds, _ = get_folds(dev, Path(temp) / 'folds.csv', seed)
        rows = [dict(filename=n, participant_id=groups[n],
                     set='hidden' if groups[n] in hidden else 'development',
                     fold='' if groups[n] in hidden else str(folds[n])) for n in names]
    seen = {}
    for row in rows:
        n, group = row['filename'], row['participant_id']
        if group != groups[n]:
            raise ValueError('Participant IDs in saved splits disagree with filenames')
        if row['set'] == 'hidden' and row['fold'] == '':
            assignment = ('hidden', '')
        elif row['set'] == 'development' and row['fold'] in ('1', '2', '3', '4', '5'):
            assignment = ('development', row['fold'])
        else:
            raise ValueError('Invalid set/fold assignment in saved splits')
        if group in seen and seen[group] != assignment:
            raise ValueError(f'Participant {group} crosses sets or folds')
        seen[group] = assignment
    if not any(r['set'] == 'hidden' for r in rows):
        raise ValueError('Saved splits have no hidden recordings')
    if {r['fold'] for r in rows if r['set'] == 'development'} != set('12345'):
        raise ValueError('Development split must contain all five folds')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, rows)
    return sorted(rows, key=lambda r: r['filename'])


def mae_summary(rows):
    """Pool by matched attempt, giving start/end equal weight, in milliseconds."""
    stats = aggregate(rows)
    n = stats['tp']
    start = 1000 * stats['onset_error_sum_seconds'] / n if n else UNAVAILABLE
    end = 1000 * stats['offset_error_sum_seconds'] / n if n else UNAVAILABLE
    return dict(matched_attempts=n, missed_attempts=stats['fn'], extra_attempts=stats['fp'],
                start_mae_ms=start, end_mae_ms=end,
                end_late_over_100ms_pct=100 * sum(r['late_end_count'] for r in rows) / n if n else UNAVAILABLE,
                end_early_over_100ms_pct=100 * sum(r['early_end_count'] for r in rows) / n if n else UNAVAILABLE,
                combined_mae_ms=(start + end) / 2 if n else UNAVAILABLE,
                reference_attempts=n + stats['fn'], precision=stats['precision'],
                recall=stats['recall'], f1=stats['f1'])


def eligible_references(name, reference):
    """Apply the experiment-specific reference duration without changing splits."""
    minimum = 1.5 if name in RMS_VARIANT_GRIDS else 2.0 if name in COMPARISON_GRIDS else 0.
    return [(a, b) for a, b in reference if b - a >= minimum - 1e-9]


def bootstrap_ci(rows, stats, seed=42, resamples=2000):
    """Resample participant clusters, then recompute pooled matched boundary MAE."""
    groups = {}
    for row, score in zip(rows, stats):
        group = row['participant_id']
        groups.setdefault(group, np.zeros(3))
        groups[group] += [score['tp'], score['onset_error_sum_seconds'],
                          score['offset_error_sum_seconds']]
    result = dict(participants=len(groups), bootstrap_resamples=resamples,
                  bootstrap_valid_resamples=0, bootstrap_undefined_resamples=0,
                  ci_status='insufficient participants (need at least two)')
    for metric in ('start', 'end', 'combined'):
        for bound in ('low', 'high'):
            result[f'{metric}_mae_ci95_{bound}_ms'] = UNAVAILABLE
    if not groups:
        result.update(bootstrap_undefined_resamples=resamples, ci_status='no eligible participants')
        return result
    values = np.array([groups[g] for g in sorted(groups)])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        n, start, end = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        if n:
            estimates.append([1000 * start / n, 1000 * end / n, 500 * (start + end) / n])
    result.update(bootstrap_valid_resamples=len(estimates),
                  bootstrap_undefined_resamples=resamples - len(estimates))
    if len(groups) < 2:
        return result
    if len(estimates) < 2:
        result['ci_status'] = 'insufficient valid resamples'
        return result
    bounds = np.percentile(estimates, [2.5, 97.5], axis=0)
    for i, metric in enumerate(('start', 'end', 'combined')):
        result[f'{metric}_mae_ci95_low_ms'] = float(bounds[0, i])
        result[f'{metric}_mae_ci95_high_ms'] = float(bounds[1, i])
    result['ci_status'] = 'conditional on matched resamples' if len(estimates) < resamples else 'ok'
    return result


def choose_candidate(summaries, metric='combined_mae_ms'):
    eligible = [i for i, s in enumerate(summaries) if s['matched_attempts'] > 0]
    if not eligible:
        return None
    return min(eligible, key=lambda i: summaries[i][metric])


def load_audio(path, reference):
    audio, sr = sf.read(path, dtype='float64', always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or sr < 8000 or not np.isfinite(audio).all():
        raise ValueError(f'{path.name}: need finite nonempty audio, sample rate >= 8000 Hz')
    if any(b > len(audio) / sr + 1e-6 for _, b in reference):
        raise ValueError(f'{path.name}: annotation exceeds recording duration')
    return audio, sr


def candidates(grid):
    grid = {'gap': [0., .025, .075], 'min_duration': [.030], **grid}
    # Adjacent candidates reuse a raw detection across the ten merge gaps.
    grid = {**{k: v for k, v in grid.items() if k not in ('gap', 'min_duration')},
            'min_duration': grid['min_duration'], 'gap': grid['gap']}
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def stream_csv(path, rows):
    """Append one method's rows without retaining the full search/fold export."""
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        if not exists:
            writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['develop', 'test'])
    parser.add_argument('--audio-dir', type=Path, default=Path('say_a_1_time_clean'))
    parser.add_argument('--csv', type=Path, default=Path('phonation-collection.csv'))
    parser.add_argument('--splits', type=Path, default=Path('results/mae_splits.csv'))
    parser.add_argument('--method', choices=['all', 'comparison', 'rms_variants', 'ten_methods', *REGISTRY, *GRIDS], default='all')
    parser.add_argument('--selection', type=Path, help='Frozen selection.json from a development run')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.stage == 'test' and (not args.selection or not args.splits.exists()):
        raise ValueError('Hidden testing requires --selection and the existing --splits file')
    annotations = read_annotations(args.csv)
    splits = split_data(annotations, args.splits, args.seed)
    dev_names = [r['filename'] for r in splits if r['set'] == 'development']
    dev_hash = digest({n: annotations[n] for n in dev_names})
    sources = source_hashes()
    selected = list(REGISTRY) if args.method == 'all' else [args.method]
    if args.method == 'rms_variants':
        selected = list(RMS_VARIANT_GRIDS)
    if args.method == 'comparison':
        selected = list(COMPARISON_GRIDS)
    if args.method == 'ten_methods':
        selected = list(GRIDS)
    frozen = None
    if args.stage == 'test':
        frozen = json.loads(args.selection.read_text())
        if frozen.get('split_hash') != digest(splits) or frozen.get('development_annotations_hash') != dev_hash:
            raise ValueError('Frozen selection does not match these splits/development annotations')
        if frozen.get('source_hashes') != sources or frozen.get('iou_threshold') != IOU_THRESHOLD:
            raise ValueError('Code or matching policy changed since parameter selection; freeze a new development run')
        selected = list(frozen['methods']) if args.method == 'all' else selected
        if args.method == 'ten_methods':
            selected = [name for name in GRIDS if name in frozen['methods']]
        if not selected:
            raise ValueError('Frozen selection has no eligible methods to test')
        if any(m not in frozen['methods'] or (m not in REGISTRY and m not in GRIDS) for m in selected):
            raise ValueError('Requested method has no eligible frozen parameters')
    require_dependencies(selected)
    if any(name in RMS_VARIANT_GRIDS for name in selected):
        import librosa  # noqa: F401
    active_rows = [r for r in splits if r['set'] == ('hidden' if args.stage == 'test' else 'development')]
    for row in active_rows:
        if any(eligible_references(name, annotations[row['filename']]) for name in selected) and not (args.audio_dir / row['filename']).is_file():
            raise ValueError(f'Missing audio: {args.audio_dir / row["filename"]}')
    output = args.output or Path('results') / datetime.now(timezone.utc).strftime(f'mae_{args.stage}_%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'splits.csv', splits)
    print(f'{len(dev_names)} development / {len(splits) - len(dev_names)} hidden recordings', flush=True)
    artifact = dict(split_hash=digest(splits), development_annotations_hash=dev_hash,
                    source_hashes=sources, iou_threshold=IOU_THRESHOLD, methods={})
    config = dict(stage=args.stage, csv=str(args.csv.resolve()), audio_dir=str(args.audio_dir.resolve()),
                  selection=str(args.selection.resolve()) if args.selection else None,
                  seed=args.seed, source_hashes=sources, split_hash=digest(splits),
                  primary_metric='pooled matched combined boundary MAE in milliseconds',
                  iou_threshold=IOU_THRESHOLD, versions={p: importlib.metadata.version(p)
                  for p in ('numpy', 'scipy', 'soundfile')}, audio_sha256={})
    record_metrics, summaries, predictions = [], [], []
    ci_rows = []
    method_rows = {name: [r for r in active_rows if eligible_references(name, annotations[r['filename']])]
                   for name in selected}
    config['reference_min_duration_seconds'] = {name: (1.5 if name in RMS_VARIANT_GRIDS else 2.0)
        for name in selected if name in COMPARISON_GRIDS or name in RMS_VARIANT_GRIDS}
    for name, package in [('rvad_fast', 'rVADfast'), ('praat_intensity', 'praat-parselmouth'),
                          ('librosa_split', 'librosa')]:
        if name in selected:
            config['versions'][package] = importlib.metadata.version(package)
    if any(name in RMS_VARIANT_GRIDS for name in selected):
        config['versions']['librosa'] = importlib.metadata.version('librosa')
        config['rms_variants_policy'] = dict(framing='centered', statistics='linear RMS; population SD',
            reference_percentile_domain='linear RMS', minimum_duration=1.5,
            incomplete_reference='reject', rearm='background quiet longer than gap')
    config['bootstrap'] = dict(resamples=2000, seed=args.seed, unit='participant', confidence=.95,
                               interpretation='final held-out' if frozen else 'exploratory after selection')
    config['eligibility_counts'] = {name: dict(included_clips=len(rows),
        excluded_clips=len(active_rows) - len(rows)) for name, rows in method_rows.items()}
    searches = {}
    all_stats = {}
    frozen_records = {}
    selection_metrics = {}
    for name in selected:
        selection_metrics[name] = 'end_mae_ms' if name in GRIDS or name in RMS_VARIANT_GRIDS else 'combined_mae_ms'
        if frozen:
            params = [frozen['methods'][name]['parameters']]
        elif name in GRIDS:
            grid = GRIDS[name]
            params = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
        else:
            params = candidates(REGISTRY[name][1])
        searches[name] = params
        all_stats[name] = np.zeros((len(params), 1 if frozen else 5, len(SUM_KEYS)))
        frozen_records[name] = []
    config.update(candidate_parameters=searches, selection_metrics=selection_metrics)
    if any(name in GRIDS for name in selected):
        config['shared_adaptive_rms'] = SHARED_START
        if all(name in GRIDS for name in selected):
            config['primary_metric'] = 'pooled matched end MAE in milliseconds'
    if all(name in GRIDS or name in RMS_VARIANT_GRIDS for name in selected):
        config['primary_metric'] = 'pooled matched end MAE in milliseconds'
    for row in active_rows:
        filename = row['filename']
        if not any(eligible_references(name, annotations[filename]) for name in selected):
            continue
        path = args.audio_dir / filename
        audio, sr = load_audio(path, annotations[filename])
        config['audio_sha256'][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        recording = EndpointRecording(audio, sr) if any(m in GRIDS for m in selected) else None
        comparison = ComparisonRecording(audio, sr)
        for name, params in searches.items():
            reference = eligible_references(name, annotations[filename])
            if not reference:
                continue
            score_cache = {}
            for i, p in enumerate(params):
                pred = predict(name, audio, sr, p, comparison, recording)
                if any(not np.isfinite([a, b]).all() or not 0 <= a < b <= len(audio) / sr + 1e-6
                       for a, b in pred):
                    raise ValueError(f'{name}: invalid predicted interval for {filename}')
                key = tuple(pred)
                if key not in score_cache:
                    if len(score_cache) >= 512:
                        score_cache.clear()
                    score_cache[key] = evaluate(reference, pred, IOU_THRESHOLD)
                score = score_cache[key]
                fold_index = 0 if frozen else int(row['fold']) - 1
                all_stats[name][i, fold_index] += [score[k] for k in SUM_KEYS]
                if frozen:
                    frozen_records[name].append((score, pred))
            print(f'{args.stage}: {name}: {filename}', flush=True)
    for name, params in searches.items():
        stats = all_stats.pop(name)
        rows = method_rows[name]
        pooled = [mae_summary([summed_score(values)]) for values in stats.sum(axis=1)]
        best = choose_candidate(pooled, selection_metrics[name])
        if args.stage == 'develop':
            stream_csv(output / 'parameter_search.csv',
                (dict(method=name, candidate=i, parameters=json.dumps(p), **pooled[i])
                 for i, p in enumerate(params)))
            stream_csv(output / 'fold_metrics.csv',
                (dict(method=name, candidate=i, fold=fold,
                      **mae_summary([summed_score(stats[i, fold - 1])]))
                 for i in range(len(params)) for fold in range(1, 6)))
        # No-match hidden results remain reportable; only development selection excludes them.
        if frozen:
            best = 0
        if best is None:
            ci_rows.append(dict(method=name, stage=args.stage,
                **bootstrap_ci(rows, [evaluate(eligible_references(name, annotations[r['filename']]), [])
                                      for r in rows], args.seed)))
            summaries.append(dict(method=name, status='No eligible configuration: MAE unavailable',
                                  candidate='', **pooled[0]))
            continue
        artifact['methods'][name] = dict(parameters=params[best], candidate=best,
                                        selection_metric=selection_metrics[name], development_metrics=pooled[best])
        summaries.append(dict(method=name, status='ok' if pooled[best]['matched_attempts'] else UNAVAILABLE,
                              candidate=best, **pooled[best]))
        # Recompute only the selected development configuration. Retaining every
        # candidate's intervals/record scores scales poorly for the large grids.
        chosen_records = frozen_records[name]
        if not frozen:
            for row in rows:
                path = args.audio_dir / row['filename']
                if hashlib.sha256(path.read_bytes()).hexdigest() != config['audio_sha256'][row['filename']]:
                    raise ValueError('Audio changed during the experiment')
                audio, sr = load_audio(path, annotations[row['filename']])
                endpoint = EndpointRecording(audio, sr) if name in GRIDS else None
                pred = predict(name, audio, sr, params[best], ComparisonRecording(audio, sr), endpoint)
                score = evaluate(eligible_references(name, annotations[row['filename']]), pred, IOU_THRESHOLD)
                chosen_records.append((score, pred))
        ci_rows.append(dict(method=name, stage=args.stage,
                            **bootstrap_ci(rows, [s for s, _ in chosen_records], args.seed)))
        for row, (score, pred) in zip(rows, chosen_records):
            record_metrics.append(dict(method=name, filename=row['filename'], fold=row['fold'],
                                       **mae_summary([score])))
            for index, (a, b) in enumerate(pred, 1):
                predictions.append(dict(method=name, filename=row['filename'], attempt=index,
                                        start_seconds=a, end_seconds=b))
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'confidence_intervals.csv', ci_rows)
    write_csv(output / 'metrics.csv', record_metrics,
              ['method', 'filename', 'fold', *mae_summary([])])
    write_csv(output / 'predictions.csv', predictions,
              ['method', 'filename', 'attempt', 'start_seconds', 'end_seconds'])
    if args.stage == 'develop':
        (output / 'selection.json').write_text(json.dumps(artifact, indent=2))
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'Results: {output.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, KeyError) as error:
        raise SystemExit(f'Error: {error}') from error
