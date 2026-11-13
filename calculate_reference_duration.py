"""Summarize annotation durations in seconds; requires only standard Python."""
import argparse
import csv
import math
import statistics
from pathlib import Path


def read_rows(path, required):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: required columns: {", ".join(sorted(required))}')
        return list(reader)


def summarize(label, records):
    print(f'\n{label}')
    if not records:
        print('  No reference attempts; average unavailable.')
        return
    durations = [duration for _, duration in records]
    print(f'  Reference attempts: {len(durations)}')
    print(f'  Recordings:         {len({name for name, _ in records})}')
    print(f'  Mean duration:      {statistics.mean(durations):.6f} seconds')
    print(f'  Median duration:    {statistics.median(durations):.6f} seconds')
    print(f'  Minimum duration:   {min(durations):.6f} seconds')
    print(f'  Maximum duration:   {max(durations):.6f} seconds')
    print(f'  Total duration:     {sum(durations):.6f} seconds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('phonation-collection.csv'),
                        help='Annotation CSV (default: phonation-collection.csv).')
    parser.add_argument('--results', type=Path,
                        help='Optional RMS-variants results folder containing metrics.csv; '
                             'restricts annotations to the evaluated recordings.')
    parser.add_argument('--min-duration', type=float, default=1.5,
                        help='Eligibility cutoff in seconds (default: 1.5).')
    args = parser.parse_args()
    if not math.isfinite(args.min_duration) or args.min_duration < 0:
        parser.error('--min-duration must be finite and nonnegative')
    try:
        rows = read_rows(args.csv, {'audio_filename', 'start_seconds', 'end_seconds'})
        records, seen = [], set()
        for line, row in enumerate(rows, 2):
            name = row['audio_filename'].strip()
            start, end = float(row['start_seconds']), float(row['end_seconds'])
            if not name or not all(map(math.isfinite, [start, end])) or not 0 <= start < end:
                raise ValueError(f'Annotation line {line}: invalid filename or interval')
            key = (name, start, end)
            if key in seen:
                raise ValueError(f'Annotation line {line}: duplicate interval for {name}')
            seen.add(key)
            records.append((name, end-start))
        if not records:
            raise ValueError('Annotation CSV contains no attempts')

        if args.results:
            metrics = read_rows(args.results / 'metrics.csv', {'method', 'filename'})
            methods = {'percentile_rms', 'background_rms', 'phonation_rms'}
            by_method = {method: {r['filename'] for r in metrics if r['method'] == method}
                         for method in methods}
            if not all(by_method.values()):
                raise ValueError('metrics.csv must contain all three RMS variant methods')
            filenames = by_method['percentile_rms']
            if any(names != filenames for names in by_method.values()):
                raise ValueError('The three methods evaluated different recording sets')
            missing = filenames - {name for name, _ in records}
            if missing:
                raise ValueError(f'{len(missing)} evaluated recordings are absent from annotations')
            records = [(name, duration) for name, duration in records if name in filenames]
            print(f'Scope: recordings in {args.results / "metrics.csv"}')
        else:
            print('Scope: entire annotation CSV (may include recordings outside development results)')

        print(f'Annotations: {args.csv}')
        print('Each attempt is weighted equally; duration = end_seconds - start_seconds.')
        summarize('All reference attempts in scope', records)
        eligible = [(name, duration) for name, duration in records
                    if duration >= args.min_duration - 1e-9]
        summarize(f'Eligible reference attempts (>= {args.min_duration:g} seconds)', eligible)
        print(f'  Excluded short attempts: {len(records)-len(eligible)}')
        print('\nFor the MAE comparison, send the eligible mean duration and attempt count.')
        print('The supplied three-method results had 78 eligible reference attempts.')
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
