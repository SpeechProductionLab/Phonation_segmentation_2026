#!/usr/bin/env python3
"""Flatten patient WAV folders without modifying originals (standard library only)."""

import argparse
import csv
import os
from pathlib import Path
import shutil
import sys


MANIFEST = 'collection_manifest.csv'


def plan_collection(source, output):
    """Validate folders and return deterministic (patient, source, target) entries."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f'Dataset folder does not exist: {source}')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Choose separate dataset and output folders; neither may contain the other.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('The output must be a new or completely empty folder. Existing files will not be overwritten.')

    entries = []
    targets = set()
    # Raise traversal errors instead of silently omitting unreadable patient data.
    def walk_error(error):
        raise error

    patients = sorted(source.iterdir(), key=lambda p: p.name)
    for patient in patients:
        if patient.is_symlink():
            continue
        if patient.is_file() and patient.suffix.lower() == '.wav':
            raise ValueError(f'WAV outside a patient folder: {patient.name}. Select the folder containing the patient folders.')
        if not patient.is_dir():
            continue
        recordings = []
        for directory, folders, files in os.walk(patient, followlinks=False, onerror=walk_error):
            folders[:] = sorted(name for name in folders if not (Path(directory) / name).is_symlink())
            for name in files:
                path = Path(directory) / name
                if path.suffix.lower() == '.wav' and not path.is_symlink() and path.is_file():
                    recordings.append(path)
        recordings.sort(key=lambda p: p.relative_to(patient).as_posix())
        for number, recording in enumerate(recordings, 1):
            filename = f'{patient.name}_{number}.wav'
            # Also protect case-insensitive filesystems, even when running on Linux.
            key = filename.casefold()
            if key in targets:
                raise ValueError(f'Patient names produce conflicting output filenames: {filename}')
            targets.add(key)
            entries.append((patient.name, recording, output / filename))
    if not entries:
        raise ValueError('No WAV files found below patient folders. Select the parent of C1, CNE133, etc.')
    return source, output, entries


def collect(source, output, dry_run=False):
    source, output, entries = plan_collection(source, output)
    if dry_run:
        for patient, original, target in entries:
            print(f'{original.relative_to(source)} -> {target.name}')
        print(f'Preview: {len(entries)} WAV files from {len({e[0] for e in entries})} patients. No files written.')
        return len(entries)

    output.mkdir(parents=True, exist_ok=True)
    # Check again immediately before writing; exclusive creation also protects every file.
    if any(output.iterdir()):
        raise ValueError('Output folder is no longer empty. No collection started.')
    with (output / MANIFEST).open('x', newline='', encoding='utf-8') as manifest:
        writer = csv.writer(manifest)
        writer.writerow(['patient_id', 'output_filename', 'source_path', 'source_relative_path', 'size_bytes'])
        manifest.flush()
        for index, (patient, original, target) in enumerate(entries, 1):
            created = False
            try:
                with original.open('rb') as src, target.open('xb') as dst:
                    created = True
                    shutil.copyfileobj(src, dst)
                size = target.stat().st_size
            except BaseException:
                if created:
                    target.unlink(missing_ok=True)
                raise
            writer.writerow([patient, target.name, str(original), original.relative_to(source).as_posix(), size])
            manifest.flush()
            print(f'[{index}/{len(entries)}] {target.name}')
    print(f'Collected {len(entries)} WAV files into {output}\nMapping: {output / MANIFEST}')
    return len(entries)


def choose_folders():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        root = tk.Tk()
    except Exception as error:
        raise RuntimeError('Folder dialogs are unavailable. Use: python3 collect_wav.py DATASET OUTPUT') from error
    root.withdraw()
    try:
        source = filedialog.askdirectory(title='Choose dataset folder containing patient folders (C1, CNE133, …)', parent=root)
        if not source:
            return
        output = filedialog.askdirectory(title='Choose a completely empty output folder', parent=root)
        if not output:
            return
        try:
            count = collect(source, output)
        except Exception as error:
            messagebox.showerror('Collection failed', f'{error}\n\nIf copying started, completed files and their mapping remain in the output. Use a new empty folder to retry.', parent=root)
            return 1
        messagebox.showinfo('Collection complete', f'Copied {count} WAV files.\n\nOutput: {output}\nMapping: {MANIFEST}\n\nOriginal files are unchanged.', parent=root)
        return 0
    finally:
        root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='No arguments opens folder-selection dialogs. Patient IDs are the immediate folder names beneath DATASET. Symlinks are skipped.')
    parser.add_argument('dataset', nargs='?', help='Folder containing patient folders such as C1 and CNE133')
    parser.add_argument('output', nargs='?', help='New or completely empty output folder')
    parser.add_argument('--dry-run', action='store_true', help='Preview filenames without creating or copying anything (requires both paths)')
    args = parser.parse_args()
    if not args.dataset and not args.output and not args.dry_run:
        try:
            return choose_folders() or 0
        except Exception as error:
            print(f'Error: {error}', file=sys.stderr)
            return 1
    if not args.dataset or not args.output:
        parser.error('Provide both DATASET and OUTPUT, or no arguments for folder dialogs.')
    try:
        collect(args.dataset, args.output, args.dry_run)
    except (OSError, ValueError) as error:
        print(f'Error: {error}\nOriginal files are unchanged. If copying started, completed files and their mapping remain; retry with a new empty output folder.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
