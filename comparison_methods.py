"""Standard five-detector comparison with per-recording feature reuse.

No endpoint drop refinement. Raw detector intervals are merged before duration
filtering. Native centering/timing conventions remain part of each detector.
"""
from collections import OrderedDict

import numpy as np

from methods import features, intervals_from_mask, rvad_fast

GAPS = [v / 1000 for v in (0, 10, 25, 50, 75, 100, 125, 150, 175, 200)]
WINDOWS = [10, 20, 30, 40, 50, 60, 70]
HOPS = [1, 2, 5, 10]
RMS_GRID = dict(margin_db=list(range(1, 13)), frame_ms=WINDOWS, hop_ms=HOPS,
                gap=GAPS, min_duration=[2.])
COMPARISON_GRIDS = {
    'adaptive_rms': RMS_GRID,
    'hysteresis_rms': {**RMS_GRID, 'hysteresis_db': [2, 4, 6, 8]},
    'rvad_fast': dict(vad_threshold=[.2, .3, .4, .5, .6], gap=GAPS, min_duration=[2.]),
    'praat_intensity': dict(silence_db=[10, 15, 20, 25, 30, 35, 40],
                            pitch_floor=[20, 40, 60, 80, 100], time_step_ms=HOPS,
                            gap=GAPS, min_duration=[2.]),
    'librosa_split': dict(top_db=[10, 15, 20, 25, 30, 35, 40], frame_ms=WINDOWS,
                          hop_ms=HOPS, gap=GAPS, min_duration=[2.]),
}


def merge_intervals(intervals, gap, minimum):
    merged = []
    for a, b in intervals:
        a, b = float(a), float(b)
        if b <= a:
            continue
        if merged and a - merged[-1][1] <= gap + 1e-9:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= minimum - 1e-9]


def require_dependencies(names):
    """Fail before creating an experiment directory, not halfway through a run."""
    if 'librosa_split' in names:
        import librosa  # noqa: F401
    if 'praat_intensity' in names:
        import parselmouth  # noqa: F401
    if 'rvad_fast' in names:
        import rVADfast  # noqa: F401


class ComparisonRecording:
    def __init__(self, audio, sample_rate):
        self.audio = np.asarray(audio, dtype=float)
        self.sr = sample_rate
        self.duration = len(audio) / sample_rate
        self.curves = {}
        self.raw_cache = OrderedDict()
        self.sound = None

    def _rms(self, p):
        key = ('rms', p.get('frame_ms', 30), p.get('hop_ms', 5))
        if key not in self.curves:
            _, times, db, hop = features(self.audio, self.sr, key[1], key[2])
            self.curves[key] = times, db, hop, float(np.percentile(db, 20))
        return self.curves[key]

    def _librosa(self, p):
        import librosa
        key = ('librosa', p['frame_ms'], p['hop_ms'])
        width = max(2, round(p['frame_ms'] * self.sr / 1000))
        hop = max(1, round(p['hop_ms'] * self.sr / 1000))
        if key not in self.curves:
            rms = librosa.feature.rms(y=self.audio, frame_length=width, hop_length=hop)[0]
            self.curves[key] = librosa.amplitude_to_db(rms, ref=np.max, top_db=None)
        mask = self.curves[key] > -p['top_db']
        # This is effects.split's documented/source algorithm with cached RMS.
        return intervals_from_mask(mask, np.arange(len(mask)) * hop / self.sr,
                                   self.duration, hop / self.sr, min_duration=0, gap=0)

    def _praat(self, p):
        import parselmouth
        from parselmouth.praat import call
        if self.sound is None:
            self.sound = parselmouth.Sound(self.audio, sampling_frequency=self.sr)
        key = ('praat', p['pitch_floor'], p['time_step_ms'])
        if key not in self.curves:
            # Praat needs a complete intensity window; pad only genuinely short
            # inputs, then clip every returned interval to the original duration.
            sound = self.sound
            minimum = 6.4 / p['pitch_floor'] + 2 / self.sr
            if self.duration < minimum:
                x = np.pad(self.audio, (0, int(np.ceil(minimum * self.sr)) - len(self.audio)))
                sound = parselmouth.Sound(x, sampling_frequency=self.sr)
            self.curves[key] = sound.to_intensity(minimum_pitch=p['pitch_floor'],
                                                  time_step=p['time_step_ms'] / 1000)
        # Praat requires positive cleanup durations; 1 ns is below the decision
        # resolution, leaving duration cleanup to the shared merge-then-filter.
        grid = call(self.curves[key], 'To TextGrid (silences)', -p['silence_db'],
                    1e-9, 1e-9, 'silent', 'sounding')
        result = []
        for i in range(1, call(grid, 'Get number of intervals', 1) + 1):
            if call(grid, 'Get label of interval', 1, i) == 'sounding':
                a = max(0., call(grid, 'Get start time of interval', 1, i))
                b = min(self.duration, call(grid, 'Get end time of interval', 1, i))
                if b > a:
                    result.append((a, b))
        return result

    def predict(self, name, p):
        key = (name, tuple(sorted((k, v) for k, v in p.items() if k not in ('gap', 'min_duration'))))
        if key in self.raw_cache:
            raw = self.raw_cache.pop(key)
        elif not np.any(self.audio):
            # Peak-relative silence rules otherwise label all-zero input as sound.
            raw = []
        elif name in ('adaptive_rms', 'hysteresis_rms'):
            times, db, hop, noise = self._rms(p)
            low = max(-75, noise + p.get('margin_db', 10 if name == 'adaptive_rms' else 6))
            mask = db >= low
            if name == 'hysteresis_rms':
                edges = np.diff(np.r_[False, mask, False].astype(int))
                for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    if not np.any(db[a:b] >= low + p.get('hysteresis_db', 6)):
                        mask[a:b] = False
            raw = intervals_from_mask(mask, times, self.duration, hop, min_duration=0, gap=0)
        elif name == 'rvad_fast':
            raw, _ = rvad_fast(self.audio, self.sr, vad_threshold=p['vad_threshold'], gap=0, min_duration=0)
        elif name == 'librosa_split':
            raw = self._librosa(p)
        elif name == 'praat_intensity':
            raw = self._praat(p)
        else:
            raise ValueError(f'Unknown comparison detector: {name}')
        self.raw_cache[key] = raw
        if len(self.raw_cache) > 512:
            self.raw_cache.popitem(last=False)
        return merge_intervals(raw, p.get('gap', .025), p.get('min_duration', 2.))


def praat_intensity(audio, sample_rate, **params):
    return ComparisonRecording(audio, sample_rate).predict('praat_intensity', params), {}


def librosa_split(audio, sample_rate, **params):
    return ComparisonRecording(audio, sample_rate).predict('librosa_split', params), {}
