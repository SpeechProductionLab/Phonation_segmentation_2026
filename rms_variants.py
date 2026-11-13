"""Three centered-RMS detectors; thresholds use no annotation information."""
from collections import OrderedDict
import math

import numpy as np

from comparison_methods import GAPS, WINDOWS, HOPS, merge_intervals
from methods import intervals_from_mask

SHARED = dict(frame_ms=WINDOWS, hop_ms=HOPS, gap=GAPS, min_duration=[1.5])
RMS_VARIANT_GRIDS = {
    'percentile_rms': dict(reference_percentile=[80, 85, 90, 95],
                           top_db=[10, 15, 20, 25, 30, 35, 40], **SHARED),
    'background_rms': dict(margin_db=list(range(1, 21)), **SHARED),
    'phonation_rms': dict(margin_db=list(range(1, 21)), delay_seconds=[.25, .5, .75],
                         reference_seconds=[.5, .75, 1.], sd_multiplier=[1., 1.5, 2., 2.5, 3.],
                         **SHARED),
}


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


class RMSRecording:
    def __init__(self, audio, sr):
        self.audio = np.asarray(audio, dtype=float)
        self.sr = sr
        self.duration = len(audio) / sr
        self.curves = {}
        self.cache = OrderedDict()

    def cached(self, key, calculate):
        if key in self.cache:
            value = self.cache.pop(key)
        else:
            value = calculate()
        self.cache[key] = value
        if len(self.cache) > 512:
            self.cache.popitem(last=False)
        return value

    def curve(self, p):
        import librosa
        key = (p['frame_ms'], p['hop_ms'])
        if key not in self.curves:
            width = max(2, round(p['frame_ms'] * self.sr / 1000))
            hop = max(1, round(p['hop_ms'] * self.sr / 1000)) / self.sr
            rms = librosa.feature.rms(y=self.audio, frame_length=width,
                                      hop_length=round(hop*self.sr))[0].astype(float)
            # Absolute amplitude dB, with a numerical floor (not peak normalization).
            db = 20 * np.log10(np.maximum(rms, 1e-12))
            self.curves[key] = rms, db, hop, float(np.percentile(db, 20))
        return self.curves[key]

    def background_regions(self, rms, threshold, hop, gap):
        """One candidate per background region; short terminal dips reach EOF."""
        active = rms >= threshold
        # The extra centered frame at EOF has no duration and cannot start an attempt.
        active = active[:int(math.ceil(self.duration / hop - 1e-9))]
        quiet = [(a, b) for a, b in runs(~active)
                 if min(b*hop, self.duration) - a*hop > gap + 1e-9]
        regions = []
        cursor = 0
        for a, b in [*quiet, (len(active), len(active))]:
            hits = np.flatnonzero(active[cursor:a])
            if len(hits):
                regions.append((cursor + int(hits[0]), min(a*hop, self.duration)))
            cursor = b
        return regions

    def references(self, rms, background, hop, p):
        result = []
        for start, stop in self.background_regions(rms, background, hop, p['gap']):
            begin_time = start*hop + p['delay_seconds']
            end_time = begin_time + p['reference_seconds']
            if end_time > stop + 1e-9:
                continue  # Reference incomplete: no prediction, never consult labels.
            first = int(math.ceil(begin_time / hop - 1e-9))
            end = int(math.ceil(end_time / hop - 1e-9))
            values = rms[first:end]
            if not len(values):
                continue
            result.append((start*hop, stop, end, float(values.mean()), float(values.std(ddof=0))))
        return result

    def predict(self, name, p):
        if not np.any(self.audio):
            return []
        rms, db, hop, noise_db = self.curve(p)
        frame_key = (p['frame_ms'], p['hop_ms'])
        if name == 'percentile_rms':
            reference = self.cached(('percentile', *frame_key, p['reference_percentile']),
                                    lambda: float(np.percentile(rms, p['reference_percentile'])))
            threshold_db = 20*np.log10(max(reference, 1e-12)) - p['top_db']
            key = (name, *frame_key, p['reference_percentile'], p['top_db'])
            mask = db > threshold_db
        else:
            background = 10 ** (max(-75., noise_db + p['margin_db']) / 20)
            key = (name, *frame_key, p['margin_db'])
            mask = rms >= background
        if name != 'phonation_rms':
            raw = self.cached(key, lambda: intervals_from_mask(mask,
                np.arange(len(rms))*hop, self.duration, hop, min_duration=0, gap=0))
            return merge_intervals(raw, p['gap'], p['min_duration'])
        ref_key = ('reference', *frame_key, p['margin_db'], p['delay_seconds'],
                   p['reference_seconds'], p['gap'])
        references = self.cached(ref_key, lambda: self.references(rms, background, hop, p))
        result = []
        for start, stop, ready, mean, sd in references:
            threshold = max(mean - p['sd_multiplier']*sd, background)
            last = int(math.ceil(min(self.duration, stop + p['gap'] + hop) / hop - 1e-9))
            end_time = stop
            for a, b in runs(rms[ready:last] < threshold):
                a, b = a + ready, b + ready
                if min(b*hop, self.duration) - a*hop > p['gap'] + 1e-9:
                    end_time = a*hop
                    break
            if end_time - start >= p['min_duration'] - 1e-9:
                result.append((start, end_time))
        return result


def percentile_rms(audio, sample_rate, **params):
    return RMSRecording(audio, sample_rate).predict('percentile_rms', params), {}


def background_rms(audio, sample_rate, **params):
    return RMSRecording(audio, sample_rate).predict('background_rms', params), {}


def phonation_rms(audio, sample_rate, **params):
    return RMSRecording(audio, sample_rate).predict('phonation_rms', params), {}
