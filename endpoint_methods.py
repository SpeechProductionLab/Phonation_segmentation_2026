"""Offline endpoint experiments for sustained vowels. All times are seconds."""
from functools import cached_property
from math import gcd

import numpy as np
from scipy.signal import resample_poly
from scipy.special import betainc

from methods import adaptive_rms

SHARED_START = dict(margin_db=10, gap=.025, min_duration=.030)
HOLDS = [40, 80]
GRIDS = {
    'cpp_drop': dict(drop_db=[3, 6, 9], hold_ms=HOLDS),
    'vowel_template': dict(similarity=[.65, .75, .85], hold_ms=HOLDS),
    'fade_to_background': dict(search_ms=[200, 400, 800], tail_ms=[50, 100]),
    'vowel_band_drop': dict(high_hz=[1000, 2000], drop_db=[6, 9, 12], hold_ms=[60]),
    'relative_vowel_level': dict(drop_db=[9, 12, 15], hold_ms=[50, 100]),
    'hnr_drop': dict(drop_db=[4, 7, 10], hold_ms=HOLDS),
    'pitch_loss': dict(probability=[.2, .4, .6], hold_ms=HOLDS),
    'high_low_ratio': dict(rise_db=[3, 6, 9], hold_ms=HOLDS),
    'flatness_rise': dict(rise_db=[3, 6, 9], hold_ms=HOLDS),
    'endpoint_vote': dict(hold_ms=[40, 60, 80]),
}


def power_db(power):
    return 10 * np.log10(np.maximum(power, 1e-20))


class EndpointRecording:
    """Reuse starts, frame features and reference windows across the entire search."""

    def __init__(self, audio, sr):
        self.attempts, _ = adaptive_rms(audio, sr, **SHARED_START)
        # Standardize spectral features to the bandwidth shared by all supported WAVs.
        divisor = gcd(sr, 8000)
        x = audio if sr == 8000 else resample_poly(audio, 8000 // divisor, sr // divisor)
        # Fully observed 40 ms frames, timestamps at centers; no fabricated EOF silence.
        self.frames = (np.lib.stride_tricks.sliding_window_view(x, 320)[::40]
                       if len(x) >= 320 else np.empty((0, 320)))
        self.times = .020 + np.arange(len(self.frames)) * .005
        self.db = power_db(np.mean(self.frames ** 2, axis=1))
        self.floor = np.percentile(self.db, 20) if len(self.db) else -200
        self.regions = []
        for start, end in self.attempts:
            first, stop = np.searchsorted(self.times, [start + .030, end - .020 + 1e-9])
            ref = self.reference(first, stop)
            self.regions.append((first, stop, ref))

    def reference(self, first, stop):
        # First 100 ms with <= 3 dB range and clear signal above the background.
        for i in range(first, stop - 19):
            levels = self.db[i:i + 20]
            if np.ptp(levels) <= 3 and np.median(levels) >= self.floor + 6:
                return slice(i, i + 20)
        return None

    @cached_property
    def spectrum(self):
        return abs(np.fft.rfft(self.frames * np.hanning(320), 512)) ** 2

    def band(self, low, high):
        freq = np.fft.rfftfreq(512, 1 / 8000)
        return self.spectrum[:, (freq >= low) & (freq < high)].sum(axis=1)

    @cached_property
    def bands(self):
        return {high: power_db(self.band(100, high)) for high in (1000, 2000)}

    @cached_property
    def profile(self):
        edges = np.geomspace(100, 3800, 21)
        bands = np.stack([self.band(lo, hi) for lo, hi in zip(edges, edges[1:])], axis=1)
        amplitude = np.sqrt(bands)
        return amplitude / np.maximum(np.linalg.norm(amplitude, axis=1, keepdims=True), 1e-12)

    @cached_property
    def ratio(self):
        return power_db(self.band(2000, 3800)) - self.bands[1000]

    @cached_property
    def flatness(self):
        freq = np.fft.rfftfreq(512, 1 / 8000)
        bands = self.spectrum[:, (freq >= 100) & (freq < 3800)]
        return np.mean(power_db(bands), axis=1) - power_db(bands.mean(axis=1))

    @cached_property
    def cpp(self):
        # Power cepstrum in dB; fit a straight background from 1 to 20 ms.
        cep = power_db(np.fft.irfft(power_db(self.spectrum), n=512) ** 2)
        q = np.arange(8, 161)
        centered = q - q.mean()
        slope = cep[:, q] @ centered / (centered @ centered)
        # 60-500 Hz pitch range.
        peaks = np.arange(16, 134)
        trend = cep[:, q].mean(axis=1)[:, None] + slope[:, None] * (peaks - q.mean())
        return (cep[:, peaks] - trend).max(axis=1)

    @cached_property
    def lag_statistics(self):
        """Equal-length pairs for lags 1..134; normalized correlation and YIN differences."""
        frames = self.frames - self.frames.mean(axis=1, keepdims=True)
        n = 186
        base = frames[:, :n]
        lags = np.arange(1, 135)
        cross = np.fft.irfft(np.conj(np.fft.rfft(base, 512)) * np.fft.rfft(frames, 512),
                             512)[:, lags]
        cumulative = np.pad(np.cumsum(frames ** 2, axis=1), ((0, 0), (1, 0)))
        base_energy = cumulative[:, [n]]
        shifted = cumulative[:, n + lags] - cumulative[:, lags]
        corr = cross / np.sqrt(np.maximum(base_energy * shifted, 1e-30))
        difference = np.maximum(base_energy + shifted - 2 * cross, 0)
        norm = difference * lags / np.maximum(np.cumsum(difference, axis=1), 1e-30)
        norm[base_energy[:, 0] < 1e-15] = 1
        return corr, norm

    @cached_property
    def hnr(self):
        r = np.clip(self.lag_statistics[0][:, 15:133].max(axis=1), 1e-6, 1 - 1e-6)
        return 10 * np.log10(r / (1 - r))

    @cached_property
    def voicing(self):
        # Probability that a Beta(2,18) YIN threshold admits a pitch trough.
        # This is a frame-level threshold model, without pitch-track smoothing.
        curve = self.lag_statistics[1]
        values = curve[:, 15:133]
        trough = (values <= curve[:, 14:132]) & (values < curve[:, 16:134])
        minimum = np.where(trough, values, 1).min(axis=1)
        return 1 - betainc(2, 18, np.clip(minimum, 0, 1))

    def fade(self, start, end, stop_time, search_ms, tail_ms):
        window = (self.times >= max(start + .030, end - search_ms / 1000)) & (
            self.times <= min(stop_time - .020, end + tail_ms / 1000))
        t, y = self.times[window], self.db[window]
        hold = round(tail_ms / 5)
        if len(t) < hold + 6:
            return end
        best = end
        single_fit = np.polyval(np.polyfit(t - t[0], y, 1), t - t[0])
        best_error = np.sum((y - single_fit) ** 2)
        for i in range(6, len(t) - hold + 1):
            tail = y[i:]
            if t[i] >= end or abs(tail.mean() - self.floor) > 3 or np.std(tail) > 3:
                continue
            slope, level = np.polyfit(t[:i] - t[i], y[:i], 1)
            if slope >= 0 or slope * (t[0] - t[i]) < 3 or abs(level - tail.mean()) > 3:
                continue
            error = np.sum((y[:i] - (slope * (t[:i] - t[i]) + level)) ** 2)
            error += np.sum((tail - tail.mean()) ** 2)
            if error < best_error:
                best, best_error = t[i], error
        return best

    def refine(self, name, params):
        predictions = []
        for index, (start, end) in enumerate(self.attempts):
            first, stop, ref = self.regions[index]
            if name == 'fade_to_background':
                limit = self.attempts[index + 1][0] if index + 1 < len(self.attempts) else np.inf
                end = self.fade(start, end, limit, **params)
            elif first < stop and (ref is not None or name == 'vowel_band_drop'):
                begin = ref.stop if ref is not None and name != 'vowel_band_drop' else first
                if name == 'vowel_band_drop':
                    begin = max(begin, np.searchsorted(self.times, start + .020) + 6)
                times = self.times[begin:stop]
                hold = round(params['hold_ms'] / 5)
                if len(times) >= hold:
                    if name == 'vowel_band_drop':
                        sustained = self.band_holds(begin, stop, hold, params['high_hz'], params['drop_db'])
                    else:
                        mask = self.cue(name, params, begin, stop, ref)
                        sustained = np.lib.stride_tricks.sliding_window_view(mask, hold).all(axis=1)
                    hits = np.flatnonzero(sustained)
                    if hits.size:
                        end = float(times[hits[0]])
            predictions.append((start, end))
        return predictions

    def cue(self, name, p, begin, stop, ref):
        region = slice(begin, stop)
        if name in ('cpp_drop', 'hnr_drop', 'relative_vowel_level'):
            levels = {'cpp_drop': 'cpp', 'hnr_drop': 'hnr', 'relative_vowel_level': 'db'}
            series = getattr(self, levels[name])
            return series[region] <= np.median(series[ref]) - p['drop_db']
        if name in ('high_low_ratio', 'flatness_rise'):
            series = self.ratio if name == 'high_low_ratio' else self.flatness
            return series[region] >= np.median(series[ref]) + p['rise_db']
        if name == 'pitch_loss':
            return (np.median(self.voicing[ref]) >= .7) & (self.voicing[region] < p['probability'])
        if name == 'vowel_template':
            template = self.profile[ref].mean(axis=0)
            similarity = self.profile[region] @ template / max(np.linalg.norm(template), 1e-12)
            return similarity < p['similarity']
        if name == 'endpoint_vote':
            cpp = self.cue('cpp_drop', {'drop_db': 6}, begin, stop, ref)
            template = self.cue('vowel_template', {'similarity': .75}, begin, stop, ref)
            band = self.bands[2000]
            band_drop = band[region] <= np.median(band[ref]) - 9
            return (cpp.astype(int) + template + band_drop) >= 2
        raise ValueError(f'Unknown endpoint method: {name}')

    def band_holds(self, begin, stop, hold, high_hz, drop_db):
        # Freeze each candidate's preceding 30 ms reference across its entire hold.
        series = self.bands[high_hz]
        before = np.lib.stride_tricks.sliding_window_view(series[begin-6:stop-1], 6)
        after = np.lib.stride_tricks.sliding_window_view(series[begin:stop], hold)
        return np.median(before[:len(after)], axis=1) - after.max(axis=1) >= drop_db
