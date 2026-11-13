"""Six training-free detectors. Times are seconds; samples must be floating point.

The first five are exploratory baselines, not exact reproductions of papers.
rVAD-fast wraps https://github.com/zhenghuatan/rVADfast (pinned in requirements).
"""
import math

import numpy as np
from scipy.signal import resample_poly


def features(audio, sample_rate, frame_ms=30., hop_ms=5.):
    """Left-aligned frames (default 30 ms / 5 ms); decisions occupy hop cells.

    Boundary uncertainty includes the analysis window. No per-file peak scaling:
    fixed dBFS thresholds retain the meaning of the original recording amplitude.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('Expected nonempty finite mono audio')
    if not np.isfinite([frame_ms, hop_ms]).all() or frame_ms <= 0 or not 0 < hop_ms <= frame_ms:
        raise ValueError('Require finite 0 < hop_ms <= frame_ms')
    width = max(2, round(frame_ms * sample_rate / 1000))
    hop = max(1, round(hop_ms * sample_rate / 1000))
    starts = np.arange(0, len(x), hop)
    padded = np.pad(x, (0, width))
    frames = np.lib.stride_tricks.sliding_window_view(padded, width)[::hop][:len(starts)]
    # Bound temporary squared-frame memory for fine-hop, long-window searches.
    rms = np.concatenate([np.sqrt(np.mean(frames[i:i + 2048] ** 2, axis=1))
                          for i in range(0, len(frames), 2048)])
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    return frames, starts / sample_rate, db, hop / sample_rate


def intervals_from_mask(mask, times, duration, hop, min_duration=.030, gap=.025):
    """Merge gaps first, then reject short islands. Raw mask remains available."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    intervals = [(float(times[a]), min(duration, float(times[b - 1] + hop)))
                 for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    merged = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= gap + 1e-9:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(a, b) for a, b in merged if b - a >= min_duration - 1e-9]


def _result(audio, sr, times, mask, hop, score, params, **extra):
    intervals = intervals_from_mask(mask, times, len(audio) / sr, hop,
                                    params.get('min_duration', .030), params.get('gap', .025))
    return intervals, dict(times=times, raw_activity=np.asarray(mask), score=score, **extra)


def _noise(db):
    # Entire-recording lower percentile, not an assumption that the first frame is silent.
    return float(np.percentile(db, 20))


def fixed_rms(audio, sample_rate, **params):
    _, times, db, hop = features(audio, sample_rate)
    threshold = params.get('threshold_db', -40)
    return _result(audio, sample_rate, times, db >= threshold, hop, db, params,
                   threshold=np.full(len(db), threshold))


def adaptive_rms(audio, sample_rate, **params):
    """Per-recording noise-relative threshold (not an online noise tracker)."""
    _, times, db, hop = features(audio, sample_rate, params.get('frame_ms', 30.), params.get('hop_ms', 5.))
    threshold = max(-75, _noise(db) + params.get('margin_db', 10))
    return _result(audio, sample_rate, times, db >= threshold, hop, db, params,
                   threshold=np.full(len(db), threshold))


def local_rms_energy_drop(audio, sample_rate, **params):
    """Trim each adaptive RMS attempt at its first sustained local energy drop."""
    attempts, diagnostics = adaptive_rms(audio, sample_rate, **params)
    times = diagnostics['times']
    db = diagnostics['score']
    width = max(2, round(.030 * sample_rate))
    hop = max(1, round(.005 * sample_rate))
    # The RMS frames are left-aligned. Compare a full frame before each
    # boundary to every frame across the following ~60 ms of observed audio.
    lag = math.ceil(width / hop)
    drop_db = params.get('drop_db', 9)
    min_duration = params.get('min_duration', .030)
    refined = []
    drop_score = np.full(len(times), np.nan)
    for start, end in attempts:
        first = max(np.searchsorted(times, start) + lag,
                    np.searchsorted(times, start + min_duration - 1e-9))
        # Exclude zero-padding and frames outside this particular attempt.
        last = (round(end * sample_rate) - width) // hop - lag
        if first <= last:
            after = np.lib.stride_tricks.sliding_window_view(
                db[first:last + lag + 1], lag + 1)
            scores = db[first - lag:last - lag + 1] - after.max(axis=1)
            drop_score[first:last + 1] = scores
            hits = np.flatnonzero(scores >= drop_db)
            if hits.size:
                # Use the strongest contrast in the first qualifying run.
                # A later rebound cannot reopen a confirmed endpoint.
                breaks = np.flatnonzero(np.diff(hits) > 1)
                run_end = hits[breaks[0]] if breaks.size else hits[-1]
                peak = hits[0] + np.argmax(scores[hits[0]:run_end + 1])
                end = float(times[first + peak])
        refined.append((start, end))
    diagnostics['drop_db'] = drop_score
    return refined, diagnostics


def hysteresis_rms(audio, sample_rate, **params):
    _, times, db, hop = features(audio, sample_rate, params.get('frame_ms', 30.), params.get('hop_ms', 5.))
    low = max(-75, _noise(db) + params.get('margin_db', 6))
    high = low + params.get('hysteresis_db', 6)
    # Extend each high-confidence seed through its surrounding low-threshold run.
    low_mask = db >= low
    edges = np.diff(np.r_[False, low_mask, False].astype(int))
    mask = np.zeros(len(db), dtype=bool)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if np.any(db[a:b] >= high):
            mask[a:b] = True
    return _result(audio, sample_rate, times, mask, hop, db, params,
                   threshold=np.full(len(db), low), start_threshold=np.full(len(db), high))


def energy_periodicity(audio, sample_rate, **params):
    frames, times, db, hop = features(audio, sample_rate)
    centered = frames - frames.mean(axis=1, keepdims=True)
    nfft = 1 << (2 * frames.shape[1] - 1).bit_length()
    ac = np.fft.irfft(abs(np.fft.rfft(centered, nfft, axis=1)) ** 2, nfft, axis=1)
    first = max(1, int(sample_rate / 500))
    last = min(frames.shape[1] - 1, int(sample_rate / 60))
    periodicity = np.max(ac[:, first:last + 1], axis=1) / np.maximum(ac[:, 0], 1e-20)
    threshold = max(-75, _noise(db) + params.get('margin_db', 6))
    mask = (db >= threshold) & (periodicity >= params.get('periodicity', .5))
    return _result(audio, sample_rate, times, mask, hop, periodicity, params,
                   energy_db=db, threshold=np.full(len(db), params.get('periodicity', .5)))


def spectral_similarity(audio, sample_rate, **params):
    """Linear-frequency spectral cosine baseline, not Wu et al.'s Mel implementation."""
    frames, times, db, hop = features(audio, sample_rate)
    spectra = abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), axis=1))
    spectra[:, 0] = 0
    unit = spectra / np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-20)
    adjacent = np.sum(unit[1:] * unit[:-1], axis=1)
    score = np.zeros(len(db))
    if len(adjacent):
        score[:-1] = adjacent
        score[1:] = np.maximum(score[1:], adjacent)
    threshold = max(-75, _noise(db) + params.get('margin_db', 6))
    mask = (db >= threshold) & (score >= params.get('similarity', .8))
    return _result(audio, sample_rate, times, mask, hop, score, params,
                   energy_db=db, threshold=np.full(len(db), params.get('similarity', .8)))


def rvad_fast(audio, sample_rate, **params):
    from rVADfast import rVADfast
    # Official implementation uses a 512-point FFT: standardize to 16 kHz.
    divisor = math.gcd(int(sample_rate), 16000)
    x = resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor)
    duration = len(audio) / sample_rate
    if not np.any(x):
        times = np.arange(0, duration, .010)
        labels = np.zeros(len(times), dtype=bool)
    else:
        # Pad very short inputs for library framing; discard padded decisions below.
        x = np.pad(x, (0, max(0, 16000 - len(x))))
        detector = rVADfast(vad_threshold=params.get('vad_threshold', .4))
        labels, times = detector(x, 16000)
        labels, times = np.asarray(labels), np.asarray(times)
        keep = times < duration
        labels, times = labels[keep], times[keep]
    return _result(audio, sample_rate, times, labels > 0, .010, labels, params)


METHODS = {f.__name__: f for f in (fixed_rms, adaptive_rms, hysteresis_rms,
                                  energy_periodicity, spectral_similarity, rvad_fast)}
