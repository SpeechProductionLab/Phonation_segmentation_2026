# RMS variants: Windows setup and reproducible evaluation

This experiment compares `percentile_rms`, `background_rms`, and `phonation_rms`.
Use `--method rms_variants` to run exactly these three. Each selects its parameters
by lowest **pooled matched offset (end) MAE**, not combined MAE.

## Parameters

All three use centered `librosa.feature.rms` frames, zero padding at recording
edges, mono channel averaging, and the audio's original sample rate.

| Parameter | Values |
| --- | --- |
| Frame length | 10, 20, 30, 40, 50, 60, 70 ms |
| Hop length | 1, 2, 5, 10 ms |
| Gap | 0, 10, 25, 50, 75, 100, 125, 150, 175, 200 ms |
| Minimum prediction/reference duration | **1.5 seconds, fixed** |

### Percentile-reference RMS: 7,840 configurations

- Reference percentile: **80, 85, 90, 95**, calculated on linear frame RMS.
- `top_db`: **10, 15, 20, 25, 30, 35, 40 dB**.
- Convert the reference to dB; keep frames strictly above `reference_db - top_db`.
- Merge gaps no longer than `gap`, then discard intervals shorter than 1.5 seconds.

### Background-reference RMS: 5,600 configurations

- Background is the **20th percentile of frame RMS dB**, fixed.
- Margin: **1 through 20 dB**, inclusive, in 1 dB steps.
- Threshold: `max(-75, background_db + margin_db)` in amplitude dB.
- Keep frames at or above the threshold; merge gaps, then filter by duration.
- This uses centered librosa frames, unlike the older left-aligned `adaptive_rms`.

### Phonation-reference RMS: 252,000 configurations

- Background margin: **1 through 20 dB**, in 1 dB steps.
- Delay after onset: **0.25, 0.5, 0.75 seconds**.
- Reference duration: **0.5, 0.75, 1 second**.
- SD multiplier `k`: **1, 1.5, 2, 2.5, 3**.