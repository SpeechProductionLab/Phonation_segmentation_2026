# Phonation segmentation with RMS thresholds

Python tools for detecting phonation onset and offset, comparing predicted intervals with manual annotations, and evaluating boundary error. The main experiment compares three RMS-based methods: `percentile_rms`, `background_rms`, and `phonation_rms`.

## Methods

All three use centered `librosa.feature.rms` frames, mono channel averaging, the original sample rate, and zero padding at recording edges.

| Method | Reference | Detection rule |
| --- | --- | --- |
| Percentile RMS | High percentile of linear RMS across the recording, converted to dB | Keep frames above reference dB minus `top_db` |
| Background RMS | Recording-wide 20th percentile of RMS dB | Keep frames at or above `max(-75, background_db + margin_db)` |
| Phonation RMS | Mean and population SD of linear RMS in a window after detected onset | Start using the background threshold; end after a sustained drop below `max(mean - k * SD, background threshold in linear units)` |

## Inputs

Supply your own audio directory and annotation CSV. The required CSV columns are:

```csv
audio_filename,start_seconds,end_seconds
participant01_1.wav,1.0,4.5
participant01_1.wav,6.0,9.2
```

## Development experiment

With the environment activated, run:

```sh
python run_mae.py develop --method rms_variants --audio-dir "path/to/audio" --csv "path/to/phonation-collection.csv" --splits "path/to/mae_splits.csv" --output "results/rms_variants_development"
```

Use a new output directory for every run. The full search evaluates 265,440 parameter configurations; it can take substantial time and disk space and does not support resuming an interrupted run. To evaluate only one method, replace `rms_variants` with its method ID.

### Eligibility, matching, and selection

- Reference and predicted attempts must last at least 1.5 seconds. Exactly 1.5 seconds is retained with a numerical tolerance.
- Recordings with no eligible reference attempts are excluded entirely. Reference intervals are not merged.
- Predictions and references are matched one-to-one at intersection-over-union (IoU) >= 0.25.
- Each method's parameters minimize **pooled matched offset MAE** across development folds. This is not the unweighted average of fold MAEs.
- Only matched attempts contribute to boundary MAE. Misses and extra detections do not penalize the selection objective, and recall/F1 are not tie-breakers. A configuration needs at least one match; exact ties retain the first configuration in grid order.
- Inspect precision, recall, F1, and missed/extra counts alongside MAE. Low MAE alone can conceal poor detection coverage.
- Confidence intervals use 2,000 participant-cluster bootstrap resamples, with seed 42 by default. Development CIs are exploratory after parameter selection. MAE and its CIs describe matched attempts, not all references.

The evaluation excludes wholly negative and short-only recordings, so it does not measure false-alarm performance over all possible inputs.

## Held-out evaluation

After freezing the methods and development selection:

```sh
python run_mae.py test --method rms_variants --selection "results/rms_variants_development/selection.json" --audio-dir "path/to/audio" --csv "path/to/phonation-collection.csv" --splits "path/to/mae_splits.csv" --output "results/rms_variants_hidden"
```

Reuse the same source code, data, and split manifest. The runner checks frozen-selection compatibility, including source hashes. Do not tune using hidden results. If a method had no matched development configurations, use `--method all` to test only methods present in the frozen selection.

## Outputs

| File | Contents |
| --- | --- |
| `summary.csv` | Selected-method boundary MAE, match counts, precision, recall, F1, and early/late endpoint rates |
| `selection.json` | Selected parameters and compatibility information |
| `parameter_search.csv`, `fold_metrics.csv` | Candidate scores and fold-level scores from development |
| `metrics.csv`, `predictions.csv` | Recording-level metrics and selected predicted intervals |
| `confidence_intervals.csv` | Participant-bootstrap 95% boundary-MAE confidence intervals |
| `config.json`, `splits.csv` | Experiment settings, provenance, and participant assignments |

Inspect the results locally:

```sh
python inspect_results.py --results "results/rms_variants_development" --audio-dir "path/to/audio" --csv "path/to/phonation-collection.csv"
```

The viewer's reference RMS display uses 30 ms frames and 5 ms hops; it is not necessarily the selected detector's internal RMS curve or threshold.

Calculate reference durations over the same evaluated recording set:

```sh
python calculate_reference_duration.py --csv "path/to/phonation-collection.csv" --results "results/rms_variants_development"
```

Without `--results`, the duration summary covers the entire annotation CSV. A ratio of MAE to mean reference duration is not a mean per-attempt percentage error.
