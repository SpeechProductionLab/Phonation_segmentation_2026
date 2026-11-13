"""One-to-one, maximum-cardinality interval matching at each IoU threshold."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def interval_iou(a, b):
    intersection = max(0., min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union > 0 else 0.


def match_intervals(reference, predicted, threshold=.5):
    if not reference or not predicted:
        return []
    ious = np.array([[interval_iou(a, b) for b in predicted] for a in reference])
    valid = ious >= threshold
    # One extra match outweighs any possible change in total IoU.
    reward = valid * (min(ious.shape) + 1 + ious)
    rows, cols = linear_sum_assignment(-reward)
    return [(int(i), int(j), float(ious[i, j])) for i, j in zip(rows, cols) if valid[i, j]]


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    return dict(precision=precision, recall=recall, f1=f1)


def evaluate(reference, predicted, threshold=.5):
    matches = match_intervals(reference, predicted, threshold)
    tp, fp, fn = len(matches), len(predicted) - len(matches), len(reference) - len(matches)
    onset = [abs(predicted[j][0] - reference[i][0]) for i, j, _ in matches]
    offset = [abs(predicted[j][1] - reference[i][1]) for i, j, _ in matches]
    signed_ends = [predicted[j][1] - reference[i][1] for i, j, _ in matches]
    # Split/merge indicators count overlap with >1 interval; independent of matching.
    splits = sum(sum(interval_iou(a, b) > 0 for b in predicted) > 1 for a in reference)
    merges = sum(sum(interval_iou(a, b) > 0 for a in reference) > 1 for b in predicted)
    short = [i for i, a in enumerate(reference) if a[1] - a[0] < 1.]
    return dict(tp=tp, fp=fp, fn=fn, **scores(tp, fp, fn),
                late_end_count=sum(v > .100 + 1e-12 for v in signed_ends),
                early_end_count=sum(v < -.100 - 1e-12 for v in signed_ends),
                matched_iou_sum=sum(v for _, _, v in matches),
                onset_error_sum_seconds=sum(onset), offset_error_sum_seconds=sum(offset),
                mean_matched_iou=np.mean([v for _, _, v in matches]) if tp else '',
                onset_mae_seconds=np.mean(onset) if tp else '',
                offset_mae_seconds=np.mean(offset) if tp else '',
                split_reference_count=splits, merged_prediction_count=merges,
                short_reference_count=len(short),
                short_matched_count=sum(i in short for i, _, _ in matches))


def aggregate(rows):
    keys = ('tp', 'fp', 'fn', 'matched_iou_sum', 'onset_error_sum_seconds',
            'offset_error_sum_seconds', 'split_reference_count', 'merged_prediction_count',
            'short_reference_count', 'short_matched_count')
    totals = {k: sum(row[k] for row in rows) for k in keys}
    n = totals['tp']
    totals.update(scores(totals['tp'], totals['fp'], totals['fn']))
    totals.update(mean_matched_iou=totals['matched_iou_sum'] / n if n else '',
                  onset_mae_seconds=totals['onset_error_sum_seconds'] / n if n else '',
                  offset_mae_seconds=totals['offset_error_sum_seconds'] / n if n else '',
                  short_recall=totals['short_matched_count'] / totals['short_reference_count']
                  if totals['short_reference_count'] else '')
    return totals
