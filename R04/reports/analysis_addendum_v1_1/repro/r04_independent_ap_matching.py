"""R04 - independent re-computation of the stored validation metrics.

NumPy re-implementation of COCO bbox AP (IoU .50:.05:.95, 101 recall points,
precision envelope, per-image-per-category maxDets, area=all, category mean over
categories with GT) and of the greedy same-class IoU>=0.5 operating matching.
It imports neither pycocotools, evaluate_r04.py nor audit_r04_results.py.
Inputs are the saved per-job outputs of the completed validation jobs.
"""
from collections import defaultdict
import json

import numpy as np

from common import R04, Inputs, write_output

SUITE = R04 / "reports/evaluation_suite/evaluations"
IOUS = np.linspace(.5, .95, 10)
RECALLS = np.linspace(0, 1, 101)


def iou_xywh(d, g):
    if not len(d) or not len(g):
        return np.zeros((len(d), len(g)))
    dx2, dy2, gx2, gy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.clip(np.minimum(dx2[:, None], gx2[None]) - np.maximum(d[:, 0][:, None], g[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(dy2[:, None], gy2[None]) - np.maximum(d[:, 1][:, None], g[:, 1][None]), 0, None)
    inter = iw * ih
    union = (d[:, 2] * d[:, 3])[:, None] + (g[:, 2] * g[:, 3])[None] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0)


def coco_ap(gt, preds, max_dets=(100, 300)):
    gts, dts = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        gts[(a["image_id"], a["category_id"])].append(a["bbox"])
    for p in preds:
        dts[(p["image_id"], p["category_id"])].append((p["score"], p["bbox"]))
    images = [i["id"] for i in gt["images"]]
    out = {m: [] for m in max_dets}
    for cat in [c["id"] for c in gt["categories"]]:
        npos = sum(len(gts[(i, cat)]) for i in images)
        if not npos:
            continue
        per_image = []
        for img in images:
            g = np.array(gts[(img, cat)], float).reshape(-1, 4)
            ordered = sorted(dts[(img, cat)], key=lambda t: -t[0])[:max(max_dets)]
            d = np.array([b for _, b in ordered], float).reshape(-1, 4)
            s = np.array([sc for sc, _ in ordered], float)
            ious = iou_xywh(d, g)
            tp = np.zeros((len(IOUS), len(d)), bool)
            for ti, t in enumerate(IOUS):
                matched = np.zeros(len(g), bool)
                for di in range(len(d)):
                    best, best_j = min(t, 1 - 1e-10), -1
                    for gj in range(len(g)):
                        if matched[gj] or ious[di, gj] < best:
                            continue
                        best, best_j = ious[di, gj], gj
                    if best_j >= 0:
                        matched[best_j] = True
                        tp[ti, di] = True
            per_image.append((s, tp))
        for m in max_dets:
            s = np.concatenate([x[0][:m] for x in per_image])
            t = np.concatenate([x[1][:, :m] for x in per_image], axis=1)
            order = np.argsort(-s, kind="mergesort")
            t = t[:, order]
            precision = np.zeros((len(IOUS), len(RECALLS)))
            for ti in range(len(IOUS)):
                tpc = np.cumsum(t[ti]).astype(float)
                fpc = np.cumsum(~t[ti]).astype(float)
                rc = tpc / npos
                pr = tpc / np.maximum(tpc + fpc, np.spacing(1))
                for i in range(len(pr) - 1, 0, -1):
                    pr[i - 1] = max(pr[i - 1], pr[i])
                idx = np.searchsorted(rc, RECALLS, side="left")
                q = np.zeros(len(RECALLS))
                ok = idx < len(pr)
                q[ok] = pr[idx[ok]]
                precision[ti] = q
            out[m].append(precision)
    return {m: {"ap50_95": float(np.stack(v).mean()), "ap50": float(np.stack(v)[:, 0].mean())} for m, v in out.items()}


def operating(samples, threshold):
    counts = np.zeros((4, 3), dtype=int)
    for s in samples:
        truths, found = s["truth"], set()
        for p in sorted((p for p in s["predictions"] if p["score"] >= threshold), key=lambda p: -p["score"]):
            best, best_j = -1.0, -1
            a = p["bbox_xyxy"]
            for j, g in enumerate(truths):
                if j in found or g["class_id"] != p["class_id"]:
                    continue
                b = g["bbox_xyxy"]
                inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                q = inter / union if union > 0 else 0.0
                if q >= best:              # exact ties -> highest GT index, as R03/R04
                    best, best_j = q, j
            if best >= .5:
                found.add(best_j); counts[p["class_id"], 0] += 1
            else:
                counts[p["class_id"], 1] += 1
        for j, g in enumerate(truths):
            if j not in found:
                counts[g["class_id"], 2] += 1
    return counts.tolist()


def main():
    inp = Inputs()
    rows, worst = [], 0.0
    for folder in sorted(p for p in SUITE.iterdir() if p.is_dir()):
        gt = inp.json(folder / "native_ground_truth.json")
        preds = inp.json(folder / "bbox_predictions.json")
        samples = inp.json(folder / "samples.json")
        stored = inp.json(folder / "metrics.json")
        mine = coco_ap(gt, preds)
        conf = stored["operating"]["confidence"]
        counts = operating(samples, conf)
        b = stored["bbox"]
        diffs = {"ap50_95_max100": abs(b["ap50_95_max100"] - mine[100]["ap50_95"]),
                 "ap50_95_max300": abs(b["ap50_95_max300"] - mine[300]["ap50_95"]),
                 "ap50_max300": abs(b["ap50_max300"] - mine[300]["ap50"])}
        worst = max(worst, *diffs.values())
        per_cell = defaultdict(int)
        for p in preds:
            per_cell[(p["image_id"], p["category_id"])] += 1
        rows.append({"job": folder.name, "stored": {k: b[k] for k in diffs}, "recomputed": {"ap50_95_max100": mine[100]["ap50_95"],
                     "ap50_95_max300": mine[300]["ap50_95"], "ap50_max300": mine[300]["ap50"]}, "abs_diff": diffs,
                     "operating_confidence": conf, "operating_counts_equal": counts == stored["operating"]["per_class_counts_tp_fp_fn"],
                     "prediction_cells_over_100": sum(v > 100 for v in per_cell.values()),
                     "prediction_cells_over_300": sum(v > 300 for v in per_cell.values()), "prediction_cells": len(per_cell)})
    write_output("r04_independent_ap_matching", {
        "coco": "IoU thresholds .50:.05:.95; 101 recall points; greedy per image/category by score (stable) with best-IoU unmatched GT; per-image-per-category truncation to maxDets before accumulation; precision envelope; mean over IoU x recall x categories with GT",
        "operating": "score-descending greedy, same class, IoU>=0.5, one-to-one; exact IoU ties to the highest native GT index",
        "caveat": "Reproduces the stored arithmetic from saved predictions; cannot see predictions removed by the evaluator's per-crop/global 1,000 caps"},
        {"jobs": len(rows), "max_abs_ap_difference": worst, "all_operating_counts_equal": all(r["operating_counts_equal"] for r in rows), "rows": rows}, inp)


if __name__ == "__main__":
    main()
