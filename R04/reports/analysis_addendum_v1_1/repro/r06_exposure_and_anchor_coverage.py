"""R06 - training label counts, actual repeated exposures and ignore anchor coverage.

Separates: original train objects; unique originals drawn as a positive label (any
visible fraction / complete); stored crop labels (views.json); actual drawn
labels = sum over views of (draw count x labels in the view); sampling weights;
and, per arm, the share of exposures from the most-drawn view/groups. Also counts
ignore regions (and stored fragment positives) that contain no anchor centre at
any stride after the training LetterBox (1024, centred, scaleup=True).
The anchor-coverage counts describe geometry only, not a performance loss.
"""
from collections import Counter, defaultdict
import math

from common import R04, RECORD, Inputs, write_output

NAMES = ["resistor", "capacitor", "ic", "connector"]


def letterboxed(box, w, h, size=1024):
    """Ultralytics 8.4.120 LetterBox(center, scaleup): labels scaled by r, shifted by the rounded left/top."""
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    left, top = round((size - nw) / 2 - .1), round((size - nh) / 2 - .1)
    cx, cy, bw, bh = box
    return ((cx - bw / 2) * w * r + left, (cy - bh / 2) * h * r + top, (cx + bw / 2) * w * r + left, (cy + bh / 2) * h * r + top)


def centres(box, stride):
    x1, y1, x2, y2 = box
    nx = max(0, math.floor(x2 / stride - .5) - math.ceil(x1 / stride - .5) + 1)
    ny = max(0, math.floor(y2 / stride - .5) - math.ceil(y1 / stride - .5) + 1)
    return nx * ny


def main():
    inp = Inputs()
    res = {}
    originals = inp.json(R04 / "data/original_records.json")["images"]
    train_objects = [o for r in originals if r["split"] == "train" for o in r["objects"]]
    res["original_train_objects"] = {"total": len(train_objects), "per_class": dict(Counter(o["class_name"] for o in train_objects))}
    train_ids = {o["instance_id"] for o in train_objects}
    for arm in ("baseline", "improved"):
        views = {v["id"]: v for v in inp.json(R04 / f"data/{arm}/views.json")["records"]}
        exposure = inp.json(RECORD / f"runs/{arm}/sampled_exposures.json")
        draws = exposure["view_draw_counts"]
        stored = Counter(NAMES[t["class_id"]] for v in views.values() for t in v["targets"])
        stored_ignore = Counter(g["reason"] for v in views.values() for g in v["ignore_regions"])
        drawn = Counter(); drawn_ignore = Counter(); by_kind = Counter(); kind_labels = Counter(); by_group = Counter()
        for vid, n in draws.items():
            v = views[vid]
            by_kind[v["kind"]] += n
            kind_labels[v["kind"]] += n * len(v["targets"])
            by_group[v["group_id"]] += n * len(v["targets"])
            for t in v["targets"]:
                drawn[NAMES[t["class_id"]]] += n
            for g in v["ignore_regions"]:
                drawn_ignore[g["reason"]] += n
        total = sum(drawn.values())
        top_view = max(draws, key=lambda k: draws[k] * len(views[k]["targets"]))
        tv = views[top_view]
        counts = sorted(draws.values(), reverse=True)
        seen = {t["instance_id"] for vid in draws for t in views[vid]["targets"]}
        seen_complete = {t["instance_id"] for vid in draws for t in views[vid]["targets"] if t["visible_fraction"] >= 1 - 1e-12}
        recorded_seen = [exposure["unique_positive_source_instances_seen"], exposure["unique_positive_source_instances_seen_complete"]]
        # anchor-centre coverage
        no_centre_regions = Counter(); no_centre_exposure = Counter(); region_count = Counter()
        frag_pos = frag_pos_no_centre = 0
        for vid, v in views.items():
            n = draws.get(vid, 0)
            for g in v["ignore_regions"]:
                box = letterboxed(g["bbox_xywhn"], v["width"], v["height"])
                empty = sum(centres(box, s) for s in (8, 16, 32)) == 0
                region_count[g["reason"]] += 1
                no_centre_regions[g["reason"]] += empty
                no_centre_exposure[g["reason"]] += n * empty
            for t in v["targets"]:
                if t["visible_fraction"] < .5:
                    frag_pos += 1
                    frag_pos_no_centre += sum(centres(letterboxed(t["bbox_xywhn"], v["width"], v["height"]), s) for s in (8, 16, 32)) == 0
        weight_sums = defaultdict(float)
        for v in views.values():
            weight_sums[v["group_id"]] += v["sampling_weight"]
        res[arm] = {
            "stored_views": len(views), "views_drawn": len(draws), "view_kinds_stored": dict(Counter(v["kind"] for v in views.values())),
            "unique_original_objects_drawn": {"as_positive": len(seen), "complete": len(seen_complete), "all_in_train_split": seen <= train_ids,
                                              "recorded": recorded_seen, "match_recorded": [len(seen), len(seen_complete)] == recorded_seen},
            "stored_positive_labels": {"total": sum(stored.values()), "per_class": dict(stored)},
            "stored_ignore_regions": dict(stored_ignore),
            "stored_fragment_positives_lt50": frag_pos, "stored_fragment_positives_without_anchor_centre": frag_pos_no_centre,
            "actual_draws": sum(draws.values()), "draws_by_view_kind": dict(by_kind),
            "actual_positive_label_exposures": {"total": total, "per_class": dict(drawn), "per_view_kind": dict(kind_labels),
                                                "per_draw": total / sum(draws.values())},
            "actual_ignore_exposures": dict(drawn_ignore),
            "recorded_class_exposures_match": {NAMES[int(k)]: v for k, v in exposure["class_exposures"].items()} == dict(drawn),
            "most_exposed_view": {"id": top_view, "group": tv["group_id"], "kind": tv["kind"], "draws": draws[top_view],
                                  "labels_in_view": len(tv["targets"]), "exposures": draws[top_view] * len(tv["targets"]),
                                  "share_of_positive_exposures": draws[top_view] * len(tv["targets"]) / total,
                                  "views_in_its_group": sum(1 for v in views.values() if v["group_id"] == tv["group_id"])},
            "group_exposure_shares": {g: e / total for g, e in by_group.most_common()},
            "top3_group_share": sum(e for _, e in by_group.most_common(3)) / total,
            "draw_count_quantiles": {"max": counts[0], "median": counts[len(counts) // 2], "min": counts[-1]},
            "effective_views_inverse_simpson": sum(counts) ** 2 / sum(c * c for c in counts),
            "sampling_weight_group_sums": {"min": min(weight_sums.values()), "max": max(weight_sums.values()), "groups": len(weight_sums)},
            "ignore_anchor_coverage": {reason: {"regions": region_count[reason], "regions_without_anchor_centre": no_centre_regions[reason],
                                                "region_share": no_centre_regions[reason] / region_count[reason],
                                                "exposures": drawn_ignore[reason], "exposures_without_anchor_centre": no_centre_exposure[reason],
                                                "exposure_share": no_centre_exposure[reason] / drawn_ignore[reason] if drawn_ignore[reason] else None}
                                       for reason in region_count}}
    b, i = res["baseline"], res["improved"]
    res["arm_differences"] = {"positive_exposure_ratio_improved_over_baseline": i["actual_positive_label_exposures"]["total"] / b["actual_positive_label_exposures"]["total"],
                              "per_class_ratio": {c: i["actual_positive_label_exposures"]["per_class"][c] / b["actual_positive_label_exposures"]["per_class"][c] for c in NAMES},
                              "native_draws": [b["draws_by_view_kind"].get("native1024", 0), i["draws_by_view_kind"].get("native1024", 0)]}
    write_output("r06_exposure_and_anchor_coverage", {
        "unique_drawn": "distinct instance_id among positive targets of views drawn at least once; complete = visible_fraction >= 1-1e-12 (prepare_r04.py:129 rule)",
        "stored_label": "one target row in one saved crop label file (views.json targets)",
        "actual_exposure": "sum over drawn views of draw_count x labels in that view (8,872 draws = 2,218 batches x 4)",
        "sampling_weight": "per view: priority (1.5 if the view has an IC/connector target else 1.0) / sum of priorities in its board group",
        "loss_weight": "not computed here: class multipliers are 1.0 in effect (cls_pw=0, no model.class_weights; see R01/start.json)",
        "anchor_centre": "centre (i+0.5)*stride inside the box after LetterBox to 1024; strides 8/16/32; inclusive bounds",
        "interpretation": "anchor-coverage shares are geometric counts, NOT performance-loss rates"}, res, inp)


if __name__ == "__main__":
    main()
