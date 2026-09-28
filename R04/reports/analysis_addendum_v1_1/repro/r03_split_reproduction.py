"""R03 - reproduce the R04 train/val group split from the R03 train+val pool only.

Re-implements prepare_r04.freeze_split: 6 of 23 pool groups, constraints on
class counts/shares and the 8-16 px bin, weighted mean |share-0.25| objective,
lexicographic tie-break. R03 test/Pi groups are not in the pool. Also reports
per-split size bins and the largest group's object share.
"""
from collections import Counter
from itertools import combinations

import numpy as np

from common import R04, Inputs, write_output


def size_bin(o):
    a, b, c, d = o["bbox_xyxy"]
    s = min(c - a, d - b)
    return 0 if s < 8 else 1 if s < 16 else 2 if s < 32 else 3


def main():
    inp = Inputs()
    rows = inp.json(R04 / "data/original_records.json")["images"]
    split = inp.json(R04 / "data/split.json")
    pool = [r for r in rows if r["r03_split"] in ("train", "val")]
    groups = sorted({r["board_group"] for r in pool})
    features = []
    for g in groups:
        v = np.zeros(25)
        for r in (x for x in pool if x["board_group"] == g):
            v[-1] += 1
            for o in r["objects"]:
                c, b = o["class_id"], size_bin(o)
                v[c] += 1; v[4 + b] += 1; v[8 + c * 4 + b] += 1
        features.append(v)
    features = np.array(features)
    total = features.sum(0); active = total > 0
    weights = np.array([2] * 4 + [2] * 4 + [1] * 16 + [2], dtype=float)
    best, eligible, combos = None, 0, 0
    for combo in combinations(range(len(groups)), 6):
        combos += 1
        counts = features[list(combo)].sum(0)
        if (counts[:4] < 25).any() or not .18 <= counts[:4].sum() / total[:4].sum() <= .34:
            continue
        share = counts[:4] / total[:4]
        if ((share < .12) | (share > .45)).any() or counts[5] < 50:
            continue
        eligible += 1
        score = float(np.average(np.abs(counts[active] / total[active] - .25), weights=weights[active]))
        item = (score, tuple(groups[i] for i in combo))
        if best is None or item < best:
            best = item
    res = {"pool_groups": len(groups), "combinations": combos, "eligible": eligible,
           "recomputed_val_groups": sorted(best[1]), "recomputed_objective": best[0],
           "stored_val_groups": split["validation_groups"], "stored_objective": split["objective"],
           "stored_eligible": split["eligible_combinations"],
           "match": sorted(best[1]) == split["validation_groups"] and abs(best[0] - split["objective"]) < 1e-15 and eligible == split["eligible_combinations"],
           "pool_contains_r03_test_or_pi": any(r["r03_split"] not in ("train", "val") for r in pool)}
    per_split = {}
    for name in ("train", "val", "test", "pi_test"):
        subset = [r for r in rows if r["split"] == name]
        objects = [o for r in subset for o in r["objects"]]
        by_group = Counter()
        for r in subset:
            by_group[r["board_group"]] += len(r["objects"])
        top = by_group.most_common(1)[0]
        cells = [n for r in subset for n in Counter(o["class_id"] for o in r["objects"]).values()]
        per_split[name] = {"images": len(subset), "groups": len(by_group), "objects": len(objects),
                           "per_class": dict(Counter(o["class_name"] for o in objects)),
                           "short_side_bins": dict(zip(["lt8", "8to16", "16to32", "ge32"], np.bincount([size_bin(o) for o in objects], minlength=4).tolist())),
                           "largest_group": top[0], "largest_group_object_share": top[1] / len(objects),
                           "group_objects": dict(by_group.most_common()),
                           "image_class_cells_gt_gt100": sum(n > 100 for n in cells), "image_class_cells_gt_gt300": sum(n > 300 for n in cells),
                           "max_gt_per_image_class": max(cells),
                           "excluded_by_source_type": dict(Counter(o["source_type"] for r in subset for o in r["excluded_source_objects"]))}
    res["per_split"] = per_split
    write_output("r03_split_reproduction", {
        "features": "per group: 4 class counts, 4 short-side bins (<8,8-16,16-32,>=32 px), 16 class x bin, image count",
        "objective": "weighted mean |val share - 0.25| over active features; weights 2 (class, bin, images) / 1 (class x bin)",
        "constraints": "each class >=25; val target share 18-34%; each class share 12-45%; 8-16 px objects >=50"}, res, inp)


if __name__ == "__main__":
    main()
