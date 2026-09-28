"""R07 - how often the evaluator's 1,000-detection caps bind in the saved validation jobs.

Per job: images whose post-NMS prediction list reached the global 1,000 cap
(evaluate_r04.deduplicate max_det=1000, class-aware NMS over all crops/scales),
images whose raw per-crop predictions were all saturated at 1,000 per crop, the
lowest score kept on each capped image, and image-class cells left with fewer than
300 kept predictions. The AP change caused by the caps is NOT computable from
these files: predictions removed before saving are not stored.
"""
from collections import Counter
import json
import os

from common import R04, Inputs, write_output

SUITE = R04 / "reports/evaluation_suite/evaluations"


def main():
    inp = Inputs()
    rows = []
    for folder in sorted(p for p in SUITE.iterdir() if p.is_dir()):
        profiles = inp.json(folder / "inference_profiles.json")
        samples = {s["id"]: s for s in inp.json(folder / "samples.json")}
        capped = [p for p in profiles if p["post_global_nms_predictions"] >= 1000]
        saturated = [p["id"] for p in profiles if p["pre_global_nms_predictions"] == 1000 * p["tiles"]]
        detail = []
        for p in capped:
            s = samples[p["id"]]
            per_class = Counter(q["class_id"] for q in s["predictions"])
            detail.append({"image": p["id"], "kept": len(s["predictions"]), "lowest_kept_score": min(q["score"] for q in s["predictions"]),
                           "classes_with_fewer_than_300_kept": {str(c): per_class.get(c, 0) for c in range(4) if per_class.get(c, 0) < 300},
                           "tiles": p["tiles"], "raw_before_global_nms": p["pre_global_nms_predictions"]})
        rows.append({"job": folder.name, "images": len(profiles), "images_at_global_cap": len(capped),
                     "images_all_crops_saturated": len(saturated), "capped_detail": detail})
    total_images = sum(r["images"] for r in rows)
    write_output("r07_eval_caps", {
        "caps": "per crop: model.predict max_det=1000 (conf .001, iou .6); per image: class-aware global NMS iou .5 then first 1,000 by score; COCO: maxDets 100/300 per image per category",
        "ap300_label": "AP50-95 'max300' = COCO max 300 per image per category AFTER the per-image 1,000 limit across all classes",
        "unmeasured": "AP change due to the caps; needs predictions before the caps (not saved)"},
        {"jobs": len(rows), "image_evaluations": total_images,
         "image_evaluations_at_global_cap": sum(r["images_at_global_cap"] for r in rows), "rows": rows}, inp)


if __name__ == "__main__":
    main()
