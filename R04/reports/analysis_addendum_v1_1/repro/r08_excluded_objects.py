"""R08 - the excluded source objects that the auxiliary FP taxonomy (addendum section 3) will use.

Read-only checks, no prediction is classified here:
- native_manifest.json and original_records.json hold the same excluded_source_objects;
- each source VOC XML: <object> count = targets + excluded (nothing dropped), and every
  record's bbox_voc_raw equals the XML box at its 0-based source_object_index;
- bbox_xyxy = (xmin-1, ymin-1, xmax, ymax) for targets and excluded objects alike, and
  every box is non-degenerate and inside the image;
- the saved validation truth equals the original target boxes, and all saved jobs share
  one native_ground_truth.json, so excluded boxes, GT and predictions share one native
  pixel frame;
- every source_type named in the fixed reference subtotals exists verbatim.
"""
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from common import R04, Inputs, write_output

SUITE = R04 / "reports/evaluation_suite/evaluations"
TRUTH_JOB = "baseline__epoch_20__dual__val"
SUBTOTALS = {
    "look_alike_passives": ["resistor network", "resistor jumper", "capacitor jumper", "ferrite bead",
                            "inductor", "fuse", "emi filter", "potentiometer"],
    "pins_pads": ["pins", "pads"],
    "text": ["text", "component text"],
}


def xml_objects(path):
    root = ET.parse(path).getroot()
    return [(o.findtext("name"), [float(o.find("bndbox").findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
            for o in root.findall("object")]


def main():
    inp = Inputs()
    originals = inp.json(R04 / "data/original_records.json")["images"]
    manifest = {r["id"]: r for r in inp.json(R04 / "data/native_manifest.json")["records"]}
    problems, name_differences = [], []
    per_cohort = defaultdict(Counter)
    totals = Counter()
    for rec in originals:
        image = rec["image_id"]
        man = manifest.get(image)
        if man is None:
            problems.append(f"{image}: not in native_manifest.json")
            continue
        if man["excluded_source_objects"] != rec["excluded_source_objects"]:
            problems.append(f"{image}: excluded objects differ between manifest and original records")
        xml_path = inp.add(rec["annotation"])
        if inp.items[str(xml_path.resolve())]["sha256"] != rec["annotation_sha256"]:
            problems.append(f"{image}: annotation SHA-256 differs from the record")
        xml = xml_objects(xml_path)
        objects = rec["objects"] + rec["excluded_source_objects"]
        if sorted(o["source_object_index"] for o in objects) != list(range(len(xml))):
            problems.append(f"{image}: {len(xml)} XML objects but {len(objects)} records, or indices not one-to-one")
        for o in objects:
            i, raw, box = o["source_object_index"], o["bbox_voc_raw"], o["bbox_xyxy"]
            if i < len(xml):
                if [float(v) for v in raw] != xml[i][1]:
                    problems.append(f"{image}:{i}: bbox_voc_raw differs from the XML box")
                if o["source_name"] != xml[i][0]:
                    name_differences.append([image, i, o["source_name"], xml[i][0]])
            if list(box) != [raw[0] - 1, raw[1] - 1, raw[2], raw[3]]:
                problems.append(f"{image}:{i}: bbox_xyxy is not (xmin-1, ymin-1, xmax, ymax)")
            if not (0 <= box[0] < box[2] <= rec["width"] and 0 <= box[1] < box[3] <= rec["height"]):
                problems.append(f"{image}:{i}: box degenerate or outside the image")
        totals["xml_objects"] += len(xml)
        totals["targets"] += len(rec["objects"])
        totals["excluded"] += len(rec["excluded_source_objects"])
        for e in rec["excluded_source_objects"]:
            per_cohort[man["cohort"]][e["source_type"]] += 1

    by_image = {r["image_id"]: r for r in originals}
    samples = inp.json(SUITE / TRUTH_JOB / "samples.json")
    truth_equal = all(
        sorted((t["class_id"], *map(float, t["bbox_xyxy"])) for t in s["truth"])
        == sorted((o["class_id"], *map(float, o["bbox_xyxy"])) for o in by_image[s["id"]]["objects"])
        for s in samples)
    gt_files = {}
    for folder in sorted(p for p in SUITE.iterdir() if p.is_dir()):
        path = inp.add(folder / "native_ground_truth.json")
        gt_files[folder.name] = inp.items[str(path.resolve())]["sha256"]

    all_types = Counter()
    for counts in per_cohort.values():
        all_types.update(counts)
    listed = {t for names in SUBTOTALS.values() for t in names}
    write_output("r08_excluded_objects", {
        "excluded_source_object": "a source VOC object whose type is not one of the four targets; kept with its source_type, never a GT",
        "bbox_xyxy": "continuous native pixels, (xmin-1, ymin-1, xmax, ymax) from the 1-based VOC box; same rule as target GT",
        "source_object_index": "0-based position of the <object> element in the source XML",
        "subtotals": "fixed reference groups declared in addendum section 3.6; not labels or classes"}, {
        "images": len(originals),
        "xml_objects": totals["xml_objects"], "targets": totals["targets"], "excluded": totals["excluded"],
        "targets_plus_excluded_equals_xml": totals["targets"] + totals["excluded"] == totals["xml_objects"],
        "problems": problems,
        "source_name_differs_from_xml_name": {"count": len(name_differences), "examples": name_differences[:5]},
        "source_types": len(all_types),
        "excluded_by_source_type": dict(all_types.most_common()),
        "excluded_by_cohort": {c: {"total": sum(n.values()), "unknown": n.get("unknown", 0), "by_source_type": dict(n.most_common())}
                               for c, n in sorted(per_cohort.items())},
        "validation_truth_equals_original_targets": truth_equal, "truth_checked_job": TRUTH_JOB,
        "native_ground_truth_sha256_distinct": sorted(set(gt_files.values())), "native_ground_truth_files": len(gt_files),
        "subtotal_names_missing_from_data": sorted(listed - set(all_types)),
        "subtotal_counts": {k: sum(all_types.get(t, 0) for t in names) for k, names in SUBTOTALS.items()},
        "types_in_rest": sorted(set(all_types) - listed)}, inp)


if __name__ == "__main__":
    main()
