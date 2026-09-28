"""Audit original WACV component boxes, freeze board splits, and tile TRAIN only.

Original algorithmic code. Standard native sliding windows and VOC geometry; no
predictions, generated masks, or downloaded executable code. Run with existing
Python + Pillow + NumPy + matplotlib. Archive is acquired separately from the
author-linked Georgia Tech server. Raw images are local research assets: explicit
redistribution/training license was not supplied in the archive or author page.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


NAMES = ["resistor", "capacitor", "ic", "connector"]
MAPPING = {"resistor": 0, "capacitor": 1, "electrolytic capacitor": 1,
           "ic": 2, "connector": 3}
SOURCE_URL = "https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection"


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positions(length: int, window: int, overlap: float):
    if length <= window:
        return [0]
    step = max(1, int(window * (1 - overlap)))
    result = list(range(0, length - window + 1, step))
    if result[-1] != length - window:
        result.append(length - window)
    return result


def source_type(name):
    match = re.match(r'^\s*(?:"([^"]+)"|(\S+))', name)
    if not match:
        raise ValueError(f"Empty class name: {name!r}")
    return (match.group(1) or match.group(2)).strip().lower()


def yolo_line(class_id, box, width, height):
    x1, y1, x2, y2 = box
    return f"{class_id} {(x1+x2)/(2*width):.10f} {(y1+y2)/(2*height):.10f} {(x2-x1)/width:.10f} {(y2-y1)/height:.10f}"


def audit(root: Path):
    records = []
    source_counts = collections.Counter()
    dropped = collections.Counter()
    border_zero = 0
    for annotation in sorted(root.rglob("*.xml")):
        tree = ET.parse(annotation).getroot()
        image_candidates = [p for p in annotation.parent.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
        if len(image_candidates) != 1:
            raise ValueError(f"Image pair ambiguous: {annotation}")
        image = image_candidates[0].resolve()
        with Image.open(image) as original:
            width, height = original.size
        if (width, height) != (int(tree.findtext("size/width")), int(tree.findtext("size/height"))):
            raise ValueError(f"XML/image size mismatch: {image}")
        objects, excluded = [], []
        for index, node in enumerate(tree.findall("object")):
            raw_name = node.findtext("name", "")
            typ = source_type(raw_name)
            source_counts[typ] += 1
            raw_box = [int(node.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")]
            # VOC is conventionally 1-based inclusive; convert to half-open XYXY.
            # Preserve the raw box and explicitly clip any source coordinate 0.
            x1, y1, x2, y2 = raw_box
            border_zero += int(x1 == 0 or y1 == 0)
            box = [max(0, x1 - 1), max(0, y1 - 1), x2, y2]
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"Invalid bbox: {annotation}:{index}: {raw_box}")
            item = {"instance_id": f"{annotation.stem}:{index:05d}", "source_object_index": index,
                    "source_name": raw_name, "source_type": typ, "bbox_voc_raw": raw_box,
                    "bbox_xyxy": box, "difficult": int(node.findtext("difficult", "0")),
                    "truncated": int(node.findtext("truncated", "0"))}
            if typ not in MAPPING:
                dropped[typ] += 1
                excluded.append(item)
            else:
                item.update(class_id=MAPPING[typ], class_name=NAMES[MAPPING[typ]])
                objects.append(item)
        group = re.sub(r"_(?:Top|Bottom)\d*$", "", annotation.stem, flags=re.IGNORECASE)
        records.append({"image_id": annotation.stem, "board_group": group,
                        "image": str(image), "annotation": str(annotation.resolve()),
                        "width": width, "height": height, "image_sha256": digest(image),
                        "annotation_sha256": digest(annotation), "objects": objects,
                        "excluded_source_objects": excluded})
    if len(records) != 47:
        raise ValueError(f"Unexpected native image count {len(records)}; review source version")
    if len(set(x["image_sha256"] for x in records)) != len(records):
        raise ValueError("Exact duplicate native image found; grouping review required")
    return records, source_counts, dropped, border_zero


def freeze_split(records):
    groups = sorted({r["board_group"] for r in records})
    if "RPI3B" not in groups:
        raise ValueError("Required unseen Pi group absent")
    other = [g for g in groups if g != "RPI3B"]
    random.Random(42).shuffle(other)
    split_groups = {"val": other[:5], "test": other[5:10], "train": other[10:], "pi_test": ["RPI3B"]}
    assigned = {g: s for s, gs in split_groups.items() for g in gs}
    for r in records:
        r["split"] = assigned[r["board_group"]]
    for split in split_groups:
        present = {o["class_id"] for r in records if r["split"] == split for o in r["objects"]}
        if present != set(range(4)):
            raise ValueError(f"A target class is missing from fixed {split} split")
    return split_groups


def create_data(output, records, tile_size=1024, overlap=.2):
    dataset = output / "component_dataset"
    tile_records = []
    for r in records:
        split = r["split"]
        idir, ldir = dataset / "images" / split, dataset / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        if split != "train":
            target = idir / (r["image_id"] + ".jpg")
            if not target.exists():
                # A local hardlink avoids duplicating the source image bytes.
                os.link(r["image"], target)
            elif digest(target) != r["image_sha256"]:
                raise ValueError(f"Existing native image differs: {target}")
            (ldir / (r["image_id"] + ".txt")).write_text("\n".join(
                yolo_line(o["class_id"], o["bbox_xyxy"], r["width"], r["height"])
                for o in r["objects"]) + "\n", encoding="utf-8")
            r["dataset_native_image"] = str(target.resolve())
            continue
        with Image.open(r["image"]) as im:
            im = im.convert("RGB")
            for y in positions(r["height"], tile_size, overlap):
                for x in positions(r["width"], tile_size, overlap):
                    right, bottom = min(x + tile_size, r["width"]), min(y + tile_size, r["height"])
                    width, height = right-x, bottom-y
                    tid = f"{r['image_id']}__x{x}_y{y}_w{width}_h{height}"
                    exposures = []
                    for o in r["objects"]:
                        a, b, c, d = o["bbox_xyxy"]
                        clip = [max(a, x), max(b, y), min(c, right), min(d, bottom)]
                        cw, ch = clip[2]-clip[0], clip[3]-clip[1]
                        if cw <= 0 or ch <= 0:
                            continue
                        if cw < 1 or ch < 1:
                            raise ValueError("Positive subpixel fragment: native integer coordinate assumption changed")
                        local = [clip[0]-x, clip[1]-y, clip[2]-x, clip[3]-y]
                        exposures.append({"instance_id": o["instance_id"], "class_id": o["class_id"],
                                          "bbox_xyxy": local, "source_bbox_xyxy": o["bbox_xyxy"],
                                          "visible_fraction": cw*ch/((c-a)*(d-b)),
                                          "clipped": local != [a-x, b-y, c-x, d-y]})
                    target = idir / (tid + ".jpg")
                    if not target.exists():
                        im.crop((x,y,right,bottom)).save(target, quality=92, subsampling=0)
                    (ldir / (tid + ".txt")).write_text("\n".join(
                        yolo_line(o["class_id"], o["bbox_xyxy"], width, height) for o in exposures)
                        + ("\n" if exposures else ""), encoding="utf-8")
                    tile_records.append({"tile_id": tid, "image": str(target.resolve()),
                                         "source_image_id": r["image_id"], "board_group": r["board_group"],
                                         "crop_xyxy": [x,y,right,bottom], "width": width, "height": height,
                                         "objects": exposures})
    expected = {o["instance_id"] for r in records if r["split"] == "train" for o in r["objects"]}
    exposed = {o["instance_id"] for t in tile_records for o in t["objects"]}
    if expected != exposed:
        raise ValueError("Train tiling lost a source instance")
    return dataset, tile_records


def draw_review(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.font_manager import fontManager, FontProperties
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        fontManager.addfont(str(font))
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#ffbc00", "#27d1e0", "#ff4c7a", "#7ced70"]
    pi = [r for r in records if r["split"] == "pi_test"]
    selected = pi + [next(r for r in records if r["split"] == s) for s in ("train", "val", "test")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), facecolor="#f6f8fa")
    for ax, r in zip(axes.flat[:5], selected):
        with Image.open(r["image"]) as im:
            ax.imshow(im)
        for obj in r["objects"]:
            x1,y1,x2,y2 = obj["bbox_xyxy"]
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,lw=.85,edgecolor=colors[obj["class_id"]]))
        ax.set_title(f"{r['image_id']} · {r['split']}\n{r['width']}×{r['height']} · 정답 bbox {len(r['objects'])}개", fontsize=12)
        ax.axis("off")
    ax = axes.flat[5]
    r = next(r for r in pi if r["image_id"].endswith("Top"))
    # Deterministic native central ROI for coordinate QA; never saved as test GT.
    x,y = max(0, r["width"]//2-320), max(0,r["height"]//2-320)
    with Image.open(r["image"]) as im:
        ax.imshow(im)
    for obj in r["objects"]:
        a,b,c,d = obj["bbox_xyxy"]
        ax.add_patch(Rectangle((a,b),c-a,d-b,fill=False,lw=1.1,edgecolor=colors[obj["class_id"]]))
    ax.set_xlim(x, x+640); ax.set_ylim(y+640,y); ax.axis("off")
    ax.set_title("RPI3B Top · 중앙 640px 좌표 확인\n원본 bbox만 표시 · 예측/마스크 생성 없음", fontsize=12)
    handles = [Rectangle((0,0),1,1,facecolor=c) for c in colors]
    fig.legend(handles,["저항", "커패시터", "IC", "커넥터"], loc="upper center", ncol=4,bbox_to_anchor=(.5,.943),frameon=False,fontsize=12)
    fig.suptitle("WACV 원본 부품 정답 좌표 확인 — Raspberry Pi 3B 상·하면은 미관측 시험용", fontsize=17, weight="bold",y=.985)
    fig.text(.03,.025,"원본: Kuo et al., WACV 2019 / 저자 공개 ZIP · 명시적 재배포 라이선스 미확인: 로컬 연구용 검토 자료\n범주: 사람 표기 bbox. 경계 마스크가 아니며 D455 실측도 아님. Pi 단일 보드 2면만으로 전체 Raspberry Pi 성능을 대표할 수 없음.",fontsize=10,color="#414a55")
    fig.subplots_adjust(left=.02,right=.98,bottom=.105,top=.885,hspace=.23,wspace=.07)
    fig.savefig(output/"wacv_coordinate_review.png",dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"component_assets")
    args=parser.parse_args(); output=args.output.resolve()
    records,source_counts,excluded,border_zero=audit(output/"wacv_original"/"pcb_wacv_2019")
    groups=freeze_split(records)
    dataset,tiles=create_data(output,records)
    counts={s: {"board_groups": len(gs), "images":sum(r["split"]==s for r in records),
                "unique_instances":dict(collections.Counter(o["class_name"] for r in records if r["split"]==s for o in r["objects"]))}
            for s,gs in groups.items()}
    exposure=collections.Counter(NAMES[o["class_id"]] for t in tiles for o in t["objects"])
    payload={"schema_version":1,"source":"WACV 2019 PCB component detection","source_url":SOURCE_URL,
             "license_status":"No explicit license in source page/archive; author provides unrestricted public download and requests citation. Local research copy; redistribution rights not verified.",
             "annotation_kind":"human-source axis-aligned bounding boxes, NOT masks", "class_names":NAMES,
             "source_type_to_class_id":MAPPING,"coordinate_convention":"zero-based half-open xyxy; VOC xmin/ymin minus one, xmax/ymax retained; raw source coordinates preserved",
             "group_policy":"board name with final _Top/_Bottom and index removed; inferred board identity, not verified serial number; crop/face variants stay together",
             "split_seed":42,"split_groups":groups,"native_split_counts":counts,
             "source_object_counts":dict(source_counts),"excluded_source_type_counts":dict(excluded),
             "source_zero_minimum_coordinate_objects":border_zero,
             "negative_scope":"Other source classes are background for this target4 task. Unknown items remain unknown; no guessed R/C labels. Annotation completeness/unknown ambiguity requires expert review.",
             "weighting":{"class_loss_weights":[1,1,1,1],"applied_special_class_weights":False,"sampler":"uniform generated tile sampling; original board/object exposure is not uniform", "oversampling":False},
             "train_tiling":{"tile_size":1024,"overlap":.2,"stride":819,"edge_alignment":True,"native_pixels":True,"resize_before_crop":False,"positive_clip_min_width_height_px":1,"drop_tiny_positives":False,"all_background_tiles_retained":True,"tile_count":len(tiles),"negative_tiles":sum(not t["objects"] for t in tiles),"class_exposure_counts":dict(exposure),"clipped_exposures":sum(o["clipped"] for t in tiles for o in t["objects"])},
             "images":records}
    save_json(output/"component_records.json",payload)
    save_json(output/"component_tile_records.json",{"source":"component_records.json", "tiles":tiles})
    summary={k:v for k,v in payload.items() if k!="images"}
    summary["image_sha256_duplicates"]=0
    summary["invalid_source_boxes"]=0
    summary["source_image_xml_dimension_mismatches"]=0
    summary["board_tile_counts"]=dict(collections.Counter(t["board_group"] for t in tiles))
    save_json(output/"component_data_manifest.json",summary)
    yaml=f"# WACV target4 HUMAN BBOX; no instance masks. Pi3B entirely held out.\npath: {dataset.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"+"".join(f"  {i}: {n}\n" for i,n in enumerate(NAMES))
    (output/"componentdetectdata.yaml").write_text(yaml,encoding="utf-8")
    (output/"componentdetect_pi_test.yaml").write_text(yaml.replace("test: images/test", "test: images/pi_test"),encoding="utf-8")
    save_json(output/"component_training_recommendation.json",{"task":"detect","model":"yolo11n.pt", "epochs":20,"imgsz":1024,"batch":2,"amp":False,"nbs":8,"warmup_bias_lr":0.0,"seed":42,"deterministic":True,"workers":0,"data":str(output/"componentdetectdata.yaml"),"status":"proposal only; this preparation script never trains or evaluates a model","class_weights":[1,1,1,1],"hyperparameter_selection":"validation boards only; native and tiled inference can be compared on val. Both general test and Pi test remain untouched until choice frozen."})
    draw_review(output, records)
    # Logical sum conservatively counts local hardlinks twice.
    logical_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    save_json(output/"storage_budget.json",{"initial_cap_bytes":1_000_000_000,"logical_bytes_including_hardlinks":logical_bytes,"within_cap":logical_bytes<=1_000_000_000})
    if logical_bytes>1_000_000_000:
        raise RuntimeError(f"Initial 1GB cap exceeded: {logical_bytes}")
    print(json.dumps({"counts":counts,"tiles":len(tiles),"exposures":dict(exposure),"logical_bytes":logical_bytes},indent=2))


if __name__=="__main__":
    main()
