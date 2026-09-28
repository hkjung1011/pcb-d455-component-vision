"""Plot recorded R03 training and native evaluation results, without inference.

Missing results are recorded as PENDING, never filled with estimated values.
Example: python scripts/make_r03_result_figures.py --suite reports/evaluation_suite/aggregate.json
Use --dry-run to inspect available inputs without writing figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ("parts_yolo11n", "parts_yolo11s", "board_yolo11n", "board_yolo11n_seg")
MODEL_LABELS = {"parts_yolo11n": "부품 YOLO11n", "parts_yolo11s": "부품 YOLO11s",
                "board_yolo11n": "보드 YOLO11n", "board_yolo11n_seg": "보드 YOLO11n-seg"}
PART_NAMES = ["resistor", "capacitor", "ic", "connector"]
PART_KO = ["저항", "콘덴서", "IC", "커넥터"]
BG = "#f5f7fb"; INK = "#1d2d44"; MUTED = "#516478"
BLUE = "#2468ba"; TEAL = "#008875"; ORANGE = "#db8b22"; RED = "#b34453"
COLORS = [BLUE, TEAL, ORANGE, RED]
AP100 = "ap50_95_max100"; AP300 = "ap50_95_max300"
CITATIONS = {
    "WACV2019": {"title": "Kuo et al., PCB component detection, WACV 2019 (author page)",
        "url": "https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection",
        "license": "Explicit reuse/redistribution license not confirmed; local research copy",
        "annotation": "Source component bbox; no component instance masks"},
    "IoTKITs2025": {"title": "IoTKITs v1, Mendeley Data, DOI 10.17632/x5thzmkxhy.1",
        "url": "https://data.mendeley.com/datasets/x5thzmkxhy/1", "license": "CC BY 4.0 (dataset level)",
        "annotation": "Source board polygons; per-image upstream internet provenance may be unknown"},
    "MicroPCB": {"title": "frettapper / micropcb-images",
        "url": "https://www.kaggle.com/datasets/frettapper/micropcb-images", "license": "CC BY 4.0",
        "annotation": "Corrected binary board mapping G/H/M; old A/H/I baseline excluded"},
    "UltralyticsYOLO11": {"title": "Ultralytics YOLO11 official model documentation",
        "url": "https://docs.ultralytics.com/models/yolo11/", "code_url": "https://github.com/ultralytics/ultralytics",
        "license": "AGPL-3.0", "license_url": "https://github.com/ultralytics/ultralytics/blob/main/LICENSE"},
    "COCOeval": {"title": "COCO API / official evaluation implementation",
        "url": "https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py",
        "annotation": "AP uses native source-image GT and maxDets 100/300; no Commons drafts"},
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def ratio(value):
    number = finite(value)
    if number is not None and not 0 <= number <= 1:
        raise ValueError(f"Ratio outside [0,1]: {number}")
    return number


def pct(value):
    number = ratio(value)
    return None if number is None else 100 * number


def configure():
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 11,
        "axes.titleweight": "bold", "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def pending(provenance, item, reason):
    provenance["pending"].append({"item": item, "reason": reason})


def track(provenance, path, role):
    path = Path(path).resolve()
    value = {"path": str(path), "sha256": sha(path), "role": role}
    if value not in provenance["inputs"]:
        provenance["inputs"].append(value)


def figure(size, title, subtitle):
    fig = plt.figure(figsize=size)
    fig.text(.055, .965, title, fontsize=21, weight="bold", va="top")
    fig.text(.055, .915, subtitle, fontsize=11, color=MUTED, va="top")
    return fig


def footer(fig, text, sources):
    fig.text(.055, .052, text, fontsize=9.5, color=MUTED, va="bottom", linespacing=1.45)
    fig.text(.055, .022, "출처: " + " · ".join(f"[{s}]" for s in sources), fontsize=9, color=MUTED)


def save(fig, out, name, provenance, sources, inputs=None):
    files = []
    for ext in ("png", "svg"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=.15)
        files.append({"path": str(path.resolve()), "sha256": sha(path)})
    plt.close(fig)
    provenance["figures"].append({"name": name, "files": files, "citation_handles": sources,
                                  "input_paths": inputs or []})


def style(ax, percent=False):
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=.17)
    ax.set_axisbelow(True)
    if percent:
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("%")


def grouped(ax, labels, series, colors=None, percent=True):
    """None is displayed as N/A without drawing a zero-height bar."""
    positions = np.arange(len(labels))
    width = .76 / max(1, len(series))
    for i, (name, values) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        color = (colors or COLORS)[i % len(colors or COLORS)]
        for j, value in enumerate(values):
            height = pct(value) if percent else finite(value)
            if height is None:
                ax.text(positions[j] + offset, 2 if percent else .02, "N/A", ha="center", va="bottom", fontsize=8, color=MUTED)
                continue
            ax.bar(positions[j] + offset, height, width*.9, label=name if j == 0 else None, color=color)
            ax.annotate(f"{height:.1f}", (positions[j]+offset, height), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .35, "alpha": .92})
        # An all-N/A first category must not silently erase the legend entry.
        if values and values[0] is None:
            ax.plot([], [], color=color, linewidth=8, label=name)
    ax.set_xticks(positions, labels)
    style(ax, percent)
    ax.legend(fontsize=9, loc="upper left", framealpha=.9)


def training_figures(root, out, provenance):
    for model_id in MODEL_IDS:
        run = root / "runs" / model_id
        csv_path = run / "fit" / "results.csv"
        update_path = run / "epoch_updates.json"
        if not csv_path.is_file() or not update_path.is_file():
            pending(provenance, "training:" + model_id, "results.csv or epoch_updates.json not available")
            continue
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = [{k.strip(): finite(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        rows = [r for r in rows if r.get("epoch") is not None]
        updates = read(update_path)
        updates = [r for r in updates if finite(r.get("epoch")) is not None]
        if not rows or not updates:
            pending(provenance, "training:" + model_id, "No recorded epochs")
            continue
        track(provenance, csv_path, "training monitor CSV, not native test AP")
        track(provenance, update_path, "Actual cumulative optimizer step counters")
        actual_epochs = {int(r["epoch"]) for r in rows}
        extra_callbacks = [r for r in updates if int(r["epoch"]) not in actual_epochs]
        if extra_callbacks:
            provenance.setdefault("presentation_notes", []).append({"model_id": model_id,
                "item": "Optimizer callbacks outside CSV training epochs are excluded from the epoch curve",
                "excluded_callback_epochs": [r["epoch"] for r in extra_callbacks],
                "reason": "A final evaluation callback is not another training epoch; source JSON preserved"})
        updates = [r for r in updates if int(r["epoch"]) in actual_epochs]
        if not updates:
            raise ValueError(f"No optimizer records align with actual training CSV epochs: {update_path}")
        summary_path = run / "run_summary.json"
        summary = read(summary_path) if summary_path.is_file() else {}
        if summary_path.is_file():
            track(provenance, summary_path, "Training completion status")
        finished = summary.get("status") == "TRAINED_OFFLINE_NOT_D455_VALIDATED"
        status = "완료 기록" if finished else "기록된 구간만 표시 · 실행 상태는 run_summary 확인"
        fig = figure((18, 6.8), MODEL_LABELS[model_id] + " 학습 기록",
            f"{int(max(r['epoch'] for r in rows))} epoch 기록 · {status} · 실제 저장된 CSV와 optimizer 상태에서 산출")
        axes = fig.subplots(1, 3)
        fig.subplots_adjust(left=.065, right=.975, top=.80, bottom=.24, wspace=.30)
        epochs = [r["epoch"] for r in rows]
        for i, key in enumerate(("box", "cls", "dfl", "seg")):
            for phase, linestyle in (("train", "-"), ("val", "--")):
                field = f"{phase}/{key}_loss"
                if not any(r.get(field) is not None for r in rows):
                    continue
                axes[0].plot(epochs, [r.get(field) if r.get(field) is not None else np.nan for r in rows],
                             color=COLORS[i], linestyle=linestyle, label=f"{phase} {key}", linewidth=1.6)
        axes[0].set_title("손실 · 실선 train / 점선 val")
        axes[0].set_ylabel("손실 값")
        axes[0].legend(fontsize=8, ncol=2)
        losses = [value for row in rows for key, value in row.items() if key.endswith("_loss") and value is not None]
        if losses and min(losses) > 0 and max(losses) > 40 * float(np.median(losses)):
            axes[0].set_yscale("log")
            axes[0].set_ylabel("손실 값 (로그축)")
            axes[0].text(.03, .05, f"급증 포함 전체 표시 · 최대 {max(losses):,.2f}",
                         transform=axes[0].transAxes, fontsize=8, color=MUTED)
        for i, (field, label) in enumerate((("metrics/mAP50(B)", "bbox mAP50"),
                ("metrics/mAP50-95(B)", "bbox mAP50–95"), ("metrics/mAP50(M)", "mask mAP50"),
                ("metrics/mAP50-95(M)", "mask mAP50–95"))):
            if any(r.get(field) is not None for r in rows):
                axes[1].plot(epochs, [pct(r.get(field)) if r.get(field) is not None else np.nan for r in rows],
                             label=label, color=COLORS[i], linewidth=1.8)
        axes[1].set_title("학습 중 VAL 모니터")
        axes[1].set_ylim(0, 100); axes[1].set_ylabel("mAP (%)"); axes[1].legend(fontsize=8)
        ue = [r["epoch"] for r in updates]
        lo = [finite(r.get("optimizer_steps_min")) for r in updates]
        hi = [finite(r.get("optimizer_steps_max")) for r in updates]
        if any(v is None for v in lo+hi):
            raise ValueError(f"Invalid recorded optimizer counters: {update_path}")
        if any(b < a for a, b in zip(hi, hi[1:])):
            raise ValueError(f"Optimizer counter decreased without a recorded reset: {update_path}")
        axes[2].fill_between(ue, lo, hi, color=TEAL, alpha=.25, label="매개변수별 최소–최대")
        axes[2].plot(ue, hi, color=TEAL, linewidth=2, label="누적 최대 step")
        axes[2].set_title("실제 optimizer 갱신")
        axes[2].set_ylabel("누적 step 수"); axes[2].legend(fontsize=8)
        axes[2].text(.98, .08, f"마지막 {lo[-1]:,.0f}–{hi[-1]:,.0f} step", transform=axes[2].transAxes, ha="right", color=TEAL, weight="bold")
        for ax in axes:
            style(ax); ax.set_xlabel("epoch")
            ax.set_xticks(sorted(actual_epochs) if len(actual_epochs) <= 6 else [e for e in sorted(actual_epochs) if e == 1 or e % 5 == 0])
        sources = ["WACV2019", "UltralyticsYOLO11"] if model_id.startswith("parts") else ["IoTKITs2025", "UltralyticsYOLO11"]
        if model_id == "board_yolo11n":
            sources.insert(0, "MicroPCB")
        footer(fig, "VAL 모니터는 학습용 loader의 기록입니다. 최종 원본 좌표·타일 결합 평가는 별도 그래프에서 비교합니다.\noptimizer는 매개변수 상태의 누적 갱신 수입니다. CSV 학습 epoch만 표시하고 학습 종료 후 평가 callback은 제외했습니다.", sources)
        save(fig, out, "05_learning_"+model_id, provenance, sources, [str(csv_path), str(update_path)])


def resolve_metric_path(aggregate_path, compact):
    candidate = Path(compact["metrics_path"])
    alternatives = [candidate, aggregate_path.parent/candidate]
    # Portable reports may retain original absolute paths; a matching evidence
    # file under evaluations is accepted only when its recorded hash matches.
    alternatives += list((aggregate_path.parent/"evaluations").glob(candidate.parent.name+"/metrics.json"))
    for path in alternatives:
        if path.is_file() and sha(path) == compact.get("metrics_sha256"):
            return path.resolve()
    raise ValueError(f"Missing/changed native metrics evidence: {candidate}")


def load_suites(paths, provenance):
    models = {}
    for aggregate_path in paths:
        aggregate_path = Path(aggregate_path).resolve()
        if not aggregate_path.is_file():
            pending(provenance, str(aggregate_path), "Suite aggregate not available")
            continue
        aggregate = read(aggregate_path)
        selection_path = aggregate_path.parent / "selection.json"
        protocol_path = aggregate_path.parent / "protocol.json"
        if not selection_path.is_file() or sha(selection_path) != aggregate.get("selection_sha256"):
            raise ValueError(f"Missing/changed frozen selection evidence: {selection_path}")
        if not protocol_path.is_file():
            raise ValueError(f"Missing frozen protocol evidence: {protocol_path}")
        protocol = read(protocol_path)
        protocol_hash = fingerprint({k: protocol[k] for k in ("plan", "evidence")})
        if protocol_hash != protocol.get("protocol_sha256") or protocol_hash != aggregate.get("protocol_sha256"):
            raise ValueError(f"Changed frozen protocol payload: {protocol_path}")
        locked = read(selection_path)
        if locked.get("protocol_sha256") != protocol_hash or locked.get("test_results_used_for_selection") is not False:
            raise ValueError(f"Selection is not validation-only under this protocol: {selection_path}")
        for path in (selection_path, protocol_path):
            track(provenance, path, "Frozen validation selection/protocol")
        track(provenance, aggregate_path, "Evaluation suite aggregate")
        for model_id, model in aggregate.get("models", {}).items():
            if model_id not in MODEL_IDS:
                continue
            if model_id in models:
                raise ValueError(f"Duplicate model across suites; specify one authoritative suite per model: {model_id}")
            expected_names = PART_NAMES if model_id.startswith("parts_") else ["raspberry_pi_sbc"]
            model = json.loads(json.dumps(model))
            selection = model.get("selection", {})
            if selection != locked.get("selections", {}).get(model_id):
                raise ValueError(f"Aggregate selection differs from frozen evidence: {model_id}")
            for collection, metrics in (("validation", model.get("validation_candidates", [])), ("test", list(model.get("tests", {}).values()))):
                for compact in metrics:
                    path = resolve_metric_path(aggregate_path, compact)
                    full = read(path)
                    if full.get("class_names") != expected_names:
                        raise ValueError(f"Unexpected result ontology: {path}")
                    if full.get("split") != ("val" if collection == "validation" else "test"):
                        raise ValueError(f"Wrong split for {collection}: {path}")
                    if full.get("d455_verified") is not False:
                        raise ValueError(f"Expected offline result with explicit d455_verified=false: {path}")
                    if collection == "test" and (full.get("checkpoint_sha256") != selection.get("checkpoint_sha256")
                            or abs(full["operating"]["confidence"] - selection["operating_confidence"]) > 1e-9):
                        raise ValueError(f"Test differs from frozen checkpoint/confidence: {path}")
                    for key in ("bbox", "mask", "operating", "latency", "images", "groups", "native_gt_instances",
                                "checkpoint_sha256", "manifest_sha256", "group_bootstrap_recall95", "class_names"):
                        if key == "operating":
                            expected = {k: v for k, v in full[key].items() if k != "per_image"}
                        else:
                            expected = full.get(key)
                        if compact.get(key) != expected:
                            raise ValueError(f"Aggregate/native evidence mismatch {key}: {path}")
                    track(provenance, path, collection+" native metric evidence")
                    compact["_resolved_metrics_path"] = str(path)
            models[model_id] = model
    return models


def ap(metric, key=AP100, kind="bbox"):
    return ratio((metric.get(kind) or {}).get(key))


def context(metric):
    return f"{metric['images']:,}장 / {metric['groups']:,}그룹 / GT {metric['native_gt_instances']:,}개"


def val_parts(models, out, provenance):
    present = [m for m in MODEL_IDS[:2] if models.get(m, {}).get("validation_candidates")]
    if not present:
        pending(provenance, "06_parts_val_selection", "No native parts validation candidates")
        return
    fig = figure((16, 8.1), "부품 모델 · VAL에서 whole / tile 선택", "원본 좌표 GT로 평가 · best/last × 전체 사진/1024 타일 · ★는 저장된 VAL 선택 결과")
    axes = fig.subplots(1, len(present), squeeze=False)[0]
    fig.subplots_adjust(left=.065, right=.97, top=.80, bottom=.27, wspace=.22)
    inputs = []
    order = {name: i for i, name in enumerate(("best_whole", "best_tile1024", "last_whole", "last_tile1024"))}
    for ax, model_id in zip(axes, present):
        model = models[model_id]
        candidates = sorted(model["validation_candidates"], key=lambda r: order.get(r["candidate_id"], 99))
        selected = model["selection"]["candidate"]["candidate_id"]
        labels = [("★ " if r["candidate_id"] == selected else "")+r["candidate_id"].replace("_", "\n") for r in candidates]
        grouped(ax, labels, [("AP50–95 · 최대 100", [ap(r) for r in candidates]),
                             ("AP50–95 · 최대 300", [ap(r, AP300) for r in candidates])])
        ax.set_title(MODEL_LABELS[model_id]+"\n"+context(candidates[0]), fontsize=12)
        inputs.extend(r["_resolved_metrics_path"] for r in candidates)
    footer(fig, "선택 순서: VAL bbox AP50–95(최대 100) → AP50–95(최대 300) → 오프라인 지연시간. TEST는 선택에 사용하지 않았습니다.\n최대 100/300은 COCO 평가의 이미지·클래스별 검출 상한입니다. 두 수치는 별도로 계산했습니다.", ["WACV2019", "UltralyticsYOLO11", "COCOeval"])
    save(fig, out, "06_parts_val_selection", provenance, ["WACV2019", "UltralyticsYOLO11", "COCOeval"], inputs)


def board_test(models, out, provenance):
    present = [(m, models.get(m, {}).get("tests", {}).get("general_test")) for m in MODEL_IDS[2:]]
    present = [(m, r) for m, r in present if r]
    if not present:
        pending(provenance, "07_board_native_test", "No board native TEST results")
        return
    fig = figure((16, 7.8), "보드 전체 검출·분할 · 원본 TEST 결과", "Raspberry Pi 보드 1종 · 각 모델의 검증된 GT 범위에서 측정 · 부품 성능 순위와 구분")
    axes = fig.subplots(1, len(present), squeeze=False)[0]
    fig.subplots_adjust(left=.07, right=.97, top=.78, bottom=.25, wspace=.22)
    for ax, (model_id, metric) in zip(axes, present):
        kinds = ["bbox", "mask"] if metric.get("mask") else ["bbox"]
        grouped(ax, ["보드 상자" if k == "bbox" else "보드 외곽 mask" for k in kinds],
            [("AP50–95 · 최대 100", [ap(metric, AP100, k) for k in kinds]),
             ("AP50 · 최대 100", [ap(metric, "ap50_max100", k) for k in kinds])])
        ax.set_title(MODEL_LABELS[model_id]+"\n"+context(metric), fontsize=12)
    footer(fig, "검출 TEST는 수정 micro + IoTKITs, 분할 TEST는 native polygon이 있는 IoTKITs입니다. 서로 다른 시험 사진 집합입니다.\n보드 mask 결과이며 부품별 mask AP가 아닙니다. 학습 분할은 G/H/M 정정판을 사용하고 이전 A/H/I 성능은 제외했습니다.", ["MicroPCB", "IoTKITs2025", "UltralyticsYOLO11"])
    save(fig, out, "07_board_native_test", provenance, ["MicroPCB", "IoTKITs2025", "UltralyticsYOLO11"], [r["_resolved_metrics_path"] for _, r in present])


def part_test(models, out, provenance, cohort):
    present = [(m, models.get(m, {}).get("tests", {}).get(cohort)) for m in MODEL_IDS[:2]]
    present = [(m, r) for m, r in present if r]
    stem = "08_parts_general_test" if cohort == "general_test" else "09_parts_pi_test"
    if not present:
        pending(provenance, stem, "No native parts "+cohort+" results")
        return
    sets = {(r["manifest_sha256"], r["images"], r["groups"], r["native_gt_instances"]) for _, r in present}
    if len(sets) != 1:
        raise ValueError("Cannot directly compare models evaluated on different native GT: "+cohort)
    is_pi = cohort == "pi_test"
    title = "학습에서 제외한 Raspberry Pi 3B · 부품 TEST" if is_pi else "일반 보드 · 부품 4종 원본 TEST"
    subtitle = context(present[0][1])+" · 모델별 VAL 선택 checkpoint·처리 방식·confidence를 고정"
    fig = figure((16, 7.8), title, subtitle)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.065, right=.97, top=.80, bottom=.25, wspace=.24)
    labels = [MODEL_LABELS[m]+"\n"+models[m]["selection"]["candidate"]["candidate_id"] for m, _ in present]
    grouped(axes[0], labels, [("AP50–95 · 최대 100", [ap(r) for _, r in present]),
                            ("AP50–95 · 최대 300", [ap(r, AP300) for _, r in present]),
                            ("AP50 · 최대 100", [ap(r, "ap50_max100") for _, r in present])])
    axes[0].set_title("bbox AP · confidence 범위를 통합")
    grouped(axes[1], labels, [("precision", [r["operating"]["precision"] for _, r in present]),
                             ("recall", [r["operating"]["recall"] for _, r in present])])
    axes[1].set_title("VAL 고정 confidence · bbox IoU ≥ 0.5")
    for j, (_, r) in enumerate(present):
        ci = r.get("group_bootstrap_recall95")
        if ci and ci.get("lower") is not None and ci.get("upper") is not None:
            # Percentile bounds need not bracket a point estimate in all datasets.
            axes[1].vlines(j+.19, pct(ci["lower"]), pct(ci["upper"]), color=INK, linewidth=1.5)
        axes[1].text(j, 106, f"conf={r['operating']['confidence']:.2f}", ha="center", fontsize=9, color=MUTED)
    note = ("Pi 3B 한 보드 그룹의 두 면입니다. 189개 source target4 정답 외 unknown 67개가 있어 모든 물리 부품 성능을 뜻하지 않습니다.\n한 그룹이므로 신뢰구간을 제시하지 않습니다. 사전학습 자료 포함 여부와 D455 현장 성능은 확인하지 않았습니다." if is_pi else
            "recall 세로선은 고정 confidence에서 원본 그룹을 1,000회 재표집한 95% 구간입니다. AP의 신뢰구간이 아닙니다.\n동일 그룹의 사진은 함께 묶었습니다. 실제 물리 시리얼 독립성과 D455 촬영 성능은 검증하지 않았습니다.")
    footer(fig, note, ["WACV2019", "UltralyticsYOLO11", "COCOeval"])
    save(fig, out, stem, provenance, ["WACV2019", "UltralyticsYOLO11", "COCOeval"], [r["_resolved_metrics_path"] for _, r in present])


def pi_breakdown(models, out, provenance):
    present = [(m, models.get(m, {}).get("tests", {}).get("pi_test")) for m in MODEL_IDS[:2]]
    present = [(m, r) for m, r in present if r]
    if not present:
        pending(provenance, "10_pi_class_and_size", "No Pi native results")
        return
    fig = figure((17, 11), "Raspberry Pi 3B · 클래스와 부품 크기별 결과", "source target4 bbox 기준 · AP와 고정 confidence recall은 서로 다른 지표")
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=.065, right=.97, top=.84, bottom=.17, wspace=.21, hspace=.42)
    count_sets = []
    for _, r in present:
        counts = r["operating"]["per_class_counts_tp_fp_fn"]
        if len(counts) != len(PART_NAMES):
            raise ValueError("Pi per-class counts do not match target4 ontology")
        count_sets.append([int(x[0]+x[2]) for x in counts])
    if any(c != count_sets[0] for c in count_sets):
        raise ValueError("Pi model class GT counts differ")
    labels = [f"{name}\nGT {n}" for name, n in zip(PART_KO, count_sets[0])]
    for ax, key, heading in ((axes[0, 0], AP100, "클래스 AP50–95 · 최대 100"),
                             (axes[0, 1], AP300, "클래스 AP50–95 · 최대 300")):
        grouped(ax, labels, [(MODEL_LABELS[m], [(r["bbox"].get("per_class", {}).get(c) or {}).get(key) for c in PART_NAMES]) for m, r in present])
        ax.set_title(heading, fontsize=12)
    recall_series = []
    for model_id, r in present:
        recall_series.append((MODEL_LABELS[model_id], [tp/(tp+fn) if tp+fn else None for tp, fp, fn in r["operating"]["per_class_counts_tp_fp_fn"]]))
    grouped(axes[1, 0], labels, recall_series)
    axes[1, 0].set_title("클래스 recall · bbox IoU ≥ 0.5", fontsize=12)
    bins = ("lt8", "8to16", "16to32", "ge32")
    totals = [[r["operating"]["size_recall"][b]["total"] for b in bins] for _, r in present]
    if any(t != totals[0] for t in totals):
        raise ValueError("Pi native size-bin GT counts differ")
    binlabels = [f"{label}\nGT {count}" for label, count in zip(("< 8 px", "8–<16 px", "16–<32 px", "≥ 32 px"), totals[0])]
    grouped(axes[1, 1], binlabels, [(MODEL_LABELS[m], [r["operating"]["size_recall"][b]["recall"] for b in bins]) for m, r in present])
    axes[1, 1].set_title("원본 bbox 짧은 변 크기별 recall", fontsize=12)
    footer(fig, "크기는 원본 사진의 bbox 짧은 변입니다. COCO area 구간이나 D455 촬영 시 픽셀 크기와 다릅니다. GT가 없는 구간은 N/A로 표시합니다.\n각 모델의 VAL 선택 confidence를 적용했습니다. 한 보드 그룹의 결과이므로 Pi 4/5/Zero 전체로 일반화할 수 없습니다.", ["WACV2019", "UltralyticsYOLO11", "COCOeval"])
    save(fig, out, "10_pi_class_and_size", provenance, ["WACV2019", "UltralyticsYOLO11", "COCOeval"], [r["_resolved_metrics_path"] for _, r in present])


def degradation(models, out, provenance):
    keys = ("pi_test", "pi_test_half_resolution", "pi_test_gaussian_sigma1")
    present = [(m, models.get(m, {}).get("tests", {})) for m in MODEL_IDS[:2]]
    present = [(m, tests) for m, tests in present if tests.get("pi_test") and any(tests.get(k) for k in keys[1:])]
    if not present:
        pending(provenance, "11_pi_simulated_degradation", "No actual degradation evaluation results")
        return
    for _, tests in present:
        for key in keys[1:]:
            r = tests.get(key)
            if r:
                simulation = r.get("simulation") or {}
                if not (simulation.get("camera_proxy_not_D455") is True and simulation.get("native_gt_unchanged") is True):
                    raise ValueError("Degradation result lacks explicit camera proxy / unchanged native GT evidence")
                if (r["images"], r["native_gt_instances"]) != (tests["pi_test"]["images"], tests["pi_test"]["native_gt_instances"]):
                    raise ValueError("Degradation GT population differs from original Pi test")
    fig = figure((16, 8.0), "입력 세부 정보 저하 · Raspberry Pi 모의 시험", "동일 원본 GT와 VAL 고정 선택 사용 · 실제 D455 카메라 시험이 아닌 이미지 변환 실험")
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.065, right=.97, top=.80, bottom=.26, wspace=.22)
    labels = ["원본", "가로·세로 1/2\n축소 후 원본 크기 복원", "Gaussian blur\nσ = 1 원본 px"]
    grouped(axes[0], labels, [(MODEL_LABELS[m], [ap(tests[k]) if tests.get(k) else None for k in keys]) for m, tests in present])
    grouped(axes[1], labels, [(MODEL_LABELS[m], [tests[k]["operating"]["recall"] if tests.get(k) else None for k in keys]) for m, tests in present])
    axes[0].set_title("bbox AP50–95 · 최대 100")
    axes[1].set_title("고정 confidence recall · bbox IoU ≥ 0.5")
    footer(fig, "축소: INTER_AREA 1/2 → INTER_LINEAR 원본 복원. 블러: Gaussian σ=1, 캔버스와 GT 좌표 보존. 두 변환은 각각 적용했습니다.\n모델·threshold를 다시 선택하지 않았습니다. 실제 렌즈·거리·조명·흔들림·D455 센서 특성의 검증을 대체하지 않습니다.", ["WACV2019", "UltralyticsYOLO11"])
    save(fig, out, "11_pi_simulated_degradation", provenance, ["WACV2019", "UltralyticsYOLO11"], [tests[k]["_resolved_metrics_path"] for _, tests in present for k in keys if tests.get(k)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--suite", type=Path, action="append", help="aggregate.json; repeat for disjoint model suites")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Read and validate evidence only; write no figures")
    args = parser.parse_args()
    root = args.root.resolve()
    suites = args.suite or [root/"reports"/"evaluation_suite"/"aggregate.json"]
    out = args.output or root/"reports"/"figures"
    provenance = {"schema": "r03-result-figures-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()), "generator_sha256": sha(__file__), "inputs": [],
        "figures": [], "pending": [], "citations": CITATIONS,
        "rules": ["No estimated/filled metric values", "No inference or model selection in this script",
            "Missing metrics skipped; undefined class/size-bin metrics are N/A, not zero",
            "Board and component AP are not combined into one ranking",
            "Training monitor AP is distinct from native-source evaluation AP",
            "Commons assistant draft annotations are excluded from quantitative evaluation",
            "Pi holdout is one fine-tuning-excluded source board group, not a multi-model Pi benchmark",
            "D455 physical camera performance remains unverified"]}
    models = load_suites(suites, provenance)
    if args.dry_run:
        print(json.dumps({"models_available": list(models), "pending": provenance["pending"],
            "training_available": {m: (root/"runs"/m/"fit"/"results.csv").is_file() for m in MODEL_IDS},
            "verified_evidence_files": len(provenance["inputs"])}, ensure_ascii=False, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    configure()
    training_figures(root, out, provenance)
    val_parts(models, out, provenance)
    board_test(models, out, provenance)
    part_test(models, out, provenance, "general_test")
    part_test(models, out, provenance, "pi_test")
    pi_breakdown(models, out, provenance)
    degradation(models, out, provenance)
    target = out/"result_figure_provenance.json"
    target.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"figures": [f["name"] for f in provenance["figures"]], "pending": provenance["pending"],
                      "provenance": str(target.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
