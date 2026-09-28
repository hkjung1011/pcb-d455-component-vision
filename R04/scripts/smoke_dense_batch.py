"""One authorized dense R04 CUDA forward/backward; never optimizer.step/train.

Run only after the root agent has established exclusive GPU access.  This uses
the exact four improved views drawn at epoch 1, batch 65 by the frozen sampler.
It checks the highest positive-per-image dimension (543) that drives padded TAL
assignment memory.  A CPU fallback inside TAL is explicitly recorded and is not
silently treated as a successful GPU-only stress check.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runtime_config"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from ultralytics import YOLO, __version__
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ignore_adapter import IgnoreYOLODataset, IgnoreDetectionLoss

DENSE_VIEW_IDS = [
    "ML365_Bottom__0019_context2048",
    "Spartan3__0068_native1024",
    "Spartan6Redux_Bottom__0000_native1024",
    "ATTIOT_Bottom__0024_object_centered",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_sha(model):
    hasher = hashlib.sha256()
    for name, parameter in model.named_parameters():
        hasher.update(name.encode())
        hasher.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


class AssignmentLogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        message = record.getMessage()
        if "TaskAlignedAssigner" in message or "out of memory" in message.lower():
            self.messages.append(message)


def run_cuda():
    """Return scalar-only evidence so all CUDA references can then be freed."""
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    initial = Path(protocol["initial_weights"])
    if sha(initial) != protocol["initial_weights_sha256"]:
        raise ValueError("Frozen official initial checkpoint SHA mismatch")
    data_path = ROOT / "data/improved/data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["nc"] = len(data["names"])
    data.setdefault("channels", 3)
    views_path = Path(data["r04_views_manifest"])
    records = json.loads(views_path.read_text(encoding="utf-8"))["records"]
    by_id = {row["id"]: row for row in records}
    dense = [by_id[key] for key in DENSE_VIEW_IDS]
    source_counts = [{"id": row["id"], "positive": len(row["targets"]),
                      "ignore": len(row["ignore_regions"])} for row in dense]
    assert [row["positive"] for row in source_counts] == [543, 6, 34, 84]
    assert [row["ignore"] for row in source_counts] == [3, 1, 38, 8]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU is not a substitute for this memory check")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    hyp = get_cfg(overrides=dict(
        imgsz=1024, batch=4, amp=False, nbs=8, workers=0, deterministic=True,
        mosaic=0., mixup=0., cutmix=0., copy_paste=0., degrees=0., translate=0.,
        scale=0., shear=0., perspective=0., multi_scale=0., fliplr=.5, flipud=.5,
        hsv_h=.005, hsv_s=.15, hsv_v=.2, bgr=0., box=7.5, cls=.5, dfl=1.5))
    dataset = IgnoreYOLODataset(
        img_path=str(Path(data["path"]) / data["train"]), imgsz=1024,
        batch_size=4, augment=True, hyp=hyp, rect=False, cache=False,
        data=data, task="detect", fraction=1.0)
    indices = {Path(path).stem: i for i, path in enumerate(dataset.im_files)}
    batch = dataset.collate_fn([dataset[indices[key]] for key in DENSE_VIEW_IDS])
    actual_positive = int((batch["cls"] >= 0).sum())
    actual_ignore = int((batch["cls"] == -1).sum())
    assert actual_positive == 667 and actual_ignore == 50
    positive_per_image = [int(((batch["batch_idx"] == i) & (batch["cls"].flatten() >= 0)).sum()) for i in range(4)]
    assert positive_per_image == [543, 6, 34, 84]

    pretrained = YOLO(str(initial)).model
    model = DetectionModel(deepcopy(pretrained.yaml), nc=4, verbose=False)
    model.names = data["names"]
    model.load(pretrained)
    del pretrained
    model.args = hyp
    model = model.cuda().train()
    model.criterion = IgnoreDetectionLoss(model)
    before = parameter_sha(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    batch["img"] = batch["img"].float() / 255
    loss, parts = model(batch)
    if not torch.isfinite(loss).all():
        raise RuntimeError("Nonfinite dense forward loss")
    loss.sum().backward()
    torch.cuda.synchronize()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Missing or nonfinite dense backward gradients")
    if not any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients):
        raise RuntimeError("All dense gradients are zero")
    after = parameter_sha(model)
    if before != after:
        raise RuntimeError("Parameters changed although no optimizer step was authorized")
    return {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "ultralytics": __version__,
        "batch": 4, "imgsz": 1024, "dtype": "float32", "optimizer_steps": 0,
        "forward_calls": 1, "backward_calls": 1, "finite_gradients": True,
        "sampler_location": {"epoch": 1, "batch": 65, "seed": 42},
        "sample_source_counts": source_counts, "positive_labels": actual_positive,
        "ignore_labels": actual_ignore, "positive_per_image": positive_per_image,
        "suppressed_negative_anchors": int(model.criterion.bce.last_suppressed_anchor.sum().item()),
        "loss_items": {key: float(value) for key, value in parts.items()},
        "parameter_sha256_before": before, "parameter_sha256_after": after,
        "parameters_unchanged": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocated_before_cleanup_bytes": torch.cuda.memory_allocated(),
        "initial_checkpoint_sha256": sha(initial), "data_yaml_sha256": sha(data_path),
        "views_sha256": sha(views_path), "sidecar_sha256": dataset.ignore_sidecar_sha256,
        "scope": "One fresh forward/backward without AdamW state or EMA memory. Dense TAL input check, not a training run, AP result or camera test.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/dense_batch_smoke.json")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Existing dense smoke report is preserved; use a distinct output for a deliberate retest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha(__file__), "ignore_adapter_sha256": sha(Path(__file__).with_name("ignore_adapter.py"))}
    capture = AssignmentLogCapture()
    LOGGER.addHandler(capture)
    stopped = threading.Event()
    progress = ROOT.parent / "r04_review/progress.log"
    def announce():
        while not stopped.wait(30):
            with progress.open("a", encoding="utf-8") as handle:
                handle.write("Dense batch smoke running: 667 positives + 50 ignore regions; no optimizer step.\n")
    threading.Thread(target=announce, daemon=True).start()
    error = None
    try:
        report.update(run_cuda())
        report["status"] = "PASS_DENSE_BATCH_FORWARD_BACKWARD"
    except BaseException as exc:
        report.update(status="FAIL_DENSE_BATCH", error=f"{type(exc).__name__}: {exc}")
        error = exc
    finally:
        stopped.set()
        LOGGER.removeHandler(capture)
        fallback = any("TaskAlignedAssigner" in message and "using CPU" in message for message in capture.messages)
        report["assignment_log_messages"] = capture.messages
        report["tal_cpu_fallback_observed"] = fallback
        if fallback and error is None:
            report["status"] = "COMPLETED_WITH_TAL_CPU_FALLBACK_REVIEW_REQUIRED"
        gc.collect()
        if torch.cuda.is_available():
            report.setdefault("peak_allocated_bytes", torch.cuda.max_memory_allocated())
            report.setdefault("peak_reserved_bytes", torch.cuda.max_memory_reserved())
            torch.cuda.empty_cache()
            report["allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated()
            report["reserved_after_cleanup_bytes"] = torch.cuda.memory_reserved()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if error is not None:
        raise error
    if report["tal_cpu_fallback_observed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
