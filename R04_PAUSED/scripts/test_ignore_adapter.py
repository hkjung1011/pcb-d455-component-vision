"""Meaningful CPU gradient, geometry, and checkpoint tests for R04 ignore."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import ModelEMA

from ignore_adapter import IgnoreYOLODataset, IgnoreDetectionLoss, _anchors_inside_boxes, _IgnoreAwareBCE


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(1))
        self.model = [SimpleNamespace(stride=torch.tensor([8., 16., 32.]), nc=4, reg_max=16)]
        self.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)


def predictions(seed=7, batch_size=1):
    generator = torch.Generator().manual_seed(seed)
    feats = [torch.zeros(batch_size, 1, size, size) for size in [8, 4, 2]]
    return {"feats": feats,
            "boxes": torch.randn(batch_size, 64, 84, generator=generator, requires_grad=True),
            "scores": torch.randn(batch_size, 4, 84, generator=generator, requires_grad=True)}


def labels(rows):
    return {"cls": torch.tensor([[r[1]] for r in rows], dtype=torch.float32).reshape(-1, 1),
            "batch_idx": torch.tensor([r[0] for r in rows], dtype=torch.float32),
            "bboxes": torch.tensor([r[2:] for r in rows], dtype=torch.float32).reshape(-1, 4)}


def run_loss(criterion, preds, batch):
    info, loss, _ = criterion.get_assigned_targets_and_loss(preds, batch)
    loss.sum().backward()
    return info, loss.detach(), {key: None if preds[key].grad is None else preds[key].grad.clone() for key in ["scores", "boxes"]}


class IgnoreLossTests(unittest.TestCase):
    def test_no_ignore_matches_stock_loss_and_gradients(self):
        model = FakeModel()
        batch = labels([(0, 2, .5, .5, .5, .5)])
        _, stock, stock_grad = run_loss(v8DetectionLoss(model), predictions(), batch)
        _, custom, custom_grad = run_loss(IgnoreDetectionLoss(model), predictions(), batch)
        torch.testing.assert_close(stock, custom, rtol=0, atol=0)
        for name in stock_grad:
            torch.testing.assert_close(stock_grad[name], custom_grad[name], rtol=0, atol=0)

    def test_ignored_negative_gradient_zero_and_outside_unchanged(self):
        model = FakeModel()
        _, _, reference = run_loss(v8DetectionLoss(model), predictions(), labels([]))
        criterion = IgnoreDetectionLoss(model)
        _, _, gradients = run_loss(criterion, predictions(), labels([(0, -1, .25, .25, .5, .5)]))
        ignored = criterion.bce.last_suppressed_anchor[0]
        self.assertTrue(ignored.any()); self.assertTrue((~ignored).any())
        self.assertEqual(float(gradients["scores"][0, :, ignored].abs().sum()), 0)
        torch.testing.assert_close(gradients["scores"][0, :, ~ignored], reference["scores"][0, :, ~ignored], rtol=0, atol=0)

    def test_positive_regions_and_foreground_keep_stock_gradients(self):
        model = FakeModel()
        positive = (0, 2, .5, .5, .5, .5)
        stock_info, stock, ref = run_loss(v8DetectionLoss(model), predictions(), labels([positive]))
        criterion = IgnoreDetectionLoss(model)
        info, actual, grad = run_loss(criterion, predictions(), labels([positive, (0, -1, .5, .5, 1., 1.)]))
        foreground = info[0]
        self.assertTrue(foreground.any())
        self.assertFalse((criterion.bce.last_suppressed_anchor & foreground).any())
        torch.testing.assert_close(info[0], stock_info[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[[0, 2]], stock[[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(grad["boxes"], ref["boxes"], rtol=0, atol=0)
        kept = ~criterion.bce.last_suppressed_anchor[0]
        torch.testing.assert_close(grad["scores"][0, :, kept], ref["scores"][0, :, kept], rtol=0, atol=0)

    def test_assigned_anchor_positive_override_even_outside_gt_region(self):
        bce = _IgnoreAwareBCE(); bce.suppress_anchor = torch.ones(1, 2, dtype=torch.bool)
        pred = torch.zeros(1, 2, 4, requires_grad=True)
        target = torch.zeros_like(pred); target[0, 0, 2] = .5
        bce(pred, target).sum().backward()
        self.assertGreater(float(pred.grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(pred.grad[0, 1].abs().sum()), 0)

    def test_assigned_anchor_with_zero_soft_target_is_still_preserved(self):
        bce = _IgnoreAwareBCE(); bce.suppress_anchor = torch.ones(1, 2, dtype=torch.bool)
        bce.assigned_anchor = torch.tensor([[True, False]])
        pred = torch.zeros(1, 2, 4, requires_grad=True)
        bce(pred, torch.zeros_like(pred)).sum().backward()
        self.assertGreater(float(pred.grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(pred.grad[0, 1].abs().sum()), 0)

    def test_ignore_is_per_image_and_next_batch_does_not_reuse_mask(self):
        model = FakeModel(); criterion = IgnoreDetectionLoss(model)
        p = predictions(batch_size=2)
        run_loss(criterion, p, labels([(0, -1, .5, .5, 1., 1.)]))
        self.assertEqual(float(p["scores"].grad[0].abs().sum()), 0)
        self.assertGreater(float(p["scores"].grad[1].abs().sum()), 0)
        _, second, second_grad = run_loss(criterion, predictions(), labels([]))
        _, stock, stock_grad = run_loss(v8DetectionLoss(model), predictions(), labels([]))
        torch.testing.assert_close(second, stock, rtol=0, atol=0)
        torch.testing.assert_close(second_grad["scores"], stock_grad["scores"], rtol=0, atol=0)

    def test_invalid_negative_class_rejected(self):
        with self.assertRaises(ValueError):
            IgnoreDetectionLoss(FakeModel()).get_assigned_targets_and_loss(predictions(), labels([(0, -2, .5, .5, .5, .5)]))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="r04_ignore_test_")
        self.root = Path(self.tmp.name)
        self.image_dir = self.root / "images/train"; self.image_dir.mkdir(parents=True)
        self.label_dir = self.root / "labels/train"; self.label_dir.mkdir(parents=True)
        self.image = self.image_dir / "sample.png"
        Image.new("RGB", (100, 50), (80, 100, 120)).save(self.image)
        self.label = self.label_dir / "sample.txt"
        self.label.write_text("0 0.25 0.3 0.2 0.2\n", encoding="utf-8")
        self.label_hash = hashlib.sha256(self.label.read_bytes()).hexdigest()
        self.sidecar = self.root / "ignore.json"
        self.sidecar.write_text(json.dumps({"schema":"r04-training-ignore-v1", "images":{
            str(self.image.resolve()): [
                {"bbox_xywhn":[.25,.3,.2,.2], "reason":"source_unknown", "source_instance_id":"overlap"},
                {"bbox_xywhn":[.8,.5,.01,.02], "reason":"target_fragment_lt50", "source_instance_id":"one_pixel"}
            ]}}), encoding="utf-8")
        self.data={"names":{0:"resistor",1:"capacitor",2:"ic",3:"connector"},"nc":4,"channels":3,"r04_ignore_sidecar":str(self.sidecar)}
        self.hyp=get_cfg(overrides=dict(imgsz=128, mosaic=0., mixup=0., cutmix=0., copy_paste=0., degrees=0., translate=0., scale=0., shear=0., perspective=0., fliplr=1., flipud=1., hsv_h=0., hsv_s=0., hsv_v=0.))

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self):
        return IgnoreYOLODataset(img_path=str(self.image_dir), imgsz=128, batch_size=1, augment=True, hyp=deepcopy(self.hyp), rect=False, cache=False, data=self.data, task="detect", fraction=1.0)

    def test_letterbox_flip_and_collate_preserve_ignore_coordinates(self):
        dataset=self.dataset(); sample=dataset[0]
        torch.testing.assert_close(sample["cls"].flatten(), torch.tensor([0.,-1.,-1.]), rtol=0, atol=0)
        # 100x50 -> 128x64 + 32px top/bottom, then both flips.
        expected=torch.tensor([[.75,.60,.20,.10],[.75,.60,.20,.10],[.20,.50,.01,.01]])
        torch.testing.assert_close(sample["bboxes"],expected,rtol=0,atol=2e-7)
        batch=dataset.collate_fn([dataset[0],dataset[0]])
        torch.testing.assert_close(batch["batch_idx"],torch.tensor([0.,0.,0.,1.,1.,1.]),rtol=0,atol=0)
        self.assertEqual(hashlib.sha256(self.label.read_bytes()).hexdigest(),self.label_hash)

    def test_cache_and_validation_do_not_gain_negative_labels(self):
        self.dataset()
        stock=YOLODataset(img_path=str(self.image_dir),imgsz=128,batch_size=1,augment=False,hyp=deepcopy(self.hyp),rect=False,cache=False,data=self.data,task="detect")
        self.assertEqual(len(stock.labels[0]["cls"]),1)
        self.assertTrue((stock[0]["cls"]>=0).all())
        self.assertEqual(hashlib.sha256(self.label.read_bytes()).hexdigest(),self.label_hash)

    def test_sidecar_missing_image_and_unsupported_geometry_fail_closed(self):
        original=self.sidecar.read_text()
        self.sidecar.write_text(json.dumps({"schema":"r04-training-ignore-v1","images":{}}))
        with self.assertRaises(ValueError): self.dataset()
        self.sidecar.write_text(original)
        self.hyp.translate=.01
        with self.assertRaises(ValueError): self.dataset()


class CheckpointTests(unittest.TestCase):
    def test_standard_model_ema_snapshot_contains_no_custom_criterion(self):
        torch.set_num_threads(2)
        model=DetectionModel("yolo11n.yaml",nc=4,verbose=False)
        model.args=get_cfg(overrides=dict(box=7.5,cls=.5,dfl=1.5))
        ema=ModelEMA(model)
        model.criterion=IgnoreDetectionLoss(model)
        ema.update(model)
        self.assertNotIsInstance(getattr(ema.ema,"criterion",None),IgnoreDetectionLoss)
        # Exact current upstream checkpoint snapshot operation.
        snapshot=deepcopy(ema.ema).half().to(memory_format=torch.contiguous_format)
        if hasattr(snapshot,"criterion"): snapshot.criterion=None
        buffer=io.BytesIO(); torch.save({"model":snapshot},buffer)
        self.assertNotIn(b"ignore_adapter",buffer.getvalue())
        buffer.seek(0); recovered=torch.load(buffer,map_location="cpu",weights_only=False)["model"]
        self.assertIs(type(recovered),DetectionModel)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--report",type=Path,default=Path(__file__).resolve().parents[1]/"reports/ignore_adapter_tests.json")
    args=parser.parse_args(); torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    payload={"status":"PASS" if result.wasSuccessful() else "FAIL","utc":datetime.now(timezone.utc).isoformat(),"tests_run":result.testsRun,"failures":[str(v) for v in result.failures],"errors":[str(v) for v in result.errors],"device":"CPU","training_runs":0,"production_inference":0,"scope":"Synthetic loss/gradient equivalence, real dataset letterbox/flips/collate and on-disk cache isolation, standard checkpoint portability. No physical camera or AP verification.","source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('ignore_adapter.py')]}}
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=="__main__": main()
