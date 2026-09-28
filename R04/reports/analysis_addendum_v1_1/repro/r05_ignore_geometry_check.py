"""R05 - CPU check of the R04 ignore anchor mask with non-square, asymmetric boxes.

Imports the frozen ignore_adapter.py from the GitHub record (byte-identical to the
local run, see R01) with the installed Ultralytics 8.4.120. Compares the
suppressed-anchor set with a NumPy re-implementation of
"anchor centre inside an ignore box AND not inside any positive box, and not a
TAL foreground anchor". Also checks zero class gradient inside, stock equality
without ignore boxes, and that a 2 px ignore box suppresses nothing.
No training, no GPU, no data access.
"""
import os
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from torch import nn

from common import RECORD, VENV_SITE, Inputs, sha256, write_output

sys.path.insert(0, str(RECORD / "scripts"))
from ignore_adapter import IgnoreDetectionLoss  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402

H, W = 64, 128                                   # letterboxed input, non-square
SIZES = [(H // s, W // s) for s in (8, 16, 32)]
A = sum(h * w for h, w in SIZES)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(1))
        self.model = [SimpleNamespace(stride=torch.tensor([8., 16., 32.]), nc=4, reg_max=16)]
        self.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)


def preds(seed=3):
    g = torch.Generator().manual_seed(seed)
    return {"feats": [torch.zeros(1, 1, h, w) for h, w in SIZES],
            "boxes": torch.randn(1, 64, A, generator=g, requires_grad=True),
            "scores": torch.randn(1, 4, A, generator=g, requires_grad=True)}


def batch(rows):
    return {"cls": torch.tensor([[r[0]] for r in rows], dtype=torch.float32).reshape(-1, 1),
            "batch_idx": torch.zeros(len(rows)),
            "bboxes": torch.tensor([r[1:] for r in rows], dtype=torch.float32).reshape(-1, 4)}


def expected(ignores, positives):
    centres = np.array([((x + .5) * s, (y + .5) * s) for s, (h, w) in zip((8, 16, 32), SIZES)
                        for y in range(h) for x in range(w)])

    def inside(boxes):
        m = np.zeros(len(centres), bool)
        for cx, cy, bw, bh in boxes:
            x1, y1, x2, y2 = (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H
            m |= (centres[:, 0] >= x1) & (centres[:, 0] <= x2) & (centres[:, 1] >= y1) & (centres[:, 1] <= y2)
        return m
    return inside(ignores) & ~inside(positives)


def main():
    inp = Inputs()
    inp.add(RECORD / "scripts/ignore_adapter.py")
    inp.add(VENV_SITE / "ultralytics/utils/loss.py")
    cases = {"tall_left_strip": ([(0.125, 0.5, 0.25, 1.0)], []),
             "wide_top_sliver": ([(0.5, 0.125, 1.0, 0.25)], []),
             "corner_box": ([(0.8, 0.8, 0.3, 0.3)], []),
             "ignore_with_positive_inside": ([(0.5, 0.5, 1.0, 1.0)], [(0.25, 0.3, 0.2, 0.3)])}
    rows = []
    for name, (ignores, positives) in cases.items():
        crit = IgnoreDetectionLoss(FakeModel())
        p = preds()
        info, loss, _ = crit.get_assigned_targets_and_loss(p, batch([(1, *b) for b in positives] + [(-1, *b) for b in ignores]))
        loss.sum().backward()
        got = crit.bce.last_suppressed_anchor[0].numpy().astype(bool)
        fg = info[0][0].numpy().astype(bool)
        grad = p["scores"].grad[0].abs().sum(0).numpy()
        want = expected(ignores, positives) & ~fg
        rows.append({"case": name, "suppressed": int(got.sum()), "anchors": A, "equals_numpy_reference": bool(np.array_equal(got, want)),
                     "zero_class_gradient_inside": bool(grad[got].sum() == 0), "nonzero_gradient_outside": bool((grad[~got] > 0).all()),
                     "foreground_never_suppressed": bool(not (got & fg).any())})
    crit, stock = IgnoreDetectionLoss(FakeModel()), v8DetectionLoss(FakeModel())
    _, a, _ = crit.get_assigned_targets_and_loss(preds(), batch([(2, .3, .6, .2, .4)]))
    _, b, _ = stock.get_assigned_targets_and_loss(preds(), batch([(2, .3, .6, .2, .4)]))
    tiny = IgnoreDetectionLoss(FakeModel())
    tiny.get_assigned_targets_and_loss(preds(), batch([(-1, .5, .5, 2 / W, 2 / H)]))
    write_output("r05_ignore_geometry_check", {
        "input_shape": f"{H}x{W} (H x W), strides 8/16/32, anchor centres (i+0.5)*stride",
        "reference": "inclusive centre-in-box test against ignore boxes minus positive boxes, minus TAL foreground",
        "adapter_sha256": sha256(RECORD / "scripts/ignore_adapter.py")},
        {"cases": rows, "all_cases_pass": all(all(v for k, v in r.items() if isinstance(v, bool)) for r in rows),
         "positive_only_equals_stock_loss": bool(torch.equal(a, b)),
         "tiny_2px_ignore_suppressed_anchors": int(tiny.bce.last_suppressed_anchor.sum()) if tiny.bce.last_suppressed_anchor is not None else None},
        inp)


if __name__ == "__main__":
    main()
