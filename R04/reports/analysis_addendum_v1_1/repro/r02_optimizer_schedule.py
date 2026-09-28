"""R02 - re-derive the optimizer-call schedule and compare with epochs.json.

Rule copied from the installed Ultralytics 8.4.120 engine/trainer.py
(_get_warmup_iterations, `if ni < nw` accumulate interpolation, stepping when
ni - last_opt_step >= accumulate, per-batch `if self.stop: break`) and the R04
cap of 1,165 direct optimizer calls. Also checks the per-group draw schedule.
"""
import numpy as np

from common import RECORD, VENV_SITE, Inputs, sha256, write_output


def schedule(nb, batch, nbs, warmup_epochs, epochs, cap):
    nw = round(min(warmup_epochs, max(epochs - 1, 0)) * nb)
    accumulate, last, calls, per_epoch = max(round(nbs / batch), 1), -1, 0, []
    for epoch in range(epochs):
        processed = 0
        for i in range(nb):
            ni = i + nb * epoch
            if ni < nw:
                accumulate = max(1, int(np.interp(ni, [0, nw], [1, nbs / batch]).round()))
            processed += 1
            if ni - last >= accumulate:
                calls += 1
                last = ni
            if calls >= cap:          # on_train_batch_end sets stop; trainer breaks after this batch
                per_epoch.append({"epoch": epoch + 1, "calls": calls, "batches": processed})
                return nw, per_epoch
        per_epoch.append({"epoch": epoch + 1, "calls": calls, "batches": processed})
    return nw, per_epoch


def main():
    inp = Inputs()
    inp.add(VENV_SITE / "ultralytics/engine/trainer.py")
    protocol = inp.json(RECORD / "protocol.json")
    res = {}
    for arm in ("baseline", "improved"):
        start = inp.json(RECORD / f"runs/{arm}/start.json")
        epochs = inp.json(RECORD / f"runs/{arm}/epochs.json")
        summary = inp.json(RECORD / f"runs/{arm}/training_summary.json")
        nw, derived = schedule(start["loader_batches_per_epoch"], start["actual_batch"], protocol["nbs"],
                               protocol["warmup_epochs"], protocol["max_epochs_per_arm"],
                               protocol["max_direct_optimizer_calls_per_arm"])
        recorded = [{"epoch": e["epoch"], "calls": e["optimizer_calls"], "batches": e["processed_batches"],
                     "ema_updates": e["ema_updates"], "partial": e["partial_epoch"]} for e in epochs]
        res[arm] = {"warmup_iterations": nw, "derived": derived, "recorded": recorded,
                    "derived_equals_recorded": [(d["epoch"], d["calls"], d["batches"]) for d in derived]
                    == [(r["epoch"], r["calls"], r["batches"]) for r in recorded],
                    "ema_equals_calls": all(r["ema_updates"] == r["calls"] for r in recorded),
                    "total_batches": summary["actual_batches"], "draws": summary["actual_batches"] * start["actual_batch"],
                    "last_epoch_partial": summary["last_epoch_partial"]}
    groups = {arm: inp.json(RECORD / f"runs/{arm}/sampled_exposures.json")["groups"] for arm in ("baseline", "improved")}
    res["group_draws_identical_between_arms"] = groups["baseline"] == groups["improved"]
    res["group_draws"] = dict(sorted(groups["baseline"].items(), key=lambda kv: kv[1]))
    write_output("r02_optimizer_schedule", {
        "trainer_sha256": sha256(VENV_SITE / "ultralytics/engine/trainer.py"),
        "rule": "nw=round(min(warmup_epochs,epochs-1)*nb); if ni<nw: accumulate=max(1,int(interp(ni,[0,nw],[1,nbs/batch]).round())); step when ni-last>=accumulate; stop after the batch reaching 1165 calls"},
        res, inp)


if __name__ == "__main__":
    main()
