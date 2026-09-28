"""R01 - review commit, GitHub/local file identity and recorded hash bindings.

Checks: archive HEAD and clean tree; byte identity of key files between the
GitHub pause record and the local run; protocol-recorded input hashes; the 22
completed validation metrics against snapshot.json, partial_validation.csv,
candidates.json and one evaluator/manifest; P/R/F1 arithmetic; the 20 candidate
checkpoint hashes; Ultralytics args.yaml differences between the arms.
"""
import csv
import json

import yaml

from common import ARCHIVE, R04, RECORD, REVIEW_COMMIT, Inputs, git, sha256, write_output

PAIRED = ["protocol.json", "stress_protocol.json",
          *[f"runs/{arm}/{name}" for arm in ("baseline", "improved")
            for name in ("start.json", "training_summary.json", "epochs.json", "sampled_exposures.json", "candidates.json")],
          *[f"scripts/{name}" for name in ("prepare_r04.py", "train_r04.py", "ignore_adapter.py", "test_ignore_adapter.py",
                                           "evaluate_r04.py", "test_evaluate_r04.py", "run_r04_evaluation.py",
                                           "audit_r04_results.py", "evaluate_resolution_stress.py")],
          "reports/data_independent_audit.json", "reports/ignore_adapter_tests.json"]


def main():
    inp = Inputs()
    res = {"archive_head": git("rev-parse", "HEAD"), "archive_status_porcelain": git("status", "--porcelain")}
    res["archive_head_is_review_commit"] = res["archive_head"] == REVIEW_COMMIT
    res["archive_clean"] = res["archive_status_porcelain"] == ""

    identity = []
    for rel in PAIRED:
        gh, local = inp.add(RECORD / rel), inp.add(R04 / rel)
        identity.append({"file": rel, "github_sha256": sha256(gh), "local_sha256": sha256(local), "identical": sha256(gh) == sha256(local)})
    res["github_local_identity"] = identity
    res["github_local_all_identical"] = all(row["identical"] for row in identity)

    protocol = inp.json(RECORD / "protocol.json")
    bindings = {"initial_weights": (protocol["initial_weights"], protocol["initial_weights_sha256"]),
                "data_summary": (R04 / "data/summary.json", protocol["data_summary_sha256"]),
                "native_manifest": (R04 / "data/native_manifest.json", protocol["native_manifest_sha256"])}
    res["protocol_input_bindings"] = {name: {"expected": expected, "actual": sha256(inp.add(path)), "match": sha256(path) == expected}
                                      for name, (path, expected) in bindings.items()}

    snapshot = inp.json(RECORD / "snapshot.json")
    snap = {job["job"]: job["metrics_sha256"] for job in snapshot["completed_jobs"]}
    csv_rows = {row["candidate"]: row for row in csv.DictReader(open(inp.add(RECORD / "partial_validation.csv"), encoding="utf-8-sig"))}
    candidates = {arm: {c["epoch"]: c for c in inp.json(RECORD / f"runs/{arm}/candidates.json")} for arm in ("baseline", "improved")}
    evaluators, manifests, table, problems = set(), set(), [], []
    for path in sorted((RECORD / "validation").glob("*/metrics.json")):
        job = path.parent.name
        arm, epoch = job.split("__")[0], int(job.split("__")[1].split("_")[1])
        metric = inp.json(path)
        local = R04 / "reports/evaluation_suite/evaluations" / job / "metrics.json"
        b, o = metric["bbox"], metric["operating"]
        p = o["tp"] / (o["tp"] + o["fp"]) if o["tp"] + o["fp"] else 0.0
        r = o["tp"] / (o["tp"] + o["fn"])
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        checks = {"snapshot_sha": sha256(path) == snap.get(job), "local_identical": sha256(inp.add(local)) == sha256(path),
                  "csv_ap300": abs(float(csv_rows[job]["ap50_95_max300"]) - b["ap50_95_max300"]) < 1e-12,
                  "csv_ap100": abs(float(csv_rows[job]["ap50_95_max100"]) - b["ap50_95_max100"]) < 1e-12,
                  "checkpoint_bound": metric["checkpoint_sha256"] == candidates[arm][epoch]["sha256"],
                  "prf_recomputed": max(abs(p - o["precision"]), abs(r - o["recall"]), abs(f1 - o["f1"])) < 1e-12,
                  "role": metric["split"] == "val" and metric["evaluation_role"] == "validation_selection" and metric["d455_verified"] is False}
        evaluators.add(metric["evaluator_sha256"]); manifests.add(metric["manifest_sha256"])
        if not all(checks.values()):
            problems.append({"job": job, "checks": checks})
        table.append({"job": job, "epoch": epoch, "partial_epoch": candidates[arm][epoch]["partial_epoch"],
                      "optimizer_calls": candidates[arm][epoch]["optimizer_calls"], "pipeline": metric["pipeline"],
                      "ap50_95_max300": b["ap50_95_max300"], "ap50_95_max100": b["ap50_95_max100"],
                      "operating_confidence": o["confidence"], "tp": o["tp"], "fp": o["fp"], "fn": o["fn"],
                      "precision": o["precision"], "recall": o["recall"], "f1": o["f1"],
                      "latency_p50_ms": metric["latency"]["offline_predict_and_postprocess_ms_p50"]})
    res["validation_jobs"] = len(table)
    res["validation_evaluator_sha256"] = sorted(evaluators)
    res["validation_manifest_sha256"] = sorted(manifests)
    res["validation_problems"] = problems
    res["validation_table"] = table

    mismatched = []
    for arm in ("baseline", "improved"):
        for epoch, row in candidates[arm].items():
            actual = sha256(inp.add(row["path"]))
            if actual != row["sha256"]:
                mismatched.append({"arm": arm, "epoch": epoch})
    res["checkpoint_count"] = sum(len(v) for v in candidates.values())
    res["checkpoint_sha_mismatches"] = mismatched

    args = {arm: yaml.safe_load(inp.add(R04 / f"training_logs/{arm}/args.yaml").read_text(encoding="utf-8")) for arm in ("baseline", "improved")}
    res["args_yaml_differences"] = {k: [args["baseline"].get(k), args["improved"].get(k)]
                                    for k in sorted(set(args["baseline"]) | set(args["improved"]))
                                    if args["baseline"].get(k) != args["improved"].get(k)}
    write_output("r01_hash_bindings", {
        "identity": "SHA-256 of whole file bytes",
        "prf": "precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2PR/(P+R) from stored operating counts",
        "scope": "Completed validation jobs only (22/40); no holdout job exists yet"}, res, inp)


if __name__ == "__main__":
    main()
