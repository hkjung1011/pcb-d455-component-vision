"""Plan or explicitly execute sequential, validation-first R03 evaluation.

Default is a CPU-only dry run. GPU inference requires --execute. This program
does not train, download weights, modify labels, or extend an epoch budget.
Examples:
  python run_evaluation_suite.py --models parts_yolo11n parts_yolo11s
  python run_evaluation_suite.py --models parts_yolo11n parts_yolo11s --execute
  python run_evaluation_suite.py --self-test

Completed jobs with identical evidence are reused when --resume is supplied.
An interrupted nonempty evaluator directory is never erased automatically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("parts_yolo11n", "parts_yolo11s", "board_yolo11n", "board_yolo11n_seg")
POLICY = {
    "version": "r03-native-evaluation-suite-v1",
    "selection": ["validation bbox AP50-95 max100 descending", "validation bbox AP50-95 max300 descending", "validation offline latency p50 ascending", "candidate ID ascending"],
    "parts_candidates": ["best whole", "best tile1024", "last whole", "last tile1024"],
    "board_candidates": ["best whole640 only"],
    "confidence": "Each validation candidate: box IoU>=0.5 micro-F1, confidence .05:.95 by .05; highest confidence wins F1 ties. Selected candidate's threshold is frozen for test.",
    "test_policy": "All selected-model validation and all selection locks precede every test call. General test and Pi test never select checkpoints, thresholds, or pipelines.",
    "bootstrap": "1000 source-group bootstrap resamples of fixed-threshold recall for general test; no Pi one-group CI. This is not an AP CI.",
    "independence": "Pi3B is held out from this fine-tuning train/validation set. Its possible inclusion in any pretrained source is unknown. Public-source physical specimen independence is not established.",
    "d455_verified": False,
    "optional_pi_robustness": "Only with --pi-robustness: half-resolution downsample then original-size upsample, and Gaussian sigma1 separately. Same native GT and frozen selection; camera proxy simulation, not D455 evidence.",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def finite_number(value, label):
    if not isinstance(value,(float,int)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite {label}: {value}")
    return float(value)


def rank(candidate):
    metric = candidate["metrics"]
    ap100 = finite_number(metric["bbox"]["ap50_95_max100"],"validation AP100")
    ap300 = finite_number(metric["bbox"]["ap50_95_max300"],"validation AP300")
    latency = finite_number(metric["latency"]["offline_predict_and_postprocess_ms_p50"],"latency")
    if not 0 <= ap100 <= 1 or not 0 <= ap300 <= 1 or latency < 0:
        raise ValueError("Invalid validation metric range")
    return (-ap100,-ap300,latency,candidate["candidate_id"])


def select_candidate(candidates):
    if not candidates or any(c["metrics"]["split"] != "val" for c in candidates):
        raise ValueError("Checkpoint selection requires validation-only candidates")
    winner = sorted(candidates,key=rank)[0]
    confidence = finite_number(winner["metrics"]["operating"]["confidence"],"validation confidence")
    if not 0 <= confidence <= 1:
        raise ValueError("Invalid selected confidence")
    return winner


def model_spec(model_id, args):
    parts = model_id.startswith("parts_")
    seg = model_id == "board_yolo11n_seg"
    manifest = args.parts_manifest if parts else args.board_seg_manifest if seg else args.board_detect_manifest
    run = args.run_root / model_id
    candidates = [(checkpoint,tile) for checkpoint in ("best","last") for tile in (0,1024)] if parts else [("best",0)]
    result = {"model_id":model_id,"task":"segment" if seg else "detect","manifest":str(manifest.resolve()),
            "run_dir":str(run.resolve()),"imgsz":1024 if parts else 640,
            "candidates":[{"candidate_id":f"{checkpoint}_{'whole' if not tile else 'tile1024'}",
                           "checkpoint":checkpoint,"weights":str((run/"fit"/"weights"/(checkpoint+".pt")).resolve()),
                           "tile_size":tile} for checkpoint,tile in candidates],
            "tests":[{"name":"general_test","split":"test","cohort":"test" if parts else None,"bootstrap":1000}]
                    + ([{"name":"pi_test","split":"test","cohort":"pi_test","bootstrap":0}] if parts else [])}
    if parts and args.pi_robustness:
        for transform in ("half_resolution", "gaussian_sigma1"):
            result["tests"].append({"name":"pi_test_"+transform,"split":"test","cohort":"pi_test","bootstrap":0,
                                    "simulation":transform,"manifest":str((args.simulation_root/transform/"native_manifest.json").resolve())})
    return result


def plan(args):
    return {"policy":POLICY,"evaluator":str((ROOT/"scripts"/"native_evaluate.py").resolve()),
            "models":[model_spec(model_id,args) for model_id in args.models]}


def preflight(value):
    evidence = {"evaluator_sha256":sha(value["evaluator"]),"models":{}}
    for spec in value["models"]:
        manifest = read(spec["manifest"])
        rows = manifest["records"]
        assert_evaluation_labels(rows)
        partitions = [("val",None)] + [(t["split"],t["cohort"]) for t in spec["tests"]]
        for split,cohort in partitions:
            subset=[r for r in rows if r["split"]==split and (cohort is None or r.get("cohort")==cohort)]
            if not subset or not any(r["objects"] for r in subset):
                raise ValueError(f"Missing positive cohort {spec['model_id']} {split}/{cohort}")
        seen = {}
        for row in rows:
            group=row["group_id"]
            if group in seen and seen[group] != row["split"]:
                raise ValueError(f"Manifest origin group crosses splits: {group}")
            seen[group]=row["split"]
        summary_path=Path(spec["run_dir"])/"run_summary.json"
        summary=read(summary_path)
        if summary.get("actual_optimizer_steps",0)<1 or summary.get("weights_changed") is not True:
            raise ValueError(f"Training completion with actual updates not established: {summary_path}")
        evidence["models"][spec["model_id"]]={"manifest_sha256":sha(spec["manifest"]),
            "run_summary_sha256":sha(summary_path),"names":manifest["names"],
            "scope":manifest.get("scope"),"annotation_limitations":manifest.get("annotation_limitations"),
            "checkpoints":{c["candidate_id"]:sha(c["weights"]) for c in spec["candidates"]}}
    return evidence


def assert_evaluation_labels(rows):
    """Keep Commons drafts and inferred/model-produced labels outside evaluation GT."""
    for row in rows:
        identity=" ".join(str(row.get(k,"")) for k in ("source","source_url","image","source_image")).lower()
        if "commons" in identity:
            raise ValueError("Commons images/draft labels are outside this predeclared benchmark; cannot serve as evaluation GT")
        for item in [row]+row.get("objects",[]):
            status=" ".join(str(item.get(k,"")) for k in ("annotation_status","annotation_type","label_status","mask_origin")).lower()
            if any(token in status for token in ("draft","pseudo","prediction","ai_generated","sam_generated","unreviewed")):
                raise ValueError(f"Unapproved/inferred evaluation annotation: {row.get('id')}: {status}")


def prepare_pi_simulations(source_manifest, destination):
    """CPU-only deterministic degradations; label geometry and canvas size unchanged."""
    import copy
    import cv2
    import numpy as np
    source_manifest=Path(source_manifest)
    original=read(source_manifest)
    rows=[r for r in original["records"] if r["split"]=="test" and r.get("cohort")=="pi_test"]
    assert_evaluation_labels(rows)
    if len(rows)!=2 or len({r["group_id"] for r in rows})!=1:
        raise ValueError("Pi simulation protocol requires the original two-face, one-group holdout")
    evidence={}
    for transform in ("half_resolution","gaussian_sigma1"):
        folder=Path(destination)/transform; folder.mkdir(parents=True,exist_ok=True)
        modified=[]
        for row in rows:
            source=Path(row.get("source_image",row["image"]))
            if not source.is_absolute():source=source_manifest.parent/source
            im=cv2.imdecode(np.fromfile(source,dtype=np.uint8),cv2.IMREAD_COLOR)
            if im is None:raise ValueError(f"Cannot decode simulation source {source}")
            height,width=im.shape[:2]
            if (width,height)!=(row["width"],row["height"]):raise ValueError("Simulation source dimensions changed")
            if transform=="half_resolution":
                down=cv2.resize(im,(max(1,width//2),max(1,height//2)),interpolation=cv2.INTER_AREA)
                derived=cv2.resize(down,(width,height),interpolation=cv2.INTER_LINEAR)
            else:
                derived=cv2.GaussianBlur(im,(0,0),sigmaX=1.0,sigmaY=1.0,borderType=cv2.BORDER_REFLECT_101)
            ok,encoded=cv2.imencode(".png",derived,[cv2.IMWRITE_PNG_COMPRESSION,6])
            if not ok:raise ValueError("PNG encoding failed")
            path=folder/(row["id"]+"__"+transform+".png")
            contents=encoded.tobytes()
            if path.exists():
                if sha(path)!=hashlib.sha256(contents).hexdigest():raise ValueError(f"Existing simulation bytes differ: {path}")
            else:path.write_bytes(contents)
            item=copy.deepcopy(row)
            item.update(id=row["id"]+"__"+transform,image=str(path.resolve()),source_image=str(path.resolve()),sha256=sha(path),
                        simulation_original_id=row["id"],simulation_original_image=str(source.resolve()),
                        simulation_original_sha256=sha(source),simulation=transform)
            if item["objects"]!=row["objects"]:raise AssertionError("Simulation must never alter native GT")
            modified.append(item)
        simulation={"kind":transform,"camera_proxy_not_D455":True,"canvas_size_preserved":True,"native_gt_unchanged":True,
                    "source_manifest_sha256":sha(source_manifest),"opencv_version":cv2.__version__,
                    "description":"INTER_AREA to floor(W/2),floor(H/2), then INTER_LINEAR back to original W,H" if transform=="half_resolution" else "GaussianBlur ksize auto, sigmaX=sigmaY=1.0 native pixels, BORDER_REFLECT_101",
                    "selection_usage":"None: predeclared stress test only; frozen validation confidence/checkpoint/pipeline"}
        manifest={"schema":"r03-native-simulation-v1","names":original["names"],"training_eligible":False,
                  "scope":"WACV Pi3B provided target4 boxes under deterministic image degradation; camera proxy simulation, not D455",
                  "annotation_limitations":original.get("annotation_limitations"),"simulation":simulation,"records":modified}
        target=folder/"native_manifest.json"
        if target.exists():
            if read(target)!=manifest:raise ValueError(f"Simulation manifest changed: {target}")
        else:save(target,manifest)
        evidence[transform]={"manifest":str(target.resolve()),"manifest_sha256":sha(target),"simulation":simulation,
                             "image_sha256":{r["id"]:r["sha256"] for r in modified}}
    return evidence


def command(python, evaluator, manifest, candidate, output, split, imgsz, *, cohort=None, confidence=None, bootstrap=0):
    result=[python,evaluator,"--manifest",manifest,"--weights",candidate["weights"],"--output",str(output),
            "--split",split,"--imgsz",str(imgsz),"--tile-size",str(candidate["tile_size"]),"--bootstrap",str(bootstrap)]
    if cohort is not None:result.extend(["--cohort",cohort])
    if confidence is not None:result.extend(["--operating-conf",format(confidence,".17g")])
    return result


def evaluate_job(args, value, evidence, spec, candidate, split, name, *, cohort=None, confidence=None, bootstrap=0, manifest_override=None, simulation=None):
    job_id=f"{spec['model_id']}__{name}"
    output=args.output_root/"evaluations"/job_id
    control=args.output_root/"jobs"/(job_id+".json")
    log=args.output_root/"logs"/(job_id+".log")
    manifest=manifest_override or spec["manifest"]
    manifest_hash=evidence["simulations"][simulation]["manifest_sha256"] if simulation else evidence["models"][spec["model_id"]]["manifest_sha256"]
    simulation_metadata=evidence["simulations"][simulation]["simulation"] if simulation else None
    binding={"model_id":spec["model_id"],"candidate_id":candidate["candidate_id"],"split":split,"cohort":cohort,"task":spec["task"],"simulation":simulation_metadata,
             "checkpoint_sha256":evidence["models"][spec["model_id"]]["checkpoints"][candidate["candidate_id"]],
             "manifest_sha256":manifest_hash,
             "evaluator_sha256":evidence["evaluator_sha256"],"tile_size":candidate["tile_size"],
             "imgsz":spec["imgsz"],"confidence":confidence,"bootstrap":bootstrap}
    binding_hash=fingerprint(binding); metric_file=output/"metrics.json"
    if control.exists():
        prior=read(control)
        if not args.resume or prior["binding_sha256"]!=binding_hash:
            raise ValueError(f"Existing job or changed evidence: {control}. Use --resume only for identical inputs.")
        if prior["status"]=="completed":
            if not metric_file.is_file() or sha(metric_file)!=prior["metrics_sha256"]:
                raise ValueError(f"Completed metrics evidence changed: {metric_file}")
            print(f"REUSE {job_id}",flush=True)
            return {"candidate_id":candidate["candidate_id"],"candidate":candidate,"metrics":read(metric_file),"metrics_path":str(metric_file.resolve()),"metrics_sha256":prior["metrics_sha256"],"simulation":simulation_metadata}
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Incomplete evaluator output retained for review: {output}. No automatic deletion/re-execution.")
    # Detect a weights/manifest mutation even if training restarted after preflight.
    if sha(candidate["weights"])!=binding["checkpoint_sha256"] or sha(manifest)!=binding["manifest_sha256"] or sha(value["evaluator"])!=binding["evaluator_sha256"]:
        raise ValueError("Input changed since preflight; stop before inference")
    cmd=command(args.python,value["evaluator"],manifest,candidate,output,split,spec["imgsz"],cohort=cohort,confidence=confidence,bootstrap=bootstrap)
    record={"status":"running","started_utc":now(),"binding":binding,"binding_sha256":binding_hash,"command":cmd}
    save(control,record); log.parent.mkdir(parents=True,exist_ok=True)
    print(f"RUN {job_id}",flush=True)
    with log.open("w",encoding="utf-8") as handle:
        completed=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,check=False,cwd=ROOT)
    if completed.returncode!=0:
        record.update(status="failed",finished_utc=now(),return_code=completed.returncode);save(control,record)
        raise RuntimeError(f"Evaluator failed: {job_id}; see {log}")
    metric=read(metric_file)
    for key in ("checkpoint_sha256","manifest_sha256","split","cohort","task","imgsz","tile_size"):
        if metric.get(key)!=binding[key]:raise ValueError(f"Evaluator evidence mismatch for {key}: {job_id}")
    if confidence is not None and abs(metric["operating"]["confidence"]-confidence)>1e-12:
        raise ValueError("Test operating threshold differs from frozen validation selection")
    record.update(status="completed",finished_utc=now(),return_code=0,metrics_sha256=sha(metric_file));save(control,record)
    return {"candidate_id":candidate["candidate_id"],"candidate":candidate,"metrics":metric,"metrics_path":str(metric_file.resolve()),"metrics_sha256":record["metrics_sha256"],"simulation":simulation_metadata}


def compact(result):
    m=result["metrics"]
    return {"candidate_id":result["candidate_id"],"metrics_path":result["metrics_path"],"metrics_sha256":result["metrics_sha256"],"simulation":result.get("simulation"),
            **{k:m.get(k) for k in ("split","cohort","task","class_names","images","groups","native_gt_instances","checkpoint_sha256","manifest_sha256","bbox","mask","latency","group_bootstrap_recall95","annotation_limitations")},
            "operating":{k:v for k,v in m["operating"].items() if k!="per_image"}}


def execute(args,value):
    evidence=preflight(value)
    if args.pi_robustness and any(m.startswith("parts_") for m in args.models):
        evidence["simulations"]=prepare_pi_simulations(args.parts_manifest,args.simulation_root)
    protocol={"plan":value,"evidence":evidence}
    protocol_hash=fingerprint(protocol)
    args.output_root.mkdir(parents=True,exist_ok=True)
    protocol_file=args.output_root/"protocol.json"
    if protocol_file.exists():
        if not args.resume or read(protocol_file)["protocol_sha256"]!=protocol_hash:
            raise ValueError("Output belongs to a different/already started protocol; use a new root or --resume unchanged protocol")
    elif any(args.output_root.iterdir()):
        raise ValueError("Nonempty output directory without protocol evidence")
    else:save(protocol_file,{"created_utc":now(),"protocol_sha256":protocol_hash,**protocol})
    validation={}; selections={}
    # PHASE 1: Every predeclared validation comparison precedes every test.
    for spec in value["models"]:
        candidates=[]
        for candidate in spec["candidates"]:
            candidates.append(evaluate_job(args,value,evidence,spec,candidate,"val","val__"+candidate["candidate_id"]))
        validation[spec["model_id"]]=candidates
        winner=select_candidate(candidates)
        selections[spec["model_id"]]={"candidate":winner["candidate"],"checkpoint_sha256":winner["metrics"]["checkpoint_sha256"],
            "manifest_sha256":winner["metrics"]["manifest_sha256"],"imgsz":spec["imgsz"],"operating_confidence":winner["metrics"]["operating"]["confidence"],
            "validation_metrics_path":winner["metrics_path"],"validation_metrics_sha256":winner["metrics_sha256"],"validation_bbox":winner["metrics"]["bbox"],
            "validation_mask":winner["metrics"].get("mask"),"rule":POLICY["selection"]}
    # PHASE 2: Persist one immutable selection covering all chosen models BEFORE test.
    selection_file=args.output_root/"selection.json"
    selection_payload={"protocol_sha256":protocol_hash,"selections":selections,"test_results_used_for_selection":False}
    if selection_file.exists():
        locked=read(selection_file)
        if fingerprint({k:v for k,v in locked.items() if k!="frozen_utc"})!=fingerprint(selection_payload):
            raise ValueError("Previously frozen selection differs; test selection may not be rewritten")
    else:save(selection_file,{"frozen_utc":now(),**selection_payload})
    selection_hash=sha(selection_file)
    aggregate={"status":"SELECTION_FROZEN_TEST_PENDING","protocol_sha256":protocol_hash,"selection_sha256":selection_hash,
               "policy":POLICY,"models":{mid:{"selection":selections[mid],"validation_candidates":[compact(r) for r in rows],"tests":{}} for mid,rows in validation.items()}}
    save(args.output_root/"aggregate.json",aggregate)
    # PHASE 3: Inference is strictly sequential; no GPU work is scheduled in parallel.
    for spec in value["models"]:
        choice=selections[spec["model_id"]]
        for test in spec["tests"]:
            if sha(selection_file)!=selection_hash:raise ValueError("Frozen selection file changed")
            result=evaluate_job(args,value,evidence,spec,choice["candidate"],test["split"],test["name"],
                                cohort=test["cohort"],confidence=choice["operating_confidence"],bootstrap=test["bootstrap"],
                                manifest_override=test.get("manifest"),simulation=test.get("simulation"))
            aggregate["models"][spec["model_id"]]["tests"][test["name"]]=compact(result)
            save(args.output_root/"aggregate.json",aggregate)
    aggregate.update(status="COMPLETED_OFFLINE_NOT_D455_VALIDATED",completed_utc=now())
    save(args.output_root/"aggregate.json",aggregate)
    print(json.dumps({"status":aggregate["status"],"aggregate":str((args.output_root/"aggregate.json").resolve())}),flush=True)


def self_test():
    def candidate(identifier,ap100,ap300,latency,split="val"):
        return {"candidate_id":identifier,"metrics":{"split":split,"bbox":{"ap50_95_max100":ap100,"ap50_95_max300":ap300},"latency":{"offline_predict_and_postprocess_ms_p50":latency},"operating":{"confidence":.35}}}
    assert select_candidate([candidate("a",.6,.6,1),candidate("b",.61,.61,99)])["candidate_id"]=="b"
    assert select_candidate([candidate("a",.6,.61,1),candidate("b",.6,.62,99)])["candidate_id"]=="b"
    assert select_candidate([candidate("a",.6,.6,99),candidate("b",.6,.6,1)])["candidate_id"]=="b"
    assert select_candidate([candidate("b",.6,.6,1),candidate("a",.6,.6,1)])["candidate_id"]=="a"
    for bad in [candidate("bad",.6,.6,1,"test"),candidate("bad",None,.6,1)]:
        try:select_candidate([bad])
        except ValueError:pass
        else:raise AssertionError("Invalid selection candidate accepted")
    cmd=command("python","eval.py","m.json",{"weights":"best.pt","tile_size":1024},Path("out"),"test",1024,cohort="pi_test",confidence=.35,bootstrap=0)
    assert cmd[cmd.index("--cohort")+1]=="pi_test" and cmd[cmd.index("--bootstrap")+1]=="0"
    assert float(cmd[cmd.index("--operating-conf")+1])==.35
    for bad in [{"source":"commons_draft","objects":[]},{"source":"test","objects":[{"annotation_status":"ai_generated"}]}]:
        try:assert_evaluation_labels([bad])
        except ValueError:pass
        else:raise AssertionError("Draft/predicted GT accepted")
    print("CPU self-tests passed: 9 checks; no inference, model loading, torch import, or output writes.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true",help="Explicitly launch sequential native_evaluate GPU subprocesses")
    parser.add_argument("--resume",action="store_true",help="Reuse completed jobs only if all evidence matches")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--pi-robustness",action="store_true",help="Predeclare two deterministic Pi test degradations; no selection or D455 claim")
    parser.add_argument("--models",nargs="+",choices=MODELS,default=list(MODELS))
    parser.add_argument("--python",default=sys.executable)
    parser.add_argument("--run-root",type=Path,default=ROOT/"runs")
    parser.add_argument("--output-root",type=Path,default=ROOT/"reports"/"evaluation_suite")
    parser.add_argument("--parts-manifest",type=Path,default=ROOT/"component_assets"/"native_manifest.json")
    parser.add_argument("--simulation-root",type=Path,default=ROOT/"component_assets"/"pi_simulations_v1")
    parser.add_argument("--board-detect-manifest",type=Path,default=ROOT/"data"/"board_detect_v2"/"manifest.json")
    parser.add_argument("--board-seg-manifest",type=Path,default=ROOT/"data"/"board_segment_v2"/"manifest.json")
    args=parser.parse_args()
    if args.self_test:
        if args.execute:parser.error("--self-test and --execute are mutually exclusive")
        self_test();return
    if len(set(args.models))!=len(args.models):parser.error("Duplicate model IDs are not allowed")
    args.output_root=args.output_root.resolve()
    value=plan(args)
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN_NO_GPU","plan":value,"next_action":"Run with --execute only after all selected training runs finish; do not overlap this suite with training."},indent=2,ensure_ascii=False))
        return
    execute(args,value)


if __name__=="__main__":
    main()
