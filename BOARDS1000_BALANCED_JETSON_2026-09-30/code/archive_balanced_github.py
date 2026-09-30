"""Prepare a complete, portable balanced-data Git/Release archive locally.

No commit/push/GitHub calls, model loading, inference, training or test evaluation.
Requires a completed reviewed output package and a clean existing main checkout.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
from urllib.parse import quote, unquote, urlsplit
import zipfile


HERE = Path(__file__).resolve().parent.parent
REPO = Path(r"C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\github_private_archive")
TRIAL = HERE / "work/boards1000_balanced_jetson_20260930"
SOURCE = HERE / "outputs/BOARDS1000_BALANCED_JETSON_2026-09-30"
PARENT_DATA = HERE / "work/boards1000_s650_v150_t200_e50_20260929/datasets/boards_v1"
# Keep Windows archive member paths below MAX_PATH; preserve the failed long stage.
STAGE = Path(r"C:\Users\hkjun\Documents\Codex\_b30_archive_20260930")
NAME = "BOARDS1000_BALANCED_JETSON_2026-09-30"
TAG = "boards1000-balanced-jetson-2026-09-30"
URL = "https://github.com/hkjung1011/pcb-d455-component-vision"
RELEASE = f"{URL}/releases/tag/{TAG}"
ZIP_NAME = "BOARDS1000_BALANCED_JETSON_FULL_SNAPSHOT.zip"
FINGERPRINT = "3120af5994f7494f608b31b59b48833eebee8423408f5a0190f516f6e708ba6f"
ORIGINAL_NAME = "BOARDS1000_2026-09-30"
FINE_NAME = "BOARDS1000_FINETUNE_2026-09-30"
AP = "metrics/mAP50-95(B)"
AP50 = "metrics/mAP50(B)"
ROOT_FILES = ("README.md", "CURRENT_SUMMARY.md", "CHANGELOG.md", "CURRENT_STATUS.json")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def jwrite(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def files(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def copy_file(source, destination):
    require(source.is_file() and not source.is_symlink(), f"Missing or unsafe input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def row(path, relative):
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)}


def manifest(root, name, scope):
    value = {"scope": scope, "files": [row(p, p.relative_to(root).as_posix())
            for p in files(root) if p.relative_to(root).as_posix() != name]}
    jwrite(root / name, value)
    return value


def fingerprint(root):
    h = hashlib.sha256()
    for path in files(root):
        if path.suffix != ".cache":
            h.update(path.relative_to(root).as_posix().encode() + b"\0" + sha(path).encode() + b"\n")
    return h.hexdigest()


def verify_package():
    m = read(SOURCE / "ARTIFACT_MANIFEST.json")
    expected = set()
    for entry in m["files"]:
        relative = PurePosixPath(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe package path")
        path = SOURCE.joinpath(*relative.parts)
        require(path.is_file() and not path.is_symlink(), f"Missing package file: {relative}")
        require(sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"], f"Package hash differs: {relative}")
        require(relative.as_posix() not in expected, "Duplicate package manifest entry")
        expected.add(relative.as_posix())
    actual = {p.relative_to(SOURCE).as_posix() for p in files(SOURCE)
              if p.relative_to(SOURCE).as_posix() != "ARTIFACT_MANIFEST.json"}
    require(expected == actual, "Package manifest must cover exact package file set")


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in files(root)
            if ".git" not in p.relative_to(root).parts}


def rewrite_links(text, asset_names):
    def replacement(match):
        target = match.group(2).strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or not parsed.path.lower().endswith(".pt"):
            return match.group(0)
        basename = Path(unquote(parsed.path).replace("\\", "/")).name
        target = f"{URL}/releases/download/{TAG}/{quote(basename)}" if basename in asset_names else RELEASE
        return f"{match.group(1)}({target})"
    return re.sub(r"(!?\[[^\]]*\])\(([^)]+)\)", replacement, text).replace(
        "(ARTIFACT_MANIFEST.json)", "(evidence/original_local_artifact_manifest.json)")


def prepend(text, section):
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("# "):
        return lines[0].rstrip("\r\n") + "\n\n" + section.rstrip() + "\n\n" + "".join(lines[1:]).lstrip("\r\n")
    return section.rstrip() + "\n\n" + text


VERIFY_SCRIPT = r'''"""Verify the restored portable snapshot with stdlib only; no inference."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, hashlib, json, math

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def verify(root):
    root=root.resolve();m=json.loads((root/'SNAPSHOT_MANIFEST.json').read_text(encoding='utf-8'))
    listed=set()
    for r in m['files']:
        p=(root/r['path']).resolve();assert p.is_relative_to(root) and p.is_file() and not p.is_symlink(),r['path']
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path'];listed.add(r['path'])
    assert listed=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SNAPSHOT_MANIFEST.json' and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo','.cache'}}
    dataset=root/'datasets/boards_v1';index=json.loads((dataset/'dataset_index.json').read_text(encoding='utf-8'))
    provenance=json.loads((root/'PORTABLE_DATASET.json').read_text(encoding='utf-8'))
    counts=Counter();groups=defaultdict(set);hashes=defaultdict(set);expected=set();boxes={s:Counter() for s in ['train','val','test']}
    for r in index['records']:
        s=r['split'];counts[s]+=1;groups[r['group']].add(s);hashes[r['image_sha256']].add(s)
        for field in ['image','label']:
            p=(dataset/r[field]).resolve();assert p.is_relative_to(dataset.resolve()) and sha(p)==r[field+'_sha256'];expected.add(r[field])
        for line in (dataset/r['label']).read_text().splitlines():
            ts=line.split();assert len(ts)==5;cid=int(ts[0]);x,y,w,h=map(float,ts[1:])
            assert 0<=cid<len(index['classes']) and all(math.isfinite(v) for v in [x,y,w,h])
            assert 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
            assert x-w/2>=-1e-5 and y-h/2>=-1e-5 and x+w/2<=1+1e-5 and y+h/2<=1+1e-5
            boxes[s][index['classes'][cid]]+=1
    assert dict(counts)=={'train':650,'val':150,'test':200}
    assert all(len(s)==1 for s in groups.values()) and all(len(s)==1 for s in hashes.values())
    actual={p.relative_to(dataset).as_posix() for d in ['images','labels'] for p in (dataset/d).rglob('*') if p.is_file() and p.suffix!='.cache'}
    assert actual==expected
    verification=json.loads((root/'evidence/dataset_verification.json').read_text(encoding='utf-8'))['boards_v1']
    assert {s:dict(c) for s,c in boxes.items()}==verification['boxes']
    h=hashlib.sha256()
    for p in sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix!='.cache'):
        h.update(p.relative_to(dataset).as_posix().encode()+b'\0'+sha(p).encode()+b'\n')
    assert h.hexdigest()==provenance['portable_dataset_fingerprint']
    plan=json.loads((root/'evidence/experiment_plan.json').read_text(encoding='utf-8'))
    for r in plan['evaluation_inventory']['records']:
        assert sha(dataset/r['image'])==r['image_sha256'] and sha(dataset/r['label'])==r['label_sha256']
    assert len(plan['evaluation_inventory']['records'])==350
    return {'status':'PASS','files_hashed':len(m['files']),'images':dict(counts),'portable_dataset_fingerprint':h.hexdigest(),'original_training_dataset_fingerprint':provenance['original_training_dataset_fingerprint'],'inference_executed':False,'scope':'Frozen bytes, labels, groups and evaluation hashes; original pHash/Commons findings preserved, no physical-scene independence claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print(json.dumps(verify(a.root),indent=2))
'''

REPRODUCE_SCRIPT = r'''"""Portable NEW training launcher; default action only checks snapshot bytes."""
from pathlib import Path
import argparse, json, os, re, tempfile
from verify_snapshot import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--action',choices=['check','train'],default='check');p.add_argument('--output',type=Path);p.add_argument('--name',default='balanced_reproduction_01');a=p.parse_args()
    root=a.root.resolve();print(json.dumps(verify(root),indent=2))
    if a.action=='check':return
    assert a.output is not None,'Specify an external --output folder for a NEW training run'
    out=a.output.resolve();assert not out.is_relative_to(root),'Preserve the frozen snapshot'
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',a.name),'Use a simple run name'
    assert not (out/a.name).exists(),'Existing run preserved; choose a new name'
    cfg=json.loads((root/'configs/train.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='balanced_yolo_settings_') as settings:
        os.environ['YOLO_CONFIG_DIR']=settings;os.environ['YOLO_OFFLINE']='true';os.environ['WANDB_MODE']='disabled';os.environ['COMET_MODE']='DISABLED'
        import torch,yaml
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update({k:False for k in ['sync','clearml','comet','dvc','mlflow','neptune','raytune','tensorboard','wandb'] if k in SETTINGS})
        assert torch.cuda.is_available(),'Archived training configuration requires CUDA';torch.set_num_threads(4)
        classes=json.loads((root/'configs/classes.json').read_text(encoding='utf-8'))
        data={'path':(root/'datasets/boards_v1').as_posix(),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(classes))}
        out.mkdir(parents=True,exist_ok=True);yaml_path=out/(a.name+'_data.yaml')
        with yaml_path.open('x',encoding='utf-8') as f:yaml.safe_dump(data,f,sort_keys=False)
        model=YOLO(str(root/'weights/original-baseline-best.pt'))
        model.train(data=str(yaml_path),project=str(out),name=a.name,exist_ok=False,**cfg)
        print('NEW training/validation finished. Test inference was not invoked. Archived selection/test evidence was not overwritten.')

if __name__=='__main__':main()
'''


def main():
    require(SOURCE.is_dir(), "Completed balanced result package is missing")
    require(not STAGE.exists() and not (REPO / NAME).exists(), "Existing archive preserved; inspect instead of rerunning")
    branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
    require(branch == "main" and not dirty, "Existing archive checkout must be clean main")
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    verify_package()
    state = read(TRIAL / "execution_state.json")
    plan = read(TRIAL / "experiment_plan.json")
    selection = read(TRIAL / "selection.json")
    summary = read(TRIAL / "training-summary.json")
    baseline = read(TRIAL / "baseline_validation.json")
    previous = read(TRIAL / "current_selected_validation.json")
    candidate = read(TRIAL / "candidate_validation.json")
    outcome_audit = read(TRIAL / "outcome_audit.json")
    verification = read(TRIAL / "datasets/verification.json")
    verified = verification["boards_v1"]
    dataset = TRIAL / "datasets/boards_v1"
    index = read(dataset / "dataset_index.json")
    prior_status = read(REPO / "CURRENT_STATUS.json")
    require(state["status"] == "complete", "Balanced phase is not complete")
    require(plan["dataset_fingerprint"] == state["dataset_fingerprint"] == verified["dataset_fingerprint"] == FINGERPRINT == fingerprint(dataset), "Training dataset fingerprint differs")
    require(verified["images"] == {"train": 650, "val": 150, "test": 200} and verified["training_ready"] and verified["structure_pass"], "Balanced dataset did not pass preparation checks")
    require(plan["validation_protocol"] == baseline["protocol"] == previous["protocol"] == candidate["protocol"], "Selection validation protocols differ")
    require(plan["initial_checkpoint_sha256"] == baseline["checkpoint_sha256"] == selection["baseline_checkpoint_sha256"], "Original initializer/baseline hashes differ")
    require(plan["current_selected_checkpoint_sha256"] == previous["checkpoint_sha256"] == selection["current_selected_checkpoint_sha256"] == prior_status["best_sha256"], "Prior selected model differs from frozen fallback")
    require(summary["best_sha256"] == candidate["checkpoint_sha256"] == selection["candidate_checkpoint_sha256"], "Candidate checkpoint hashes differ")
    require(selection["selection_split"] == "val" and selection["policy"] == plan["promotion_policy"], "Selection policy differs from frozen plan")
    policy = selection["policy"]
    gain = candidate["overall"][AP] - baseline["overall"][AP]
    current_gain = candidate["overall"][AP] - previous["overall"][AP]
    old_classes = {r["class_name"]: r["map50_95"] for r in baseline["per_class"]}
    new_classes = {r["class_name"]: r["map50_95"] for r in candidate["per_class"]}
    require(old_classes.keys() == new_classes.keys(), "Validation class sets differ")
    deltas = {n: new_classes[n] - value for n, value in old_classes.items()}
    computed = (gain >= policy["minimum_overall_ap50_95_gain"] and
                min(deltas.values()) >= -policy["maximum_per_class_ap50_95_drop"] and
                deltas["jetson"] >= policy["minimum_jetson_ap50_95_gain"] and
                current_gain >= policy["minimum_overall_gain_vs_current_selected"])
    promoted = selection["promoted"]
    require(isinstance(promoted, bool) and promoted == computed == state["promoted"], "Recorded promotion differs from validation-only gate")
    require(abs(gain-selection["overall_ap_gain"]) < 1e-10 and abs(current_gain-selection["overall_ap_gain_vs_current_selected"]) < 1e-10, "Recorded global validation gains differ")
    require(all(abs(deltas[n]-selection["per_class_ap_gains"][n]) < 1e-10 for n in deltas), "Recorded per-class validation gains differ")
    selected_sha = candidate["checkpoint_sha256"] if promoted else previous["checkpoint_sha256"]
    require(selected_sha == selection["selected_checkpoint_sha256"] == state["selected_checkpoint_sha256"], "Selected checkpoint must follow frozen validation gate")
    require(outcome_audit["status"] == "CONSISTENT" and outcome_audit["promoted"] == promoted and
            outcome_audit["selected_checkpoint_sha256"] == selected_sha and
            outcome_audit["candidate_checkpoint_sha256"] == candidate["checkpoint_sha256"], "Independent outcome audit differs from frozen selection")
    test = None
    if promoted:
        test = read(TRIAL / "test-evaluation.json")
        require(test["invocations"] == state["test_invocations"] == 1 and test["checkpoint_sha256"] == selected_sha and test["images"] == 200, "Candidate test evidence differs")
    else:
        require(state["test_invocations"] == 0 and not (TRIAL / "test-evaluation.json").exists(), "Rejected candidate must not be tested")
    csv_rows = list(csv.DictReader((TRIAL / "runs/boards1000_balanced_jetson30/results.csv").open(encoding="utf-8-sig")))
    require(len(csv_rows) == summary["epochs_completed"] == len(state["epochs"]) and len(csv_rows) <= summary["config"]["epochs"], "Actual epoch counts differ")
    require(summary["actual_optimizer_steps"] == state["optimizer_steps"], "Optimizer step counts differ")
    for filename in ("execution_state.json", "experiment_plan.json", "selection.json", "training-summary.json", "baseline_validation.json", "current_selected_validation.json", "candidate_validation.json", "outcome_audit.json"):
        require(sha(SOURCE / "evidence" / filename) == sha(TRIAL / filename), f"Packaged evidence differs: {filename}")
    original_eval = {r["id"]: r for r in read(PARENT_DATA / "dataset_index.json")["records"] if r["split"] in {"val", "test"}}
    evaluation = [r for r in index["records"] if r["split"] in {"val", "test"}]
    require(len(evaluation) == len(original_eval) == 350, "Expected 350 frozen evaluation photos")
    for record in evaluation:
        parent = original_eval[record["id"]]
        require(record == parent, f"Evaluation index record changed: {record['id']}")
        for field in ("image", "label"):
            require(sha(dataset / record[field]) == sha(PARENT_DATA / parent[field]) == record[field+"_sha256"], f"Evaluation bytes changed: {record['id']}")
    correction = read(TRIAL / "data_correction_verification.json")
    require(correction["jetson_train_subtype_image_counts"] == {"TX2": 27, "Nano": 40}, "Jetson subtype correction differs")
    require(correction["evaluation_records_and_image_label_bytes_unchanged"] == 350, "Data correction evidence differs")
    before = inventory(REPO)
    root_original = {n: (REPO / n).read_text(encoding="utf-8") for n in ROOT_FILES}
    STAGE.mkdir()
    git_root = STAGE / NAME
    zip_root = STAGE / "zip_content" / NAME
    portable = zip_root / "portable"
    assets = STAGE / "release_assets"
    assets.mkdir()
    # Preserve original output bytes, original manifest and old absolute paths.
    shutil.copytree(SOURCE, zip_root / "local_package" / SOURCE.name)
    for path in files(dataset):
        if path.suffix != ".cache":
            copy_file(path, portable / "datasets/boards_v1" / path.relative_to(dataset))
    copy_file(dataset / "data.yaml", portable / "evidence/original_training_data.yaml")
    # Omitting path makes Ultralytics resolve image paths from the YAML folder.
    portable_yaml = "train: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {cid}: {name}\n" for cid, name in enumerate(index["classes"]))
    write(portable / "datasets/boards_v1/data.yaml", portable_yaml)
    copy_file(TRIAL / "datasets/verification.json", portable / "evidence/dataset_verification.json")
    required_evidence = ("source_annotations.json", "derivation.json", "candidate_exclusions.json", "visual_review.json", "selection_proposal.json", "data_correction_verification.json", "experiment_plan.json", "execution_state.json", "selection.json", "training-summary.json", "baseline_validation.json", "current_selected_validation.json", "candidate_validation.json", "outcome_audit.json")
    for filename in required_evidence:
        copy_file(TRIAL / filename, portable / "evidence" / filename)
    for path in files(TRIAL):
        relative = path.relative_to(TRIAL)
        if (relative.parts[0].startswith("review_") or (len(relative.parts) == 1 and path.suffix.lower() in {".jpg", ".png"})):
            copy_file(path, portable / "data_review" / relative)
    for path in files(SOURCE):
        relative = path.relative_to(SOURCE)
        if path.suffix.lower() == ".pt":
            if path.name in {"selected-best.pt", "candidate-best.pt"}:
                copy_file(path, assets / path.name)
            copy_file(path, portable / "weights" / path.name)
        elif relative.as_posix() not in {"README.md", "ARTIFACT_MANIFEST.json"}:
            copy_file(path, portable / "original_result_files" / relative)
    copy_file(Path(plan["initial_checkpoint"]), portable / "weights/original-baseline-best.pt")
    last = TRIAL / "runs/boards1000_balanced_jetson30/weights/last.pt"
    copy_file(last, portable / "weights/last.pt")
    hashes = {sha(p) for p in files(portable / "weights")}
    require({selected_sha, candidate["checkpoint_sha256"], baseline["checkpoint_sha256"]} <= hashes, "Portable selected/candidate/baseline weights missing")
    jwrite(portable / "configs/train.json", summary["config"])
    jwrite(portable / "configs/validation.json", plan["validation_protocol"])
    jwrite(portable / "configs/promotion_policy.json", policy)
    jwrite(portable / "configs/classes.json", index["classes"])
    for path in files(SOURCE / "code"):
        copy_file(path, portable / "code" / path.relative_to(SOURCE / "code"))
    for filename in ("prepare_boards1000_balanced_jetson.py", "run_boards1000_balanced_jetson.py", "package_boards1000_balanced_jetson.py"):
        copy_file(HERE / "work" / filename, portable / "code" / filename)
    copy_file(Path(__file__), portable / "code/archive_balanced_github.py")
    for filename in ("class_map.json", "common.py", "assemble.py", "verify_datasets.py"):
        copy_file(TRIAL / filename, portable / "code/data_preparation" / filename)
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata.get("Name")}
    require(packages.get("torch") == state["torch"] and packages.get("ultralytics") == state["ultralytics"], "Archive must run in the recorded training venv")
    write(portable / "environment/requirements-lock.txt", "\n".join(f"{n}=={v}" for n, v in sorted(packages.items(), key=lambda p:p[0].lower())) + "\n")
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    jwrite(portable / "environment/runtime.json", {"captured_at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "gpu": state["gpu"], "gpu_driver_memory": driver, "torch_at_training": state["torch"], "ultralytics_at_training": state["ultralytics"], "packages": packages, "notes": "Package inventory captured in recorded training venv. Fresh installation not performed. Credentials and library account settings are excluded."})
    copy_file(REPO / ORIGINAL_NAME / "licenses/ULTRALYTICS_AGPL-3.0.txt", portable / "licenses/ULTRALYTICS_AGPL-3.0.txt")
    copy_file(REPO / ORIGINAL_NAME / "SOURCES.md", portable / "licenses/baseline_dataset_sources.md")
    jwrite(portable / "PORTABLE_DATASET.json", {"original_training_dataset_fingerprint": FINGERPRINT, "portable_dataset_fingerprint": fingerprint(portable / "datasets/boards_v1"), "difference": "Only data.yaml replaced with relative paths; original YAML and original result bytes preserved separately. Image/label/index bytes unchanged.", "images": verified["images"], "frozen_evaluation_images": 350, "same_source": "IoTKITs v1; existing dataset plus40 reviewed Nano candidates", "physical_scene_independence": "NOT_VERIFIED"})
    compile(VERIFY_SCRIPT, "verify_snapshot.py", "exec")
    compile(REPRODUCE_SCRIPT, "reproduce_training.py", "exec")
    write(portable / "scripts/verify_snapshot.py", VERIFY_SCRIPT)
    write(portable / "scripts/reproduce_training.py", REPRODUCE_SCRIPT)
    write(portable / "README.md", "This is a portable restoration copy. Run python scripts/verify_snapshot.py first. Original result bytes/manifest are preserved under ../local_package/. Dataset data.yaml has relative paths; use scripts/reproduce_training.py --action train with a new external --output folder. No automatic test is performed.\n")
    snapshot_manifest = manifest(portable, "SNAPSHOT_MANIFEST.json", "All portable snapshot files except this manifest; includes exact image/label bytes, relative YAML, weights and new restoration helpers")

    asset_names = {p.name for p in files(assets) if p.suffix.lower() == ".pt"}
    for path in files(SOURCE):
        relative = path.relative_to(SOURCE)
        if path.suffix.lower() == ".pt":
            continue
        if relative.as_posix() == "ARTIFACT_MANIFEST.json":
            copy_file(path, git_root / "evidence/original_local_artifact_manifest.json")
        elif relative.as_posix() == "README.md":
            write(git_root / "RESULTS.md", rewrite_links(path.read_text(encoding="utf-8"), asset_names))
            copy_file(path, git_root / "evidence/original_local_results.txt")
        elif path.suffix.lower() == ".md":
            write(git_root / relative, rewrite_links(path.read_text(encoding="utf-8"), asset_names))
        else:
            copy_file(path, git_root / relative)
    for directory in ("environment", "licenses", "configs", "scripts"):
        for path in files(portable / directory):
            copy_file(path, git_root / directory / path.relative_to(portable / directory))
    copy_file(Path(__file__), git_root / "code/archive_balanced_github.py")
    copy_file(portable / "PORTABLE_DATASET.json", git_root / "evidence/portable_dataset_provenance.json")
    for filename in ROOT_FILES:
        copy_file(REPO / filename, git_root / "evidence/root_before_balanced" / filename)
    write(git_root / ".gitignore", "*.pt\ndatasets/\nweights/\n__pycache__/\n*.py[cod]\n.runtime/\nnew_runs/\n")
    decision = "검증 승격 기준을 통과해 균형 데이터 후보를 선택했습니다." if promoted else "검증 승격 기준을 통과하지 못해 이전 미세조정 선택 모델을 유지했습니다."
    test_text = (f"선택 고정 뒤 후보 test 200장을 1회 평가한 결과 mAP50 **{test['overall'][AP50]*100:.2f}%**, mAP50–95 **{test['overall'][AP]*100:.2f}%**입니다."
                 if test else "이번 후보의 test 실행은 **0회**입니다. 유지한 이전 모델의 성능은 이전 미세조정 기록을 따릅니다.")
    close_epoch = summary["config"]["epochs"] - summary["config"]["close_mosaic"] + 1
    close_reached = bool(summary["config"]["close_mosaic"] and summary["epochs_completed"] >= close_epoch)
    mosaic_text = (f"mosaic 종료는 예정된 {close_epoch}에폭부터 적용하는 설정이며 실제 학습이 그 단계에 도달했습니다."
                   if close_reached else f"mosaic 종료는 최대 {summary['config']['epochs']}에폭 일정의 {close_epoch}에폭부터 예정됐지만 실제 {summary['epochs_completed']}에폭에서 종료되어 해당 단계에 도달하지 않았습니다.")
    jetson_text = f"Jetson val AP50–95는 원래 **{old_classes['jetson']*100:.2f}%**, 새 후보 **{new_classes['jetson']*100:.2f}%**입니다."
    original_test = read(SOURCE / "evidence/original50/test-evaluation.json")
    previous_test = read(SOURCE / "evidence/previous_finetune/test-evaluation.json")
    original_test_jetson = next(r["map50_95"] for r in original_test["per_class"] if r["class_name"] == "jetson")
    previous_test_jetson = next(r["map50_95"] for r in previous_test["per_class"] if r["class_name"] == "jetson")
    historical_test_text = f"이전 원래 모델 → 이전 미세조정 모델의 역사적 Jetson test AP50–95는 **{original_test_jetson*100:.2f}% → {previous_test_jetson*100:.2f}%**입니다. 이번 후보의 점수가 아니며 승격 조건은 val만 사용했습니다."
    status = {"revision": NAME, "promoted": promoted, "status": "VALIDATION_PROMOTED_AND_DEVELOPMENT_TESTED" if promoted else "CANDIDATE_REJECTED_PRIOR_SELECTION_RETAINED", "selection_split": "val", "selected_checkpoint_sha256": selected_sha, "candidate_checkpoint_sha256": candidate["checkpoint_sha256"], "original_baseline_checkpoint_sha256": baseline["checkpoint_sha256"], "prior_selected_checkpoint_sha256": previous["checkpoint_sha256"], "split_images": verified["images"], "jetson_train_subtypes": correction["jetson_train_subtype_image_counts"], "dataset_fingerprint": FINGERPRINT, "epochs_completed": summary["epochs_completed"], "maximum_epochs": summary["config"]["epochs"], "best_epoch": summary["best_epoch"], "optimizer_steps": summary["actual_optimizer_steps"], "training_seconds": summary["training_seconds"], "test_invocations": state["test_invocations"], "candidate_test_map50": test["overall"][AP50] if test else None, "candidate_test_map50_95": test["overall"][AP] if test else None, "baseline_val_map50_95": baseline["overall"][AP], "prior_selected_val_map50_95": previous["overall"][AP], "candidate_val_map50_95": candidate["overall"][AP], "scheduled_mosaic_close_first_epoch": close_epoch, "mosaic_close_phase_reached": close_reached, "release": RELEASE, "deployed_to_d455": False, "nucleo_included": False, "ports_trained": False, "d455_accuracy_verified": False, "test_role": policy["test_role"], "physical_scene_independence": "NOT_VERIFIED"}
    jwrite(git_root / "STATUS.json", status)
    write(git_root / "README.md", f"""# 보드 1,000장 · Jetson 학습 자료 보완 · 2026-09-30

**{decision}** {test_text}

기존 Jetson train의 반복 TX2 40장을 검수한 Nano 40장으로 바꿔 **TX2 27장 + Nano 40장**을 학습에 넣었습니다. 총 1,000장과 650 / 150 / 200 분할을 유지했고 val/test 350장의 이미지·라벨·index record는 바이트까지 그대로입니다. 다른 클래스 train 583장도 유지했습니다.

실제 추가 학습은 **{summary['epochs_completed']}에폭**(최대 {summary['config']['epochs']}, 후보 best {summary['best_epoch']})이며 initializer는 원래 50에폭 best입니다. 동일 val mAP50–95는 원래 baseline **{baseline['overall'][AP]*100:.2f}%**, 직전 선택 모델 **{previous['overall'][AP]*100:.2f}%**, 새 후보 **{candidate['overall'][AP]*100:.2f}%**입니다. 사전에 고정한 global·클래스·Jetson·직전 모델 대비 조건을 모두 확인해 선택했습니다.

{jetson_text} {mosaic_text}

{historical_test_text}

| 내용 | 파일 |
|---|---|
| 결과·학습곡선·클래스별 비교 | [RESULTS.md](RESULTS.md) |
| 설정·선택 규칙 | [train.json](configs/train.json) · [승격 정책](configs/promotion_policy.json) · [선택 원자료](evidence/selection.json) |
| 결과와 선택의 독립 감사 | [outcome_audit.json](evidence/outcome_audit.json) |
| 데이터 교체·검수 근거 | [보완 검증](evidence/dataset/data_correction_verification.json) · [검수](evidence/dataset/visual_review.json) |
| 현재 후보 상태 | [STATUS.json](STATUS.json) |
| 가중치·정확한 1,000장·완전 복원 ZIP | [Release]({RELEASE}) |
| 다른 PC 복원·검증·새 학습 | [REPRODUCE.md](REPRODUCE.md) |
| Git 묶음 해시 | [GIT_MANIFEST.json](GIT_MANIFEST.json) |

전체 Release에는 exact1,000장/라벨, candidate/last/selected/original-baseline 가중치, source annotations, 검수·제외 근거·GT contact sheets, 실제 코드·설정·환경·라이선스가 있습니다. `local_package/`는 원래 로컬 결과·manifest를 수정 없이 보존했습니다. `portable/`는 새 PC용 상대경로 data.yaml과 별도 `SNAPSHOT_MANIFEST.json`을 갖습니다. YAML을 바꾸면서 portable 전체 fingerprint가 달라지므로 원래 학습 fingerprint `{FINGERPRINT}`와 구분합니다. 이미지·라벨 바이트는 동일합니다.

선택은 val만 사용하며 test는 반복 사용한 개발 평가입니다. 다음 데이터 실험의 방향에 앞선 개발 test 오류를 활용했으므로 새 최종시험으로 해석할 수 없습니다. 데이터와 학습 설정을 함께 바꿔 순수 데이터 효과/에폭 효과를 분리하지 않았습니다. Nano 출처 그룹·pHash·육안 라벨 검사는 실제 실물/촬영 세션 독립성의 증명이 아닙니다. Nucleo·포트·D455 실측은 미검증이고 D455 기본 프로그램에 배포하지 않았습니다.

[원래 50에폭]({URL}/tree/main/{ORIGINAL_NAME})과 [중간 미세조정]({URL}/tree/main/{FINE_NAME}) 기록·Release를 모두 보존했습니다.
""")
    write(git_root / "REPRODUCE.md", f"""# 완전 복원과 새 학습

비공개 저장소에 접근할 수 있는 계정에서 다운로드합니다. 이번 ZIP 하나에 정확한 새 1,000장과 모든 필요 가중치가 있습니다.

```powershell
gh release download {TAG} --repo hkjung1011/pcb-d455-component-vision --pattern {ZIP_NAME} --pattern SHA256SUMS.txt
Get-FileHash .\\{ZIP_NAME} -Algorithm SHA256
Get-Content .\\SHA256SUMS.txt
Expand-Archive -LiteralPath .\\{ZIP_NAME} -DestinationPath .\\restored
Set-Location .\\restored\\{NAME}\\portable
python .\\scripts\\verify_snapshot.py
```

`verify_snapshot.py`는 표준 라이브러리만 사용해 모든 portable 파일 해시, exact650/150/200 이미지·라벨·그룹, 350장 val/test의 동결 해시를 확인합니다. 이미지·라벨의 원래 바이트와 portable data.yaml의 차이를 분리해 기록했고 기존 Commons/pHash 검사를 새로 실행하는 도우미는 아닙니다. `local_package/{SOURCE.name}`에는 수정하지 않은 원래 패키지와 `ARTIFACT_MANIFEST.json`이 있습니다. `ZIP_MANIFEST.json`은 ZIP 전체, `portable/SNAPSHOT_MANIFEST.json`은 경로를 바꾼 복원 묶음, Git의 `GIT_MANIFEST.json`은 가중치/원본 이미지를 생략한 탐색용 기록의 해시입니다.

환경은 [전체 런타임](environment/runtime.json)과 [패키지 목록](environment/requirements-lock.txt)에 있습니다. Python 3.11 환경에서 예를 들면:

```powershell
py -3.11 -m venv C:\\BoardEnvs\\balanced11
C:\\BoardEnvs\\balanced11\\Scripts\\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r .\\environment\\requirements-lock.txt
C:\\BoardEnvs\\balanced11\\Scripts\\python.exe .\\scripts\\reproduce_training.py --action check
C:\\BoardEnvs\\balanced11\\Scripts\\python.exe .\\scripts\\reproduce_training.py --action train --output C:\\BoardExperiments --name balanced_reproduction_01
```

새 출력은 보존 스냅샷 바깥에 만들고, initializer는 `weights/original-baseline-best.pt`, 설정은 `configs/train.json`을 사용합니다. 도우미는 새 train/val만 실행하고 test를 자동 재실행하지 않습니다. 원래 선택·test 결과를 덮어쓰지 않습니다. 실제 runner의 관측 callback·승격 평가 전체를 자동 복제하는 도우미는 아니며 설치 자체는 이번 보관 작업에서 새 환경으로 시험하지 않았습니다. OS/드라이버/라이브러리 차이로 가중치나 점수가 비트 단위로 같다고 보장하지 않습니다.

새 환경·실험 결과는 동결된 스냅샷 밖에 둡니다. Python `__pycache__`와 Ultralytics `.cache`는 재실행 중 생성되는 캐시라 manifest 파일 집합 검사에서 제외하지만, 기록된 파일 해시는 모두 그대로 검사합니다. {mosaic_text}

```python
from ultralytics import YOLO
model = YOLO('weights/selected-best.pt')
model.predict(source='board_photo.jpg', imgsz=640, conf=0.25, save=True)
```

선택 모델 SHA: `{selected_sha}`. conf0.25는 예시이며 test로 최적화한 운영 임계값이 아닙니다. 유지/승격 판단은 [selection.json](evidence/selection.json)의 val 정책을 따릅니다. D455 기본 모델로 배포하지 않았습니다.
""")
    copy_file(git_root / "REPRODUCE.md", portable / "REPRODUCE.md")
    # Include the final restoration guide in the fresh portable manifest.
    snapshot_manifest = manifest(portable, "SNAPSHOT_MANIFEST.json", "All portable files except this manifest; exact image/label bytes and restoration-only YAML change")
    latest = {"revision": NAME, "readme": f"{NAME}/README.md", "status": f"{NAME}/STATUS.json", "selection": f"{NAME}/evidence/selection.json", "release": RELEASE, "promoted": promoted, "selection_split": "val", "selected_checkpoint_sha256": selected_sha, "candidate_checkpoint_sha256": candidate["checkpoint_sha256"], "epochs_completed": summary["epochs_completed"], "test_invocations": state["test_invocations"], "candidate_dataset_fingerprint": FINGERPRINT, "data_change": "train-only40TX2 replaced by40Nano; val/test350 frozen", "deployed_to_d455": False}
    current_status = copy.deepcopy(prior_status)
    if promoted:
        current_status.update({"revision": NAME, "status": "BALANCED_DATA_VALIDATION_PROMOTED_AND_DEVELOPMENT_TESTED", "epochs_completed": summary["epochs_completed"], "best_epoch": summary["best_epoch"], "epoch_numbering": "balanced phase only; original50 initializer and fresh optimizer/schedule", "optimizer_steps": summary["actual_optimizer_steps"], "training_seconds": summary["training_seconds"], "best_sha256": selected_sha, "initial_sha256": baseline["checkpoint_sha256"], "dataset_fingerprint": FINGERPRINT, "test_invocations": 1, "test_map50": test["overall"][AP50], "test_map50_95": test["overall"][AP], "release": RELEASE, "data_release": RELEASE, "readme": f"{NAME}/README.md", "model_card": f"{NAME}/README.md", "configuration": f"{NAME}/configs/train.json", "evaluation": f"{NAME}/evidence/test-evaluation.json", "selection": f"{NAME}/evidence/selection.json", "selection_split": "val", "previous_root_status": f"{NAME}/evidence/root_before_balanced/CURRENT_STATUS.json", "deployed_to_d455": False})
    else:
        require(current_status["best_sha256"] == selected_sha, "Failure must retain prior model registry")
    current_status["latest_tuning_experiment"] = latest
    current_status.setdefault("tuning_history", []).append(latest)
    section = f"""## 2026-09-30 · Jetson 학습 자료 보완 결과

**{decision}** {test_text}

train만 TX2 40장 → 검수 Nano 40장으로 교체해 TX2 27 + Nano 40장을 구성했습니다. exact650/150/200과 val/test350장 바이트를 유지했습니다. 새 후보 val mAP50–95 **{candidate['overall'][AP]*100:.2f}%**, 원래 baseline **{baseline['overall'][AP]*100:.2f}%**, 직전 선택 모델 **{previous['overall'][AP]*100:.2f}%**; 실제 추가 **{summary['epochs_completed']}에폭**, 후보 best {summary['best_epoch']}입니다.

{jetson_text} {mosaic_text}

{historical_test_text}

[결과·설정]({NAME}/README.md) · [검증 선택 근거]({NAME}/evidence/selection.json) · [복원]({NAME}/REPRODUCE.md) · [정확한 새1,000장·가중치 Release]({RELEASE})

이전 원래50에폭·중간 미세조정과 Release는 보존했습니다. 모델 선택은 val만 사용했고 test는 재사용 개발 평가입니다. 실제 촬영 세션 독립성, Nucleo·포트·D455 실측은 미검증이며 D455 기본 모델로 배포하지 않았습니다.
"""
    root_updates = STAGE / "root_updates"
    for filename in ("README.md", "CURRENT_SUMMARY.md", "CHANGELOG.md"):
        write(root_updates / filename, prepend(root_original[filename], section))
    jwrite(root_updates / "CURRENT_STATUS.json", current_status)
    git_manifest = manifest(git_root, "GIT_MANIFEST.json", "All Git snapshot files except this manifest; image dataset and .pt weights are in Release only")
    require(not any(p.suffix.lower() == ".pt" for p in files(git_root)), "Git snapshot contains weights")
    shutil.copytree(git_root, zip_root / "github_documents")
    shutil.copytree(root_updates, zip_root / "github_root_documents")
    zip_manifest = manifest(zip_root, "ZIP_MANIFEST.json", "All full snapshot files except ZIP_MANIFEST.json; original local package plus portable dataset/model/configuration/evidence")
    with zipfile.ZipFile(assets / ZIP_NAME, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files(zip_root):
            info = zipfile.ZipInfo(f"{NAME}/{path.relative_to(zip_root).as_posix()}", date_time=(2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(assets / ZIP_NAME) as archive:
        require(set(archive.namelist()) == {f"{NAME}/{p.relative_to(zip_root).as_posix()}" for p in files(zip_root)}, "ZIP file set differs")
        for entry in zip_manifest["files"]:
            data = archive.read(f"{NAME}/{entry['path']}")
            require(len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], f"ZIP roundtrip differs: {entry['path']}")
    write(assets / "SHA256SUMS.txt", "".join(f"{sha(p)}  {p.name}\n" for p in files(assets)))
    write(STAGE / "release_notes.md", f"""Jetson 하위 종류의 train 자료를 보완한 별도 실험입니다. {decision}

{test_text}

{jetson_text} {mosaic_text}

{historical_test_text}

- 총1,000장 train650 / val150 / test200. train의 반복TX2 40장 대신 검수Nano40장; TX227+Nano40장.
- val/test350장과 다른 클래스 train583장의 기록·이미지·라벨 바이트 보존.
- 실제{summary['epochs_completed']}에폭(최대{summary['config']['epochs']}), 후보best{summary['best_epoch']}, optimizer갱신{summary['actual_optimizer_steps']}회.
- 후보val mAP50–95{candidate['overall'][AP]*100:.2f}% / originalbaseline{baseline['overall'][AP]*100:.2f}% / priorselected{previous['overall'][AP]*100:.2f}%.
- 선택은 사전고정val gate만 사용; test로 승격/유지를 바꾸지 않음. 이 test는 재사용 개발 평가.
- 전체ZIP에 exact새1,000장+라벨, 후보/last/선택/원래baseline, source annotations, 검수/제외/GT sheets, 설정/실행코드/환경/라이선스 동봉.
- 원래 결과 bytes/manifest는 local_package에 보존; portablecopy는 상대경로YAML과 새SNAPSHOT_MANIFEST; ZIP_MANIFEST와 외부SHA256SUMS로 무결성 확인.
- 기존실험/Release 보존. 물리장면독립성·Nucleo·포트·D455정확도 미검증; D455기본프로그램에 배포하지 않음.

[설정·결과]({URL}/tree/main/{NAME})
""")
    # Only root navigation/status docs may overwrite existing checkout files.
    shutil.copytree(git_root, REPO / NAME)
    for filename in ROOT_FILES:
        copy_file(root_updates / filename, REPO / filename)
    after = inventory(REPO)
    changed = sorted(n for n, digest in before.items() if after.get(n) != digest)
    require(changed == sorted(ROOT_FILES), f"Unexpected prior archive change: {changed}")
    require(not (set(before)-set(after)), "An existing archive file was removed")
    added = sorted(set(after)-set(before))
    require(all(n.startswith(NAME + "/") for n in added), "Unexpected addition outside new snapshot")
    for entry in git_manifest["files"]:
        require(sha(REPO / NAME / entry["path"]) == entry["sha256"], "Copied Git snapshot hash differs")
    result = {"prepared_at": datetime.now(timezone.utc).isoformat(), "base_commit": head, "repository": str(REPO), "new_snapshot": str(REPO / NAME), "stage": str(STAGE), "tag": TAG, "release_title": "Boards1000 balanced Jetson train-only correction · 2026-09-30", "release_notes": str(STAGE / "release_notes.md"), "release_assets": [str(p) for p in files(assets)], "promoted": promoted, "selected_checkpoint_sha256": selected_sha, "candidate_checkpoint_sha256": candidate["checkpoint_sha256"], "changed_existing_files": changed, "new_git_files": len(added), "previous_archive_files_preserved": len(before)-len(changed), "original_dataset_fingerprint": FINGERPRINT, "portable_dataset_fingerprint": fingerprint(portable / "datasets/boards_v1"), "portable_manifest_files": len(snapshot_manifest["files"]), "full_zip_files": len(zip_manifest["files"])+1, "zip_roundtrip_verified": True, "github_network_actions_performed": False, "git_commit_or_push_performed": False, "note": "Prepared locally only. Root must review, publish and verify remote bytes separately."}
    jwrite(STAGE / "archive_preparation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
