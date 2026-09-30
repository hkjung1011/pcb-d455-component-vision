"""Prepare a local Git/Release archive; never commit, push, or contact GitHub.

Run only after the fine-tune output package is complete and reviewed. This helper
copies a new snapshot into the existing checkout and prepares Release assets.
It performs no model loading, inference, training, or test evaluation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from urllib.parse import quote, unquote, urlsplit
import zipfile


HERE = Path(__file__).resolve().parent.parent
REPO = Path(r"C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\github_private_archive")
SOURCE = HERE / "outputs/BOARDS1000_FINETUNE20_2026-09-30"
TRIAL = HERE / "work/boards1000_finetune20_20260930"
STAGE = HERE / "work/github_finetune_20260930"
NAME = "BOARDS1000_FINETUNE_2026-09-30"
TAG = "boards1000-finetune-2026-09-30"
URL = "https://github.com/hkjung1011/pcb-d455-component-vision"
RELEASE = f"{URL}/releases/tag/{TAG}"
PREVIOUS_NAME = "BOARDS1000_2026-09-30"
PREVIOUS_TAG = "boards1000-2026-09-30"
PREVIOUS_RELEASE = f"{URL}/releases/tag/{PREVIOUS_TAG}"
ZIP_NAME = "BOARDS1000_FINETUNE_FULL_SNAPSHOT.zip"
ROOT_DOCUMENTS = ("README.md", "CURRENT_SUMMARY.md", "CHANGELOG.md", "CURRENT_STATUS.json")
AP_KEY = "metrics/mAP50-95(B)"
AP50_KEY = "metrics/mAP50(B)"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def jwrite(path: Path, value) -> None:
    write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files(path: Path):
    return sorted(p for p in path.rglob("*") if p.is_file())


def copy_file(source: Path, destination: Path) -> None:
    require(not source.is_symlink(), f"Unexpected symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def record(path: Path, relative: str) -> dict:
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_local_manifest() -> dict:
    manifest = read(SOURCE / "ARTIFACT_MANIFEST.json")
    listed = set()
    for row in manifest["files"]:
        relative = PurePosixPath(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Unsafe manifest path: {relative}")
        path = SOURCE.joinpath(*relative.parts)
        require(path.is_file() and not path.is_symlink(), f"Missing/unsafe output: {relative}")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                f"Output hash mismatch: {relative}")
        require(relative.as_posix() not in listed, f"Duplicate manifest entry: {relative}")
        listed.add(relative.as_posix())
    actual = {p.relative_to(SOURCE).as_posix() for p in files(SOURCE)
              if p.name != "ARTIFACT_MANIFEST.json"}
    require(listed == actual, "Output manifest does not cover the exact output package")
    return manifest


def inventory_repo() -> dict[str, str]:
    return {p.relative_to(REPO).as_posix(): sha(p) for p in files(REPO)
            if ".git" not in p.relative_to(REPO).parts}


def rewrite_weight_links(text: str, asset_names: set[str]) -> str:
    # Git has no .pt files. Keep original Markdown unchanged inside the ZIP.
    def replace(match):
        target = match.group(2).strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or not parsed.path.lower().endswith(".pt"):
            return match.group(0)
        basename = Path(unquote(parsed.path).replace("\\", "/")).name
        destination = (f"{URL}/releases/download/{TAG}/{quote(basename)}"
                       if basename in asset_names else RELEASE)
        return f"{match.group(1)}({destination})"
    return re.sub(r"(!?\[[^\]]*\])\(([^)]+)\)", replace, text)


def prepend_section(original: str, section: str) -> str:
    lines = original.splitlines(keepends=True)
    if lines and lines[0].startswith("# "):
        return lines[0].rstrip("\r\n") + "\n\n" + section.rstrip() + "\n\n" + "".join(lines[1:]).lstrip("\r\n")
    return section.rstrip() + "\n\n" + original


def main() -> None:
    require(SOURCE.is_dir(), "Completed output package is missing")
    require(not STAGE.exists(), "Archive staging already exists; preserve it and inspect rather than rerun")
    require(not (REPO / NAME).exists(), "Git snapshot already exists; preserve it")
    require(REPO.is_dir(), "Existing archive checkout is missing")
    branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    require(branch == "main" and not dirty.strip(), "Existing checkout must be clean main before archiving")
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    verify_local_manifest()

    state = read(TRIAL / "execution_state.json")
    selection = read(TRIAL / "selection.json")
    summary = read(TRIAL / "training-summary.json")
    plan = read(TRIAL / "experiment_plan.json")
    baseline = read(TRIAL / "baseline_validation.json")
    candidate = read(TRIAL / "candidate_validation.json")
    root_status = read(REPO / "CURRENT_STATUS.json")
    require(state["status"] == "complete", "Fine-tune execution has not completed")
    require(selection["selection_split"] == "val", "Selection must use validation only")
    require(baseline["protocol"] == candidate["protocol"] == plan["validation_protocol"],
            "Paired validation protocols differ")
    require(baseline["checkpoint_sha256"] == selection["baseline_checkpoint_sha256"] ==
            plan["initial_checkpoint_sha256"] == root_status["best_sha256"],
            "Baseline checkpoint does not match currently archived selected model")
    require(candidate["checkpoint_sha256"] == summary["best_sha256"] ==
            selection["candidate_checkpoint_sha256"], "Candidate hashes differ")
    require(plan["dataset_fingerprint"] == state["dataset_fingerprint"] ==
            root_status["dataset_fingerprint"], "Fine-tune dataset differs from archived dataset")
    gain = candidate["overall"][AP_KEY] - baseline["overall"][AP_KEY]
    before = {r["class_name"]: r["map50_95"] for r in baseline["per_class"]}
    after = {r["class_name"]: r["map50_95"] for r in candidate["per_class"]}
    require(before.keys() == after.keys(), "Validation class sets differ")
    deltas = {name: after[name] - value for name, value in before.items()}
    policy = selection["policy"]
    require(policy == plan["promotion_policy"], "Promotion policy differs from frozen plan")
    expected_promotion = (gain >= policy["minimum_overall_ap50_95_gain"] and
                          min(deltas.values()) >= -policy["maximum_per_class_ap50_95_drop"])
    require(abs(gain - selection["overall_ap_gain"]) < 1e-10, "Recorded validation gain differs")
    require(all(abs(deltas[n] - selection["per_class_ap_gains"][n]) < 1e-10 for n in deltas),
            "Recorded per-class validation gains differ")
    promoted = selection["promoted"]
    require(isinstance(promoted, bool) and promoted == expected_promotion == state["promoted"],
            "Promotion result does not match frozen validation gate")
    expected_hash = (selection["candidate_checkpoint_sha256"] if promoted
                     else selection["baseline_checkpoint_sha256"])
    require(selection["selected_checkpoint_sha256"] == state["selected_checkpoint_sha256"] == expected_hash,
            "Selected checkpoint is inconsistent with validation gate")
    test = None
    if promoted:
        test = read(TRIAL / "test-evaluation.json")
        require(state["test_invocations"] == test["invocations"] == 1,
                "Promoted candidate must have exactly one completed test invocation")
        require(test["checkpoint_sha256"] == expected_hash and test["images"] == 200,
                "Candidate test checkpoint/split mismatch")
    else:
        require(state["test_invocations"] == 0 and not (TRIAL / "test-evaluation.json").exists(),
                "Non-promoted candidate should not have been tested")
    weight_files = [p for p in files(SOURCE) if p.suffix.lower() == ".pt"]
    require(len(weight_files) >= 2, "Complete output must contain candidate and selected weights")
    asset_names = {p.name for p in weight_files}
    require(len(asset_names) == len(weight_files), "Standalone Release weight names must be unique")
    weight_hashes = {sha(p) for p in weight_files}
    require(selection["candidate_checkpoint_sha256"] in weight_hashes and expected_hash in weight_hashes,
            "Candidate/selected weights are missing from output")
    for name in ("execution_state.json", "selection.json", "training-summary.json", "experiment_plan.json",
                 "baseline_validation.json", "candidate_validation.json"):
        require(sha(SOURCE / "evidence" / name) == sha(TRIAL / name),
                f"Package evidence differs from completed trial: {name}")
    before_inventory = inventory_repo()
    original_docs = {name: (REPO / name).read_text(encoding="utf-8") for name in ROOT_DOCUMENTS}
    STAGE.mkdir()
    git_snapshot = STAGE / NAME
    assets = STAGE / "release_assets"
    assets.mkdir()

    for path in files(SOURCE):
        relative = path.relative_to(SOURCE)
        if path.suffix.lower() == ".pt":
            copy_file(path, assets / path.name)
        elif relative.as_posix() == "ARTIFACT_MANIFEST.json":
            copy_file(path, git_snapshot / "evidence/original_local_artifact_manifest.json")
        elif relative.as_posix() == "README.md":
            results_text = rewrite_weight_links(path.read_text(encoding="utf-8"), asset_names)
            results_text = results_text.replace("(ARTIFACT_MANIFEST.json)", "(evidence/original_local_artifact_manifest.json)")
            write(git_snapshot / "RESULTS.md", results_text)
            copy_file(path, git_snapshot / "evidence/original_local_results.txt")
        elif path.suffix.lower() == ".md":
            write(git_snapshot / relative, rewrite_weight_links(path.read_text(encoding="utf-8"), asset_names))
        else:
            copy_file(path, git_snapshot / relative)
    for name in ROOT_DOCUMENTS:
        copy_file(REPO / name, git_snapshot / "evidence/root_before_finetune" / name)
    copy_file(Path(__file__), git_snapshot / "code/archive_finetune_github.py")
    write(git_snapshot / ".gitignore", "*.pt\n__pycache__/\n*.py[cod]\n.runtime/\nnew_runs/\n")
    jwrite(git_snapshot / "configs/train.json", summary["config"])
    jwrite(git_snapshot / "configs/validation.json", plan["validation_protocol"])
    jwrite(git_snapshot / "configs/promotion_policy.json", policy)
    jwrite(git_snapshot / "configs/classes.json", plan["classes"])
    for filename in ("requirements-lock.txt", "runtime.json"):
        origin = REPO / PREVIOUS_NAME / "environment" / filename
        if origin.is_file():
            copy_file(origin, git_snapshot / "environment/baseline_run" / filename)
    # These copied inventories are explicitly the original run's records; the
    # fine-tune state supplies current torch/Ultralytics/GPU observations.
    jwrite(git_snapshot / "environment/finetune_runtime.json", {
        "torch": state["torch"], "ultralytics": state["ultralytics"], "gpu": state["gpu"],
        "source": "completed fine-tune execution_state.json",
        "baseline_inventory_scope": "environment/baseline_run is the preceding 50-epoch run inventory, not a fresh environment installation check",
    })
    stage_status = {
        "revision": NAME, "status": "PROMOTED_BY_VALIDATION" if promoted else "BASELINE_RETAINED_BY_VALIDATION",
        "experiment_role": "intermediate validation-selected fine-tune; not a blanket improvement or deployed camera model",
        "promoted": promoted, "selection_split": "val", "selection_policy": policy,
        "split_images": {"train": 650, "val": 150, "test": 200}, "classes": plan["classes"],
        "epochs_completed": summary["epochs_completed"], "maximum_epochs": summary["config"]["epochs"],
        "best_finetune_epoch": summary["best_epoch"], "actual_optimizer_steps": summary["actual_optimizer_steps"],
        "training_seconds": summary["training_seconds"], "baseline_checkpoint_sha256": selection["baseline_checkpoint_sha256"],
        "candidate_checkpoint_sha256": selection["candidate_checkpoint_sha256"], "selected_checkpoint_sha256": expected_hash,
        "baseline_validation_map50_95": baseline["overall"][AP_KEY], "candidate_validation_map50_95": candidate["overall"][AP_KEY],
        "validation_gain": gain, "test_invocations": state["test_invocations"],
        "candidate_test_map50": test["overall"][AP50_KEY] if test else None,
        "candidate_test_map50_95": test["overall"][AP_KEY] if test else None,
        "dataset_fingerprint": plan["dataset_fingerprint"], "same_dataset_as": PREVIOUS_NAME,
        "data_release": PREVIOUS_RELEASE, "release": RELEASE,
        "nucleo_included": False, "ports_trained": False, "d455_accuracy_verified": False,
        "deployed_to_d455": False,
        "test_role": policy["test_role"], "archive_created_at": datetime.now(timezone.utc).isoformat(),
    }
    jwrite(git_snapshot / "STATUS.json", stage_status)
    decision = "검증 승격 조건을 통과해 후보 모델을 선택했습니다." if promoted else "검증 승격 조건을 통과하지 못해 기존 50에폭 모델을 유지했습니다."
    test_text = (f"후보 개발 test는 mAP50 **{test['overall'][AP50_KEY]*100:.2f}%**, mAP50–95 **{test['overall'][AP_KEY]*100:.2f}%**이며 선택 고정 후 1회 평가했습니다."
                 if test else "후보 test는 실행하지 않았습니다. 기존 모델의 test 기록은 이전 스냅샷에 있습니다.")
    test_limit = ""
    if test:
        old_test = read(SOURCE / "evidence/baseline50/test-evaluation.json")
        old_by_class = {r["class_name"]: r["map50_95"] for r in old_test["per_class"]}
        new_by_class = {r["class_name"]: r["map50_95"] for r in test["per_class"]}
        if "jetson" in old_by_class and "jetson" in new_by_class:
            test_limit = (f"재사용 개발 test의 Jetson AP50–95는 **{old_by_class['jetson']*100:.2f}% → {new_by_class['jetson']*100:.2f}%**로 내려갔습니다. "
                          "이 결과는 부족한 하위 종류/장면 다양성을 보완할 다음 데이터 실험의 근거로 기록했습니다. "
                          "이미 고정한 모델 승격을 test 점수로 뒤집지는 않았습니다.")
    write(git_snapshot / "README.md", f"""# 보드 1,000장 추가 미세조정 · 중간 실험 · 2026-09-30

**{decision}** 기존 50에폭 모델에서 낮은 학습률과 mosaic 종료 조건으로 새 optimizer/schedule을 시작했습니다. 최대 {summary['config']['epochs']}에폭 중 실제 {summary['epochs_completed']}에폭을 진행하고 fine-tune epoch {summary['best_epoch']}을 후보로 선택했습니다.

동일 프로토콜의 검증 mAP50–95는 **{baseline['overall'][AP_KEY]*100:.2f}% → {candidate['overall'][AP_KEY]*100:.2f}%**입니다. {test_text}

{test_limit}

이 기록은 검증 기준으로 선택한 중간 후보입니다. 모든 클래스에서 개선됐다는 뜻은 아니며 D455 기본 실행 모델로 배포하지 않았습니다. Nano 40장으로 반복된 TX2 40장을 바꾸는 데이터 보완은 별도의 다음 실험이고, 이 기록 작성 시점에는 그 결과가 확정되지 않았습니다.

| 확인할 내용 | 파일 |
|---|---|
| 결과·학습 곡선·클래스별 비교 | [RESULTS.md](RESULTS.md) |
| 실제 설정 | [train.json](configs/train.json) · [validation.json](configs/validation.json) |
| 선택 근거 | [selection.json](evidence/selection.json) · [동결한 승격 규칙](configs/promotion_policy.json) |
| 다른 PC에서 자료 복원 | [REPRODUCE.md](REPRODUCE.md) |
| 상태와 해시 | [STATUS.json](STATUS.json) · [Git 파일 목록](GIT_MANIFEST.json) |
| 후보·선택 가중치와 원래 결과 전체 | [새 Release]({RELEASE}) |
| 정확한 1,000장·라벨·초기 가중치 | [기존 데이터 Release]({PREVIOUS_RELEASE}) |

전체 1,000장의 train 650 / val 150 / test 200 분할, 이미지와 라벨은 이전 실험과 같습니다. 이번 ZIP은 데이터 사진을 중복 동봉하지 않으며 이전 `BOARDS1000_FULL_SNAPSHOT.zip`을 함께 복원해야 합니다. test는 여러 실험에서 재사용한 개발 평가이며 새로운 최종시험이 아닙니다. 모델 선택은 val 규칙으로만 결정했습니다.

Jetson은 train의 TX2 출처 그룹이 3개이고 Nano train 그룹은 0개입니다. 설정 변경으로 부족한 실물·장면 다양성이 보충되지는 않습니다. Nucleo·포트와 D455 실측 정확도는 이번 실험 범위에 포함되지 않습니다. 기존 D455 실행 프로그램에 모델을 자동 적용하지 않았습니다.

`evidence/original_local_artifact_manifest.json`은 원래 로컬 결과 묶음의 해시 기록입니다. Git용 문서의 가중치 링크를 Release로 바꿨으므로 Git의 현재 기준은 `GIT_MANIFEST.json`입니다. ZIP의 `local_package/`에는 원래 로컬 결과와 manifest를 그대로 보존했고, 전체 ZIP 내용은 `ZIP_MANIFEST.json`으로 검증합니다.
""")
    write(git_snapshot / "REPRODUCE.md", f"""# 미세조정 결과 복원

비공개 저장소 접근 권한이 있는 계정에서 두 Release를 다운로드합니다.

```powershell
gh release download {TAG} --repo hkjung1011/pcb-d455-component-vision --pattern {ZIP_NAME} --pattern SHA256SUMS.txt --dir .\\finetune_assets
gh release download {PREVIOUS_TAG} --repo hkjung1011/pcb-d455-component-vision --pattern BOARDS1000_FULL_SNAPSHOT.zip --pattern SHA256SUMS.txt --dir .\\baseline_assets
Get-FileHash .\\finetune_assets\\{ZIP_NAME} -Algorithm SHA256
Get-Content .\\finetune_assets\\SHA256SUMS.txt
Expand-Archive -LiteralPath .\\finetune_assets\\{ZIP_NAME} -DestinationPath .\\restored_finetune
Expand-Archive -LiteralPath .\\baseline_assets\\BOARDS1000_FULL_SNAPSHOT.zip -DestinationPath .\\restored_baseline
```

새 ZIP의 `{NAME}/local_package/{SOURCE.name}/`에 후보·선택 가중치, 원래 README, 실제 설정·코드·검증·test 기록과 그래프가 있습니다. `ZIP_MANIFEST.json`은 ZIP 전체의 파일별 SHA-256이며 `github_documents/GIT_MANIFEST.json`은 가중치를 제외한 Git 문서만의 해시입니다. ZIP 바이트 자체는 외부 `SHA256SUMS.txt`와 비교합니다.

이미지·YOLO 라벨·공식 초기값은 기존 ZIP의 `{PREVIOUS_NAME}/datasets/boards_v1/`와 `weights/`에 있습니다. 현재 데이터 fingerprint는 `{plan['dataset_fingerprint']}`이며 split은 650 / 150 / 200입니다. 기존 [복원 도우미]({URL}/blob/main/{PREVIOUS_NAME}/REPRODUCE.md)로 데이터 무결성을 확인할 수 있습니다.

새 optimizer와 스케줄로 기존 best에서 시작한 실험입니다. `resume=false`, lr0 `{summary['config']['lr0']}`, warmup `{summary['config']['warmup_epochs']}`, mosaic `{summary['config']['mosaic']}`, patience `{summary['config']['patience']}`입니다. 원래 실행 코드는 당시 절대경로를 포함한 증거 스냅샷이며 경로 수정 없이 다른 PC에서 그대로 실행되지 않습니다. 새 학습에는 별도 출력 경로를 지정하고 기존 결과를 보존해야 합니다. 환경 설치나 재학습을 이 보관 도우미가 실행하지 않습니다.

선택 SHA-256: `{expected_hash}`. 선택 판단은 `evidence/selection.json`의 검증 규칙을 따릅니다. test는 이미 재사용된 개발 자료이며 최종 성능 판단은 새로운 보드·촬영 세션 자료에서 해야 합니다.
""")

    latest = {
        "revision": NAME, "readme": f"{NAME}/README.md", "status": f"{NAME}/STATUS.json",
        "selection": f"{NAME}/evidence/selection.json", "release": RELEASE,
        "promoted": promoted, "selection_split": "val", "selected_checkpoint_sha256": expected_hash,
        "epochs_completed": summary["epochs_completed"], "best_finetune_epoch": summary["best_epoch"],
        "validation_gain": gain, "test_invocations": state["test_invocations"],
        "experiment_role": "intermediate validation-selected candidate; no D455 deployment",
    }
    new_root_status = copy.deepcopy(root_status)
    if promoted:
        new_root_status.update({
            "revision": NAME, "status": "FINETUNED_VALIDATION_PROMOTED_AND_DEVELOPMENT_TESTED",
            "epochs_completed": summary["epochs_completed"], "best_epoch": summary["best_epoch"],
            "epoch_numbering": "fine-tune phase only; preceding 50-epoch run uses a separate optimizer/schedule",
            "base_epochs_completed": root_status["epochs_completed"], "base_best_epoch": root_status["best_epoch"],
            "optimizer_steps": summary["actual_optimizer_steps"], "base_optimizer_steps": root_status["optimizer_steps"],
            "training_seconds": summary["training_seconds"], "test_invocations": 1,
            "test_map50": test["overall"][AP50_KEY], "test_map50_95": test["overall"][AP_KEY],
            "best_sha256": expected_hash, "initial_sha256": selection["baseline_checkpoint_sha256"],
            "official_coco_initial_sha256": root_status["initial_sha256"],
            "release": RELEASE, "readme": f"{NAME}/README.md", "model_card": f"{NAME}/README.md",
            "configuration": f"{NAME}/configs/train.json", "evaluation": f"{NAME}/evidence/test-evaluation.json",
            "selection": f"{NAME}/evidence/selection.json", "selection_split": "val",
            "data_release": PREVIOUS_RELEASE, "deployed_to_d455": False,
            "previous_root_status": f"{NAME}/evidence/root_before_finetune/CURRENT_STATUS.json",
        })
    new_root_status["latest_tuning_experiment"] = latest
    new_root_status.setdefault("tuning_history", []).append(latest)
    section = f"""## 2026-09-30 · 보드 1,000장 미세조정 중간 결과

**{decision}** 동일 검증 mAP50–95 **{baseline['overall'][AP_KEY]*100:.2f}% → {candidate['overall'][AP_KEY]*100:.2f}%**, 실제 추가 학습 **{summary['epochs_completed']}에폭**(최대 {summary['config']['epochs']}, 후보 best {summary['best_epoch']}). {test_text}

[설정·결과]({NAME}/README.md) · [선택 근거]({NAME}/evidence/selection.json) · [복원 안내]({NAME}/REPRODUCE.md) · [후보·선택 가중치 Release]({RELEASE})

{test_limit}

이 모델은 검증 기준으로 선택한 중간 후보이며 D455 기본 실행 모델로 배포하지 않았습니다. Nano 40장으로 반복 TX2 40장을 교체하는 데이터 보완은 별도의 다음 실험이고, 결과는 아직 확정되지 않았습니다.

데이터 1,000장과 650 / 150 / 200 분할은 [기존 기록]({PREVIOUS_NAME}/README.md)과 같습니다. 원래 50에폭 모델과 기존 Release를 보존했습니다. 이 실험은 fresh optimizer/schedule의 미세조정이며 에폭 수만 비교하는 대조 실험이 아닙니다. test는 재사용 개발 평가이고 선택은 val에만 근거합니다. Nucleo·포트와 D455 실측은 미검증입니다.
"""
    # Build the exact root document updates locally before touching the checkout.
    root_updates = STAGE / "root_updates"
    for filename in ("README.md", "CURRENT_SUMMARY.md", "CHANGELOG.md"):
        write(root_updates / filename, prepend_section(original_docs[filename], section))
    jwrite(root_updates / "CURRENT_STATUS.json", new_root_status)

    git_manifest = {"scope": "all files in this Git snapshot except GIT_MANIFEST.json; no weights or dataset images",
                    "files": [record(p, p.relative_to(git_snapshot).as_posix()) for p in files(git_snapshot)
                              if p.name != "GIT_MANIFEST.json"]}
    jwrite(git_snapshot / "GIT_MANIFEST.json", git_manifest)
    require(not any(p.suffix.lower() == ".pt" for p in files(git_snapshot)), "Git snapshot contains .pt weights")

    # ZIP stores the complete original local package, plus its Git browsing docs.
    zip_entries = []
    for folder, prefix in ((SOURCE, f"{NAME}/local_package/{SOURCE.name}"),
                           (git_snapshot, f"{NAME}/github_documents"),
                           (root_updates, f"{NAME}/github_root_documents")):
        zip_entries.extend((path, f"{prefix}/{path.relative_to(folder).as_posix()}") for path in files(folder))
    zip_manifest = {"scope": "all ZIP files except ZIP_MANIFEST.json; local_package is the unchanged original output",
                    "dataset_included": False, "dataset_release": PREVIOUS_RELEASE,
                    "files": [record(path, relative) for path, relative in zip_entries]}
    jwrite(STAGE / "ZIP_MANIFEST.json", zip_manifest)
    zip_entries.append((STAGE / "ZIP_MANIFEST.json", f"{NAME}/ZIP_MANIFEST.json"))
    with zipfile.ZipFile(assets / ZIP_NAME, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in sorted(zip_entries, key=lambda pair: pair[1]):
            info = zipfile.ZipInfo(relative, date_time=(2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(assets / ZIP_NAME) as archive:
        expected = {r["path"]: r for r in zip_manifest["files"]}
        require(set(archive.namelist()) == set(expected) | {f"{NAME}/ZIP_MANIFEST.json"}, "ZIP file set differs")
        for relative, row in expected.items():
            data = archive.read(relative)
            require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                    f"ZIP round-trip mismatch: {relative}")
    write(assets / "SHA256SUMS.txt", "".join(f"{sha(p)}  {p.name}\n" for p in files(assets)))
    write(STAGE / "release_notes.md", f"""보드 1,000장 미세조정의 중간 실험 결과입니다. {decision}

{test_limit}

검증 기준으로 선택한 중간 후보이며 D455 기본 실행 모델로 배포하지 않았습니다. Nano 40장으로 반복 TX2 40장을 교체하는 다음 데이터 실험은 별도이고 결과는 아직 확정되지 않았습니다.

- train 650 / val 150 / test 200, 데이터와 분할은 기존 50에폭 실험과 동일합니다.
- 최대 {summary['config']['epochs']}에폭 중 {summary['epochs_completed']}에폭, 후보 best {summary['best_epoch']}, 실제 optimizer 갱신 {summary['actual_optimizer_steps']}회.
- 동일 검증 mAP50–95 {baseline['overall'][AP_KEY]*100:.2f}% → {candidate['overall'][AP_KEY]*100:.2f}%; 선택은 동결한 val 승격 규칙으로만 결정했습니다.
- {test_text}
- 후보·선택 가중치와 전체 원래 결과 패키지를 {ZIP_NAME}에 보존했습니다. 가중치는 개별 자산으로도 있습니다.
- 데이터 1,000장과 라벨은 [이전 Release]({PREVIOUS_RELEASE})의 BOARDS1000_FULL_SNAPSHOT.zip을 복원합니다.
- SHA256SUMS.txt는 Release 자산 바이트, ZIP_MANIFEST.json은 ZIP 내용, GIT_MANIFEST.json은 Git용 문서·설정 파일의 해시입니다.
- test는 재사용 개발 평가이며 D455 실제 정확도, Nucleo·포트는 미검증입니다.

[설정·결과 기록]({URL}/tree/main/{NAME})
""")

    # The only existing files allowed to change are the four root navigation/status docs.
    shutil.copytree(git_snapshot, REPO / NAME)
    for filename in ROOT_DOCUMENTS:
        copy_file(root_updates / filename, REPO / filename)
    after_inventory = inventory_repo()
    changed_old = sorted(name for name, digest in before_inventory.items()
                         if after_inventory.get(name) != digest)
    require(changed_old == sorted(ROOT_DOCUMENTS), f"Unexpected existing-file change: {changed_old}")
    require(not (set(before_inventory) - set(after_inventory)), "An existing archive file was removed")
    new_files = sorted(set(after_inventory) - set(before_inventory))
    require(all(name.startswith(NAME + "/") for name in new_files), "Unexpected new path outside experiment snapshot")
    for row in git_manifest["files"]:
        require(sha(REPO / NAME / row["path"]) == row["sha256"], "Copied Git snapshot hash mismatch")
    result = {
        "prepared_at": datetime.now(timezone.utc).isoformat(), "base_commit": head,
        "repository": str(REPO), "new_snapshot": str(REPO / NAME), "stage": str(STAGE),
        "tag": TAG, "release_title": "Boards1000 intermediate validation-gated fine-tune · 2026-09-30",
        "release_notes": str(STAGE / "release_notes.md"), "release_assets": [str(p) for p in files(assets)],
        "promoted": promoted, "selected_checkpoint_sha256": expected_hash,
        "changed_existing_files": changed_old, "new_git_files": len(new_files),
        "previous_archive_files_preserved": len(before_inventory) - len(changed_old),
        "zip_roundtrip_verified": True,
        "github_network_actions_performed": False, "git_commit_or_push_performed": False,
        "note": "Prepared locally. Root agent must review, commit/push, upload Release assets, and verify remote bytes separately.",
    }
    jwrite(STAGE / "archive_preparation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
