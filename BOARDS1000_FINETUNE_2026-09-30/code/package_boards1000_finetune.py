"""Package completed fine-tuning evidence; never train or invoke inference."""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / 'work/boards1000_s650_v150_t200_e50_20260929'
ORIGINAL_RUN = SOURCE / 'runs/boards1000_yolo11s_50ep'
TRIAL = HERE / 'work/boards1000_finetune20_20260930'
RUN = TRIAL / 'runs/boards1000_finetune20'
OUT = HERE / 'outputs/BOARDS1000_FINETUNE20_2026-09-30'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert sha(source) == sha(destination), f'Copy checksum mismatch: {destination}'


def curve_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return [{key.strip(): float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def csv_write(path, rows):
    assert rows
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    state = read(TRIAL / 'execution_state.json')
    assert state['status'] == 'complete', 'Fine-tuning must complete before packaging'
    assert not OUT.exists(), 'Existing output preserved; use a separately reviewed new package'
    summary = read(TRIAL / 'training-summary.json')
    plan = read(TRIAL / 'experiment_plan.json')
    baseline = read(TRIAL / 'baseline_validation.json')
    candidate = read(TRIAL / 'candidate_validation.json')
    selection = read(TRIAL / 'selection.json')
    original = read(ORIGINAL_RUN / 'training-summary.json')
    old_test = read(ORIGINAL_RUN / 'test-evaluation.json')
    verified = read(SOURCE / 'datasets/verification.json')['boards_v1']
    derivation = read(SOURCE / 'derivation.json')
    rows = curve_rows(RUN / 'results.csv')
    original_rows = curve_rows(ORIGINAL_RUN / 'results.csv')
    classes = plan['classes']
    policy = plan['promotion_policy']
    protocol = plan['validation_protocol']
    promoted = selection['promoted']
    baseline_path = Path(plan['initial_checkpoint'])
    candidate_path = Path(summary['best'])
    selected_path = Path(selection['selected_checkpoint'])

    assert len(classes) == 8 and 'stm32_nucleo' not in classes
    assert verified['images'] == {'train': 650, 'val': 150, 'test': 200}
    assert verified['training_ready'] and verified['structure_pass']
    assert 0 < len(rows) == summary['epochs_completed'] == len(state['epochs']) <= 20
    assert [int(row['epoch']) for row in rows] == list(range(1, len(rows) + 1))
    assert len(original_rows) == original['epochs_completed'] == 50
    assert summary['actual_optimizer_steps'] == state['optimizer_steps'] > 0
    assert all(math.isfinite(value) for row in rows for value in row.values())
    assert sha(baseline_path) == plan['initial_checkpoint_sha256'] == original['best_sha256']
    assert baseline['checkpoint_sha256'] == selection['baseline_checkpoint_sha256'] == original['best_sha256']
    assert old_test['checkpoint_sha256'] == original['best_sha256'] and old_test['invocations'] == 1
    assert sha(candidate_path) == summary['best_sha256'] == candidate['checkpoint_sha256'] == selection['candidate_checkpoint_sha256']
    assert sha(selected_path) == selection['selected_checkpoint_sha256'] == state['selected_checkpoint_sha256']
    assert baseline['protocol'] == candidate['protocol'] == protocol and protocol['half'] is False
    assert protocol['imgsz'] == 640 and protocol['batch'] == 8
    assert verified['dataset_fingerprint'] == plan['dataset_fingerprint'] == state['dataset_fingerprint'] == original['dataset_fingerprint']
    assert policy['minimum_overall_ap50_95_gain'] == 0.005
    assert policy['maximum_per_class_ap50_95_drop'] == 0.05
    assert summary['config'] == plan['config'] == state['config']
    assert summary['config']['lr0'] == 0.0001 and summary['config']['warmup_bias_lr'] == 0.0
    assert summary['config']['mosaic'] == 0.0 and summary['config']['resume'] is False
    assert sha(HERE / 'work/run_boards1000_finetune.py') == plan['script_sha256']

    # Recheck current source inventory without running any model.
    sys.path.insert(0, str(SOURCE))
    from verify_datasets import dataset_fingerprint, holdout_inventory
    assert dataset_fingerprint(SOURCE / 'datasets/boards_v1') == verified['dataset_fingerprint']
    assert sha(SOURCE / 'class_map.json') == verified['class_map_sha256']
    assert holdout_inventory() == verified['holdout_inventory']

    by_baseline = {row['class_name']: row for row in baseline['per_class']}
    by_candidate = {row['class_name']: row for row in candidate['per_class']}
    assert set(by_baseline) == set(by_candidate) == set(classes)
    comparison = []
    for name in classes:
        b, c = by_baseline[name], by_candidate[name]
        assert b['ground_truth_boxes'] == c['ground_truth_boxes'] == verified['boxes']['val'][name]
        comparison.append({'class_name': name, 'val_boxes': b['ground_truth_boxes'],
                           'val_source_groups': derivation['class_group_counts']['val'][name],
                           'baseline_ap50_pct': b['map50'] * 100,
                           'candidate_ap50_pct': c['map50'] * 100,
                           'ap50_delta_pp': (c['map50'] - b['map50']) * 100,
                           'baseline_ap50_95_pct': b['map50_95'] * 100,
                           'candidate_ap50_95_pct': c['map50_95'] * 100,
                           'ap50_95_delta_pp': (c['map50_95'] - b['map50_95']) * 100})
    gain = candidate['overall']['metrics/mAP50-95(B)'] - baseline['overall']['metrics/mAP50-95(B)']
    worst = min(row['ap50_95_delta_pp'] / 100 for row in comparison)
    assert abs(gain - selection['overall_ap_gain']) < 1e-12
    assert all(abs(row['ap50_95_delta_pp'] / 100 - selection['per_class_ap_gains'][row['class_name']]) < 1e-12 for row in comparison)
    assert promoted == (gain >= 0.005 and worst >= -0.05)
    assert promoted == state['promoted']
    assert selected_path.resolve() == (candidate_path if promoted else baseline_path).resolve()
    test_path = TRIAL / 'test-evaluation.json'
    if promoted:
        test = read(test_path)
        marker = read(TRIAL / 'test-evaluation-attempt.json')
        assert test['invocations'] == state['test_invocations'] == 1 and test['images'] == 200
        assert test['checkpoint_sha256'] == marker['checkpoint_sha256'] == summary['best_sha256']
    else:
        assert state['test_invocations'] == 0 and not test_path.exists()
        assert not (TRIAL / 'test-evaluation-attempt.json').exists()
        test = old_test

    OUT.mkdir(parents=True)
    copy(selected_path, OUT / 'selected-best.pt')
    copy(candidate_path, OUT / 'candidate-best.pt')
    copy(baseline_path, OUT / 'baseline-best.pt')
    for name in ['execution_state.json', 'experiment_plan.json', 'baseline_validation.json',
                 'candidate_validation.json', 'selection.json', 'training-summary.json']:
        copy(TRIAL / name, OUT / 'evidence' / name)
    if (TRIAL / 'data_review.json').exists():
        copy(TRIAL / 'data_review.json', OUT / 'evidence/data_review.json')
    for name in ['results.csv', 'args.yaml']:
        copy(RUN / name, OUT / 'evidence' / name)
    for name in ['training-summary.json', 'test-evaluation.json', 'test-evaluation-attempt.json', 'results.csv', 'args.yaml']:
        copy(ORIGINAL_RUN / name, OUT / 'evidence/baseline50' / name)
    if promoted:
        for name in ['test-evaluation.json', 'test-evaluation-attempt.json']:
            copy(TRIAL / name, OUT / 'evidence' / name)
    for name in ['derivation.json', 'training_plan.json', 'class_map.json']:
        copy(SOURCE / name, OUT / 'evidence/dataset' / name)
    copy(SOURCE / 'datasets/verification.json', OUT / 'evidence/dataset/verification.json')
    copy(SOURCE / 'datasets/boards_v1/dataset_index.json', OUT / 'evidence/dataset/dataset_index.json')
    copy(SOURCE / 'datasets/boards_v1/data.yaml', OUT / 'evidence/dataset/data.yaml')
    for name in ['run_boards1000_finetune.py', 'package_boards1000_finetune.py', 'run_boards1000.py', 'prepare_boards1000.py']:
        copy(HERE / 'work' / name, OUT / 'code' / name)
    for name in ['common.py', 'assemble.py', 'verify_datasets.py']:
        copy(SOURCE / name, OUT / 'code' / name)
    for relative, label in [('baseline_val', 'baseline_val'), ('candidate_val', 'candidate_val'),
                            ('runs/boards1000_finetune20', 'training'), ('test', 'test')]:
        directory = TRIAL / relative
        if directory.exists():
            for picture in sorted(directory.glob('*')):
                if picture.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    copy(picture, OUT / 'plots' / label / picture.name)
    for path in sorted((HERE / 'work/next_training_review').glob('*')):
        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.json', '.csv', '.md'}:
            copy(path, OUT / 'data_review' / path.name)
    old_environment = HERE / 'work/github_boards1000_20260930/BOARDS1000_2026-09-30/environment'
    if old_environment.exists():
        for name in ['runtime.json', 'requirements-lock.txt']:
            if (old_environment / name).exists():
                copy(old_environment / name, OUT / 'evidence/baseline_environment' / name)

    csv_write(OUT / 'validation_per_class_comparison.csv', comparison)
    csv_write(OUT / ('candidate_test_per_class.csv' if promoted else 'previous_baseline_test_per_class.csv'), test['per_class'])
    csv_write(OUT / 'split_class_counts.csv', [
        {'class_name': name, **{f'{split}_boxes': verified['boxes'][split][name] for split in ['train', 'val', 'test']},
         **{f'{split}_source_groups': derivation['class_group_counts'][split][name] for split in ['train', 'val', 'test']}}
        for name in classes])
    package_summary = {'promoted': promoted, 'initial_training_epochs': original['epochs_completed'],
                       'fine_tuning_epochs': summary['epochs_completed'], 'candidate_best_fine_tuning_epoch': summary['best_epoch'],
                       'baseline_validation': baseline['overall'], 'candidate_validation': candidate['overall'],
                       'overall_ap50_95_gain_pp': gain * 100, 'worst_class_ap50_95_gain_pp': worst * 100,
                       'promotion_policy': policy, 'validation_protocol': protocol,
                       'dataset_fingerprint': verified['dataset_fingerprint'],
                       'selected_checkpoint_sha256': selection['selected_checkpoint_sha256'],
                       'candidate_checkpoint_sha256': summary['best_sha256'],
                       'baseline_checkpoint_sha256': original['best_sha256'],
                       'new_test_invocations': state['test_invocations'],
                       'reported_test_origin': 'new candidate development evaluation' if promoted else 'previous baseline evaluation; no new test',
                       'reported_test': test['overall'], 'per_class_validation_comparison': comparison}
    write(OUT / 'PACKAGE_SUMMARY.json', package_summary)

    plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    old_epochs = [int(row['epoch']) for row in original_rows]
    epochs = [int(row['epoch']) for row in rows]
    axes[0].plot(old_epochs, [row['metrics/mAP50-95(B)'] * 100 for row in original_rows], label='Original 50-epoch run')
    axes[0].axvline(48, color='gray', ls='--', label='Initialization: epoch 48 best')
    axes[0].set(title='Original phase: validation AP', xlabel='Original epoch', ylabel='mAP50-95 (%)')
    axes[1].plot(epochs, [row['metrics/mAP50-95(B)'] * 100 for row in rows], label='Fine-tuning epoch val')
    axes[1].axvline(summary['best_epoch'], color='gray', ls='--', label=f'Candidate best: epoch {summary["best_epoch"]}')
    axes[1].axhline(baseline['overall']['metrics/mAP50-95(B)'] * 100, color='#286f94', ls=':', label='Paired baseline val')
    axes[1].set(title='Fine-tuning phase: validation AP', xlabel='Fine-tuning epoch', ylabel='mAP50-95 (%)')
    for key, label in [('train/box_loss', 'train box'), ('val/box_loss', 'val box'), ('train/cls_loss', 'train class'), ('val/cls_loss', 'val class')]:
        axes[2].plot(epochs, [row[key] for row in rows], label=label)
    axes[2].set(title='Fine-tuning loss', xlabel='Fine-tuning epoch', ylabel='Loss')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'learning_curves.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(classes))
    ax.bar(x - .19, [row['baseline_ap50_95_pct'] for row in comparison], .38, label='Baseline: paired FP32 val', color='#286f94')
    ax.bar(x + .19, [row['candidate_ap50_95_pct'] for row in comparison], .38, label='Candidate: paired FP32 val', color='#df8a33')
    for i, row in enumerate(comparison):
        ax.text(i, min(103, max(row['baseline_ap50_95_pct'], row['candidate_ap50_95_pct']) + 2),
                f'{row["ap50_95_delta_pp"]:+.2f} pp', ha='center', fontsize=8)
    ax.set_xticks(x, classes, rotation=25, ha='right')
    ax.set(ylim=(0, 110), ylabel='AP50-95 (%)', title='Same validation 150 images: per-class comparison')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / 'validation_per_class_comparison.png', bbox_inches='tight')
    plt.close(fig)

    b50 = baseline['overall']['metrics/mAP50(B)'] * 100
    b95 = baseline['overall']['metrics/mAP50-95(B)'] * 100
    c50 = candidate['overall']['metrics/mAP50(B)'] * 100
    c95 = candidate['overall']['metrics/mAP50-95(B)'] * 100
    t50 = test['overall']['metrics/mAP50(B)'] * 100
    t95 = test['overall']['metrics/mAP50-95(B)'] * 100
    class_table = '\n'.join(
        f'| {row["class_name"]} | {row["val_boxes"]} | {row["val_source_groups"]} | {row["baseline_ap50_95_pct"]:.2f}% | {row["candidate_ap50_95_pct"]:.2f}% | {row["ap50_95_delta_pp"]:+.2f}%p |'
        for row in comparison)
    if promoted:
        decision = f'검증 승격 기준을 통과해 미세 조정 모델을 선택했습니다. 선택 후 개발용 test 200장을 이번 단계에서 1회 평가했고, mAP50 {t50:.2f}%, mAP50–95 {t95:.2f}%를 기록했습니다.'
        test_statement = '**새 후보 모델의 개발용 test 결과**입니다. 이번 단계에서 승격 이후 1회 평가했습니다. 이전 평가에 이미 사용한 200장이므로 독립적인 최종시험 성능으로 해석할 수 없습니다.'
        test_link = '[후보 test 기록](evidence/test-evaluation.json) · [1회 실행 표식](evidence/test-evaluation-attempt.json)'
    else:
        decision = f'검증 승격 기준을 통과하지 못해 기존 50에폭 실험의 best 모델을 유지했습니다. 이번 단계에서는 test를 실행하지 않았습니다. 유지한 기존 모델의 이전 test 결과는 mAP50 {t50:.2f}%, mAP50–95 {t95:.2f}%입니다.'
        test_statement = '**이전 baseline 실험에서 기록한 결과**입니다. 이번 미세 조정 후보의 test 점수가 아니며, 이번 단계의 새 test 실행은 0회입니다. 기존 결과를 그대로 보존했습니다.'
        test_link = '[이전 baseline test 기록](evidence/baseline50/test-evaluation.json) · [이전 1회 실행 표식](evidence/baseline50/test-evaluation-attempt.json)'
    review_images = '\n'.join(f'![검증 데이터 확인: {path.stem}](data_review/{path.name})' for path in sorted((OUT / 'data_review').glob('*')) if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    report = f'''# 보드 모델 — 50에폭 학습 후 최대 20에폭 미세 조정

**원래 50에폭 실험의 48에폭 best 가중치에서 새 optimizer로 실제 {summary['epochs_completed']}에폭 미세 조정했습니다. {decision}**

## 데이터와 비교 조건

| 항목 | 값 |
|---|---|
| 총 데이터 / 분할 | 1,000장 / train 650 · val 150 · test 200 |
| 학습 클래스 | Raspberry Pi 5/4/other, STM32 other, Arduino, ESP, Jetson, other board — 8개 |
| 데이터 변경 | 없음; 기존 분할·원본/유사도 그룹·라벨·클래스 순서 유지 |
| 기존 학습 | 실제 50에폭, 검증으로 선택한 best는 epoch 48 |
| 미세 조정 | 최대 20 / 실제 {summary['epochs_completed']}에폭; 후보 best는 이 단계의 epoch {summary['best_epoch']} |
| 비교 대상 | 기존 best와 새 후보 best를 같은 standalone FP32 val 조건으로 재평가 |
| 비교 역할 | val 150장만 모델 선택에 사용; test는 승격한 뒤에만 조건부 실행 |

Nucleo와 포트는 이번 모델에 포함하지 않았습니다. test 200장은 이전 test 106장, 이전 val 86장, 이전 train 8장을 묶은 개발용 평가셋입니다. 현재 train/val/test에서 같은 연결 그룹은 한 분할에만 있습니다. 연결 그룹이 실제 보드 개체·촬영 세션의 독립성을 입증하지는 않습니다.

## 변경한 설정과 실제 실행

| 항목 | 설정 / 기록 |
|---|---|
| 초기 가중치 | 기존 48에폭 best; `resume=False`, 새 AdamW optimizer |
| 입력 / 배치 / 계산 | 640 × 640 / 8 / FP32 (`amp=False`) |
| 미세 조정 학습률 | `lr0=0.0001`, cosine decay, `lrf=0.1` |
| Warmup | 1에폭, `warmup_bias_lr=0` |
| Mosaic | 0; 원래 학습의 마지막 mosaic 종료 단계에서 이어지는 설정 |
| 기타 증강 | 회전 10°, scale 0.5, translate 0.1, 좌우 반전 0.5, HSV 유지 |
| 조기 종료 | 검증 fitness가 8에폭 동안 개선되지 않을 때 |
| Seed | {summary['config']['seed']} |
| 실제 optimizer 갱신 | {summary['actual_optimizer_steps']}회 |
| 미세 조정 시간 | {summary['training_seconds'] / 60:.1f}분 |
| 실행 환경 | {state['gpu']}; PyTorch {state['torch']}; Ultralytics {state['ultralytics']} |
| 최대 PyTorch 할당 메모리 | {summary['peak_cuda_memory_gib']:.2f} GiB |

원래 50에폭 실행과 추가 미세 조정 단계를 구분해 기록했습니다. 새 optimizer·학습률 일정과 mosaic 설정이 적용돼 에폭 증가만의 효과를 분리한 실험이 아닙니다. 같은 val을 반복 사용한 개발 과정이므로 개선 수치를 새 데이터 일반화의 확정 증거로 해석하지 않습니다.

![단계별 학습 곡선](learning_curves.png)

곡선은 각 학습 단계의 epoch별 validation 기록입니다. 모델 승격은 아래의 **동일한 standalone 평가 조건으로 얻은 두 모델의 결과**를 사용했습니다. 두 단계의 epoch 축을 분리했으며, 미세 조정 초기값은 원래 best epoch 48입니다.

## 동일한 검증 조건의 비교와 승격 결정

검증 조건: `split=val`, `imgsz=640`, `batch=8`, `conf=0.001`, `iou=0.7`, `max_det=300`, `half=False`, GPU 0. 선택 전에 고정한 기준은 **전체 mAP50–95가 최소 +0.5%p 개선**되고 **어느 클래스도 AP50–95가 5%p 넘게 하락하지 않는 것**입니다. 소규모 검증셋에 적용한 실무 기준이며 통계적 유의성을 의미하지 않습니다.

| 지표 | 기존 best | 미세 조정 후보 | 차이 |
|---|---:|---:|---:|
| val mAP50 | {b50:.2f}% | {c50:.2f}% | {c50 - b50:+.2f}%p |
| val mAP50–95 | {b95:.2f}% | {c95:.2f}% | {gain * 100:+.2f}%p |
| 최악의 클래스 AP50–95 차이 | — | — | {worst * 100:+.2f}%p |
| 후보 승격 | — | {'통과' if promoted else '미통과'} | 전체·클래스 기준을 함께 적용 |

| 클래스 | val 정답 박스 | val 연결 그룹 | 기존 AP50–95 | 후보 AP50–95 | 차이 |
|---|---:|---:|---:|---:|---:|
{class_table}

![클래스별 검증 비교](validation_per_class_comparison.png)

Jetson 학습 데이터는 67장·77박스, TX2 계열 3개 연결 그룹에 한정되고 Nano 그룹이 없습니다. val Jetson도 2개 그룹, TX2 14장과 Nano 2장뿐입니다. 초기 자료 선택이 이미지 수 할당량에서 멈추면서 큰 TX2 증강 묶음이 Jetson 할당량을 많이 차지했고, 선택된 Nano 원본 파일명 계열 5개는 모두 val/test로 배정됐습니다. 에폭을 늘려도 빠진 보드 종류·배경·촬영 조건이 추가되지는 않습니다.

읽기 전용 조사에서 현재 데이터에 쓰이지 않은 Nano 후보 90장·58개 원본 파일명 계열을 찾았습니다. 현재 1,000장 및 Commons 44장과 설정된 pHash 임계값(분할 6 / holdout 8)으로 비교하고 후보끼리 연결을 합쳐 57개 후보 연결 그룹을 기록했으며, 원자료 SHA-256도 대조했습니다. 이 후보는 이번 모델에 사용하지 않았고 실제 보드·촬영 장면 독립성과 라벨 적합성은 아직 확인되지 않았습니다. 다음 데이터 버전은 후보의 사진·라벨을 검수한 뒤 Nano 학습용 원본 계열을 먼저 확보하고 TX2/Nano 하위 종류별 균형을 맞춰야 합니다. [자료 검토](evidence/data_review.json)와 [Nano 후보 인벤토리](data_review/nano_candidate_inventory.json)에 근거를 보존했습니다. D455 실촬영 성능은 아직 검증하지 않았습니다.

## Test 기록의 범위

{test_statement}

| 기록 | mAP50 | mAP50–95 | 이번 단계 test 실행 |
|---|---:|---:|---:|
| {'승격한 후보의 새 개발 평가' if promoted else '유지한 기존 모델의 이전 평가'} | {t50:.2f}% | {t95:.2f}% | {state['test_invocations']}회 |

{test_link}

## 가중치·설정·근거 파일

- [현재 선택 모델](selected-best.pt): SHA-256 `{selection['selected_checkpoint_sha256']}`
- [미세 조정 후보 best](candidate-best.pt): SHA-256 `{summary['best_sha256']}`
- [원래 baseline best](baseline-best.pt): SHA-256 `{original['best_sha256']}`
- [승격 결정 원자료](evidence/selection.json) · [설정과 사전 기준](evidence/experiment_plan.json)
- [baseline 동일 조건 검증](evidence/baseline_validation.json) · [candidate 동일 조건 검증](evidence/candidate_validation.json)
- [미세 조정 실행 기록](evidence/training-summary.json) · [실제 epoch CSV](evidence/results.csv) · [적용 설정](evidence/args.yaml)
- [클래스별 검증 비교 CSV](validation_per_class_comparison.csv) · [분할별 라벨·그룹 수](split_class_counts.csv)
- [데이터 검증](evidence/dataset/verification.json) · [이미지·라벨 인덱스](evidence/dataset/dataset_index.json) · [분할 근거](evidence/dataset/derivation.json)
- [학습 코드](code/run_boards1000_finetune.py) · [보고서 생성 코드](code/package_boards1000_finetune.py)
- [기계 판독 요약](PACKAGE_SUMMARY.json) · [패키지 파일 SHA-256](ARTIFACT_MANIFEST.json)

데이터 fingerprint: `{verified['dataset_fingerprint']}`. 이미지 원본은 기존 1,000장 데이터셋과 GitHub 복원 ZIP을 유지하며 이 보고서 패키지에는 중복 복사하지 않았습니다. 코드의 로컬 경로는 다른 PC에서 복원한 경로에 맞춰 조정해야 합니다. `evidence/baseline_environment`는 이전 아카이브의 환경 기록이며, 이번 실행의 라이브 버전은 위 실행 기록과 `execution_state.json`에 있습니다.

```python
from ultralytics import YOLO
model = YOLO("selected-best.pt")
model.predict(source="board_photo.jpg", imgsz=640, conf=0.25, save=True)
```

`conf=0.25`는 실행 예시이며 test로 최적화한 운영 임계값이 아닙니다.

## 검증 데이터 확인 자료

아래 자료는 기존 val 사진과 정답 박스의 육안 확인용이며, 파일명은 확인 당시의 작업명입니다.

{review_images or '별도 육안 확인 이미지는 이 패키지에 포함되지 않았습니다.'}
'''
    (OUT / 'README.md').write_text(report, encoding='utf-8')
    assert sha(OUT / 'selected-best.pt') == selection['selected_checkpoint_sha256']
    assert sha(OUT / 'candidate-best.pt') == summary['best_sha256']
    assert sha(OUT / 'baseline-best.pt') == original['best_sha256']
    write(OUT / 'ARTIFACT_MANIFEST.json', {'scope': 'All packaged files except this manifest',
          'files': [{'path': path.relative_to(OUT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
                    for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'ARTIFACT_MANIFEST.json']})
    print(json.dumps({'output': str(OUT), 'promoted': promoted, 'fine_tuning_epochs': summary['epochs_completed'],
                      'validation_ap50_95_gain_pp': gain * 100, 'new_test_invocations': state['test_invocations'],
                      'selected_checkpoint_sha256': selection['selected_checkpoint_sha256']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
