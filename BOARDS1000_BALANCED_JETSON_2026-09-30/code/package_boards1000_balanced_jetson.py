"""Package only a completed balanced Jetson experiment; no model inference."""
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
ORIGINAL = HERE / 'work/boards1000_s650_v150_t200_e50_20260929'
ORIGINAL_RUN = ORIGINAL / 'runs/boards1000_yolo11s_50ep'
FINE = HERE / 'work/boards1000_finetune20_20260930'
FINE_RUN = FINE / 'runs/boards1000_finetune20'
TRIAL = HERE / 'work/boards1000_balanced_jetson_20260930'
RUN = TRIAL / 'runs/boards1000_balanced_jetson30'
DATA = TRIAL / 'datasets/boards_v1'
OUT = HERE / 'outputs/BOARDS1000_BALANCED_JETSON_2026-09-30'
EXPECTED_FINGERPRINT = '3120af5994f7494f608b31b59b48833eebee8423408f5a0190f516f6e708ba6f'
EXPECTED_MAP_SHA = '7c6ed00977a900b18ed6efa95dab86c853680fdf716f434d1d3ad1c348a34a6c'
ORIGINAL_SHA = '25a6253ce7520d50d362fdb276e61d25b2f77ec78583d8706cd463f74282b766'
FINE_SHA = 'fa0b8fb5fc11de748e9d8051b93fc57dde5c6cc34db6167045e3fee9e4912752'


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


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(source) == sha(target), f'Copy checksum mismatch: {target}'


def curve(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def csv_write(path, rows):
    assert rows
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_inventory(dataset):
    index = read(dataset / 'dataset_index.json')
    records = sorted((r for r in index['records'] if r['split'] in {'val', 'test'}), key=lambda r: (r['split'], r['id']))
    assert len(records) == 350
    result = []
    for r in records:
        assert sha(dataset / r['image']) == r['image_sha256']
        assert sha(dataset / r['label']) == r['label_sha256']
        result.append({key: r[key] for key in ['id', 'split', 'image', 'label', 'image_sha256', 'label_sha256', 'width', 'height']})
    return {'classes': index['classes'], 'records': result}


def main():
    state = read(TRIAL / 'execution_state.json')
    assert state['status'] == 'complete', 'Balanced training must complete before packaging'
    assert not OUT.exists(), 'Existing output preserved; choose a separately reviewed new package'
    plan = read(TRIAL / 'experiment_plan.json')
    summary = read(TRIAL / 'training-summary.json')
    selection = read(TRIAL / 'selection.json')
    baseline = read(TRIAL / 'baseline_validation.json')
    previous = read(TRIAL / 'current_selected_validation.json')
    candidate = read(TRIAL / 'candidate_validation.json')
    verified = read(TRIAL / 'datasets/verification.json')['boards_v1']
    derivation = read(TRIAL / 'derivation.json')
    original_summary = read(ORIGINAL_RUN / 'training-summary.json')
    fine_summary = read(FINE / 'training-summary.json')
    original_test = read(ORIGINAL_RUN / 'test-evaluation.json')
    fine_test = read(FINE / 'test-evaluation.json')
    rows = curve(RUN / 'results.csv')
    classes = plan['classes']
    original_path = Path(plan['initial_checkpoint'])
    previous_path = Path(plan['current_selected_checkpoint'])
    candidate_path = Path(summary['best'])
    selected_path = Path(selection['selected_checkpoint'])
    policy = plan['promotion_policy']
    protocol = plan['validation_protocol']
    promoted = selection['promoted']

    assert len(classes) == 8 and classes == original_summary['classes']
    assert verified['training_ready'] and verified['structure_pass']
    assert verified['images'] == {'train': 650, 'val': 150, 'test': 200}
    assert verified['boxes']['train']['jetson'] == 74
    assert verified['dataset_fingerprint'] == plan['dataset_fingerprint'] == state['dataset_fingerprint'] == EXPECTED_FINGERPRINT
    assert verified['class_map_sha256'] == EXPECTED_MAP_SHA == sha(TRIAL / 'class_map.json')
    assert 0 < len(rows) == summary['epochs_completed'] == len(state['epochs']) <= 30
    assert [int(row['epoch']) for row in rows] == list(range(1, len(rows) + 1))
    assert summary['actual_optimizer_steps'] == state['optimizer_steps'] > 0
    assert all(math.isfinite(value) for row in rows for value in row.values())
    assert summary['config'] == plan['config'] == state['config']
    assert plan['config']['epochs'] == 30 and plan['config']['lr0'] == .0003
    assert plan['config']['warmup_bias_lr'] == 0 and plan['config']['mosaic'] == .5
    assert plan['config']['amp'] is False and plan['config']['resume'] is False
    assert sha(HERE / 'work/run_boards1000_balanced_jetson.py') == plan['script_sha256']
    assert sha(original_path) == ORIGINAL_SHA == original_summary['best_sha256'] == plan['initial_checkpoint_sha256']
    assert sha(previous_path) == FINE_SHA == fine_summary['best_sha256'] == plan['current_selected_checkpoint_sha256']
    assert read(FINE / 'selection.json')['selected_checkpoint_sha256'] == FINE_SHA
    assert baseline['checkpoint_sha256'] == selection['baseline_checkpoint_sha256'] == ORIGINAL_SHA
    assert previous['checkpoint_sha256'] == selection['current_selected_checkpoint_sha256'] == FINE_SHA
    assert sha(TRIAL / 'baseline_validation.json') == plan['baseline_validation_source_sha256']
    assert sha(TRIAL / 'current_selected_validation.json') == plan['current_selected_validation_source_sha256']
    assert sha(candidate_path) == summary['best_sha256'] == candidate['checkpoint_sha256'] == selection['candidate_checkpoint_sha256']
    assert sha(selected_path) == state['selected_checkpoint_sha256'] == selection['selected_checkpoint_sha256']
    assert protocol == baseline['protocol'] == previous['protocol'] == candidate['protocol']
    assert protocol['half'] is False and protocol['imgsz'] == 640 and protocol['batch'] == 8
    assert policy['minimum_overall_ap50_95_gain'] == .005 and policy['maximum_per_class_ap50_95_drop'] == .05
    assert policy['minimum_jetson_ap50_95_gain'] == policy['minimum_overall_gain_vs_current_selected'] == 0
    assert original_test['checkpoint_sha256'] == ORIGINAL_SHA and fine_test['checkpoint_sha256'] == FINE_SHA
    assert original_test['invocations'] == fine_test['invocations'] == 1
    inventory = evaluation_inventory(DATA)
    assert inventory == plan['evaluation_inventory'] == evaluation_inventory(ORIGINAL / 'datasets/boards_v1')
    sys.path.insert(0, str(TRIAL))
    from verify_datasets import dataset_fingerprint, holdout_inventory
    assert dataset_fingerprint(DATA) == EXPECTED_FINGERPRINT
    assert holdout_inventory() == verified['holdout_inventory']

    b = {row['class_name']: row for row in baseline['per_class']}
    p = {row['class_name']: row for row in previous['per_class']}
    c = {row['class_name']: row for row in candidate['per_class']}
    assert set(b) == set(p) == set(c) == set(classes)
    comparison = []
    for name in classes:
        assert b[name]['ground_truth_boxes'] == p[name]['ground_truth_boxes'] == c[name]['ground_truth_boxes'] == verified['boxes']['val'][name]
        comparison.append({'class_name': name, 'val_boxes': c[name]['ground_truth_boxes'],
                           'original_ap50_95_pct': b[name]['map50_95'] * 100,
                           'previous_fine_ap50_95_pct': p[name]['map50_95'] * 100,
                           'balanced_ap50_95_pct': c[name]['map50_95'] * 100,
                           'delta_vs_original_pp': (c[name]['map50_95'] - b[name]['map50_95']) * 100,
                           'delta_vs_previous_pp': (c[name]['map50_95'] - p[name]['map50_95']) * 100})
    metric = 'metrics/mAP50-95(B)'
    gain = candidate['overall'][metric] - baseline['overall'][metric]
    current_gain = candidate['overall'][metric] - previous['overall'][metric]
    worst = min(row['delta_vs_original_pp'] / 100 for row in comparison)
    jetson_gain = c['jetson']['map50_95'] - b['jetson']['map50_95']
    assert abs(gain - selection['overall_ap_gain']) < 1e-12
    assert abs(current_gain - selection['overall_ap_gain_vs_current_selected']) < 1e-12
    assert all(abs(selection['per_class_ap_gains'][row['class_name']] - row['delta_vs_original_pp'] / 100) < 1e-12 for row in comparison)
    assert promoted == (gain >= .005 and worst >= -.05 and jetson_gain >= 0 and current_gain >= 0)
    assert promoted == state['promoted']
    assert selected_path.resolve() == (candidate_path if promoted else previous_path).resolve()
    if promoted:
        test = read(TRIAL / 'test-evaluation.json')
        marker = read(TRIAL / 'test-evaluation-attempt.json')
        assert test['invocations'] == state['test_invocations'] == 1 and test['images'] == 200
        assert test['checkpoint_sha256'] == marker['checkpoint_sha256'] == summary['best_sha256']
    else:
        assert state['test_invocations'] == 0 and not (TRIAL / 'test-evaluation.json').exists()
        assert not (TRIAL / 'test-evaluation-attempt.json').exists()
        test = fine_test

    OUT.mkdir(parents=True)
    for source, name in [(selected_path, 'selected-best.pt'), (candidate_path, 'candidate-best.pt'),
                         (RUN / 'weights/last.pt', 'candidate-last.pt'), (original_path, 'baseline-best.pt'),
                         (previous_path, 'previous-selected-best.pt')]:
        copy(source, OUT / name)
    for name in ['execution_state.json', 'experiment_plan.json', 'training-summary.json', 'selection.json',
                 'baseline_validation.json', 'current_selected_validation.json', 'candidate_validation.json']:
        copy(TRIAL / name, OUT / 'evidence' / name)
    if (TRIAL / 'outcome_audit.json').exists():
        copy(TRIAL / 'outcome_audit.json', OUT / 'evidence/outcome_audit.json')
    for name in ['results.csv', 'args.yaml']:
        copy(RUN / name, OUT / 'evidence' / name)
    if promoted:
        for name in ['test-evaluation.json', 'test-evaluation-attempt.json']:
            copy(TRIAL / name, OUT / 'evidence' / name)
    for source, label in [(ORIGINAL_RUN, 'original50'), (FINE, 'previous_finetune')]:
        for name in ['training-summary.json', 'test-evaluation.json', 'test-evaluation-attempt.json']:
            copy(source / name, OUT / 'evidence' / label / name)
    for name in ['selection.json', 'experiment_plan.json', 'candidate_validation.json']:
        copy(FINE / name, OUT / 'evidence/previous_finetune' / name)
    for source, label in [(ORIGINAL_RUN, 'original50'), (FINE_RUN, 'previous_finetune')]:
        for name in ['results.csv', 'args.yaml']:
            copy(source / name, OUT / 'evidence' / label / name)
    for name in ['derivation.json', 'class_map.json', 'data_correction_verification.json', 'visual_review.json',
                 'source_annotations.json', 'selection_proposal.json', 'candidate_exclusions.json']:
        copy(TRIAL / name, OUT / 'evidence/dataset' / name)
    copy(TRIAL / 'datasets/verification.json', OUT / 'evidence/dataset/verification.json')
    for name in ['dataset_index.json', 'data.yaml']:
        copy(DATA / name, OUT / 'evidence/dataset' / name)
    for name in ['run_boards1000_balanced_jetson.py', 'package_boards1000_balanced_jetson.py', 'prepare_boards1000_balanced_jetson.py',
                 'run_boards1000.py', 'run_boards1000_finetune.py']:
        copy(HERE / 'work' / name, OUT / 'code' / name)
    for name in ['common.py', 'assemble.py', 'verify_datasets.py']:
        copy(TRIAL / name, OUT / 'code' / name)
    for path in sorted(TRIAL.glob('nano*gt_sheet*.jpg')):
        copy(path, OUT / 'data_review' / path.name)
    inventory_path = HERE / 'work/next_training_review/nano_candidate_inventory.json'
    if inventory_path.exists():
        copy(inventory_path, OUT / 'data_review/nano_candidate_inventory.json')
    for relative, label in [('candidate_val', 'candidate_val'), ('runs/boards1000_balanced_jetson30', 'training'), ('test', 'test')]:
        for path in sorted((TRIAL / relative).glob('*')):
            if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                copy(path, OUT / 'plots' / label / path.name)
    for path in sorted((FINE / 'baseline_val').glob('*.png')):
        copy(path, OUT / 'plots/original_baseline_val' / path.name)
    old_env = HERE / 'work/github_boards1000_20260930/BOARDS1000_2026-09-30/environment'
    for name in ['runtime.json', 'requirements-lock.txt']:
        if (old_env / name).exists():
            copy(old_env / name, OUT / 'evidence/baseline_environment' / name)
    csv_write(OUT / 'validation_three_experiment_comparison.csv', comparison)
    csv_write(OUT / ('balanced_test_per_class.csv' if promoted else 'previous_fine_test_per_class.csv'), test['per_class'])
    csv_write(OUT / 'split_class_counts.csv', [{'class_name': name, **{f'{split}_boxes': verified['boxes'][split][name] for split in ['train', 'val', 'test']}} for name in classes])
    package_summary = {'promoted': promoted, 'phase': 'train-only Jetson balance correction from original50best',
                       'original_epochs': original_summary['epochs_completed'], 'previous_fine_epochs': fine_summary['epochs_completed'],
                       'balanced_phase_epochs': summary['epochs_completed'], 'balanced_best_epoch': summary['best_epoch'],
                       'baseline_validation': baseline['overall'], 'previous_selected_validation': previous['overall'], 'candidate_validation': candidate['overall'],
                       'overall_gain_vs_original_pp': gain * 100, 'overall_gain_vs_previous_selected_pp': current_gain * 100,
                       'worst_class_gain_vs_original_pp': worst * 100, 'jetson_gain_vs_original_pp': jetson_gain * 100,
                       'promotion_policy': policy, 'validation_protocol': protocol, 'dataset_fingerprint': EXPECTED_FINGERPRINT,
                       'selected_checkpoint_sha256': selection['selected_checkpoint_sha256'], 'candidate_checkpoint_sha256': summary['best_sha256'],
                       'initializer_and_comparison_baseline_sha256': ORIGINAL_SHA, 'previous_selected_checkpoint_sha256': FINE_SHA,
                       'new_test_invocations': state['test_invocations'], 'reported_test_origin': 'new balanced candidate development evaluation' if promoted else 'previous fine-tuning historical test; no balanced candidate test',
                       'reported_test': test['overall'], 'original_historical_test': original_test['overall'], 'previous_fine_historical_test': fine_test['overall']}
    write(OUT / 'PACKAGE_SUMMARY.json', package_summary)

    plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})
    epochs = [int(row['epoch']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in [('metrics/mAP50(B)', 'mAP50'), ('metrics/mAP50-95(B)', 'mAP50-95')]:
        axes[0].plot(epochs, [row[key] * 100 for row in rows], label=label)
    axes[0].axhline(baseline['overall'][metric] * 100, ls=':', color='#286f94', label='Original best paired val')
    axes[0].axhline(previous['overall'][metric] * 100, ls='--', color='#b98520', label='Previous selected paired val')
    axes[0].axvline(summary['best_epoch'], color='gray', ls='--', alpha=.6, label=f'Candidate best: {summary["best_epoch"]}')
    axes[0].set(title='Balanced phase: validation 150 images', xlabel='Balanced training epoch', ylabel='AP (%)')
    for key, label in [('train/box_loss', 'train box'), ('val/box_loss', 'val box'), ('train/cls_loss', 'train class'), ('val/cls_loss', 'val class')]:
        axes[1].plot(epochs, [row[key] for row in rows], label=label)
    axes[1].set(title='Balanced phase: training and validation loss', xlabel='Balanced training epoch', ylabel='Loss')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'learning_curves.png', bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(classes))
    for offset, key, label, color in [(-.25, 'original_ap50_95_pct', 'Original 50-epoch best', '#286f94'),
                                    (0, 'previous_fine_ap50_95_pct', 'Previous selected fine-tune', '#b98520'),
                                    (.25, 'balanced_ap50_95_pct', 'Balanced candidate', '#439472')]:
        ax.bar(x + offset, [row[key] for row in comparison], .25, label=label, color=color)
    for i, row in enumerate(comparison):
        ax.text(i, max(row[key] for key in ['original_ap50_95_pct', 'previous_fine_ap50_95_pct', 'balanced_ap50_95_pct']) + 2,
                f'{row["delta_vs_original_pp"]:+.2f} pp', ha='center', fontsize=8)
    ax.set_xticks(x, classes, rotation=25, ha='right')
    ax.set(ylim=(0, 110), ylabel='AP50-95 (%)', title='Same FP32 validation protocol; annotations show delta vs original')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / 'validation_three_experiment_comparison.png', bbox_inches='tight')
    plt.close(fig)

    table = '\n'.join(f'| {r["class_name"]} | {r["val_boxes"]} | {r["original_ap50_95_pct"]:.2f}% | {r["previous_fine_ap50_95_pct"]:.2f}% | {r["balanced_ap50_95_pct"]:.2f}% | {r["delta_vs_original_pp"]:+.2f}%p |' for r in comparison)
    score = lambda value, key: value['overall'][key] * 100
    metrics_table = '\n'.join(f'| {label} | {score(baseline, key):.2f}% | {score(previous, key):.2f}% | {score(candidate, key):.2f}% |' for label, key in [('val mAP50', 'metrics/mAP50(B)'), ('val mAP50–95', metric)])
    gates = [('전체 AP: 원래 best보다 +0.5%p 이상', gain >= .005, f'{gain * 100:+.2f}%p'),
             ('모든 클래스: 원래 best보다 하락 5%p 이하', worst >= -.05, f'최악 {worst * 100:+.2f}%p'),
             ('Jetson AP: 원래 best 이상', jetson_gain >= 0, f'{jetson_gain * 100:+.2f}%p'),
             ('전체 AP: 이전 선택 모델 이상', current_gain >= 0, f'{current_gain * 100:+.2f}%p')]
    gate_table = '\n'.join(f'| {label} | {"통과" if passed else "미통과"} | {value} |' for label, passed, value in gates)
    t50, t95 = score(test, 'metrics/mAP50(B)'), score(test, metric)
    if promoted:
        decision = f'검증의 사전 승격 기준 네 가지를 모두 통과해 Jetson 균형 보정 모델을 선택했습니다. 승격 후 test 200장을 이번 단계에서 1회 평가했고, mAP50 {t50:.2f}%, mAP50–95 {t95:.2f}%를 기록했습니다.'
        new_test_text = f'새 균형 보정 후보의 개발용 test: **mAP50 {t50:.2f}% / mAP50–95 {t95:.2f}%**. [새 test 기록](evidence/test-evaluation.json), 이번 단계 실행 1회.'
    else:
        decision = '검증의 사전 승격 기준을 모두 통과하지 못해 이전 미세 조정 모델을 유지했습니다. 이번 균형 보정 후보의 test는 실행하지 않았습니다.'
        new_test_text = '균형 보정 후보의 새 test 실행: **0회**. 아래 두 점수는 각 이전 모델의 역사적 평가이며 이번 후보의 성능이 아닙니다.'
    jetson_test_ap = lambda result: next(row['map50_95'] * 100 for row in result['per_class'] if row['class_name'] == 'jetson')
    history_table = '\n'.join(f'| {label} | {score(result, "metrics/mAP50(B)"):.2f}% | {score(result, metric):.2f}% | {jetson_test_ap(result):.2f}% | 이전 실행 1회 |' for label, result in [('원래 50에폭 best', original_test), ('이전 미세 조정 선택 모델', fine_test)])
    scheduled_mosaic_close_epoch = plan['config']['epochs'] - plan['config']['close_mosaic'] + 1
    mosaic_outcome = (f"예정된 mosaic 종료는 이번 단계 epoch {scheduled_mosaic_close_epoch} 시작 시점입니다. 실제 {summary['epochs_completed']}에폭에서 조기 종료해 **mosaic를 끈 마무리 단계에 도달하지 못했습니다.**"
                      if summary['epochs_completed'] < scheduled_mosaic_close_epoch else
                      f"예정된 mosaic 종료는 이번 단계 epoch {scheduled_mosaic_close_epoch} 시작 시점이며, 실제 실행에서 종료 단계에 도달했습니다.")
    sheets = '\n'.join(f'![Nano 정답 검토: {path.stem}](data_review/{path.name})' for path in sorted((OUT / 'data_review').glob('nano_native_gt_sheet*.jpg')))
    outcome_audit_link = '\n- [결과·선택 규칙 감사](evidence/outcome_audit.json)' if (OUT / 'evidence/outcome_audit.json').exists() else ''
    report = f'''# 보드 검출 — Jetson 학습 데이터 균형 보정

**총 1,000장과 train 650 / val 150 / test 200 분할을 유지하고, 학습셋의 반복 TX2 사진 40장을 검수한 Nano 사진 40장으로 교체했습니다. 원래 50에폭 실험의 48에폭 best에서 실제 {summary['epochs_completed']}에폭 학습했습니다. {decision}**

## 변경한 데이터와 비교 범위

Jetson 학습 사진은 TX2 27장 + Nano 40장 = 67장, 정답 박스 74개입니다. 원래 학습셋은 TX2 67장·77박스와 3개 연결 그룹에 한정돼 Nano 학습 사진이 없었습니다. Nano 후보는 기존 train/val/test 및 Commons holdout과 pHash로 검사하고 원본 파일명 계열을 고려했으며, 라벨이 불완전한 장면을 제외한 뒤 최종 정답 박스 사진을 검토했습니다. 후보의 실제 보드 개체·촬영 장면 독립성까지 확인한 것은 아닙니다.

**val/test 350장의 ID, 이미지 바이트와 라벨 바이트는 원래 데이터와 동일합니다.** 다른 클래스와 8개 클래스 순서를 유지했습니다. Nucleo와 포트는 이번 모델 범위에 포함하지 않았습니다. 이번 단계는 새 데이터·optimizer·학습률 일정·mosaic를 적용했으므로 데이터 교체나 에폭 증가 한 가지의 효과를 분리한 실험이 아닙니다.

## 학습 단계·설정

| 항목 | 값 |
|---|---|
| 전체 / 분할 | 1,000장 / train 650 · val 150 · test 200 |
| 원래 학습 | 실제 50에폭, 선택 best epoch 48 |
| 이전 미세 조정 | 실제 {fine_summary['epochs_completed']}에폭, best epoch {fine_summary['best_epoch']}; 별도 이력으로 보존 |
| 이번 초기값 | 원래 50에폭 best, 이전 미세 조정에서 이어 학습하지 않음 |
| 이번 단계 | 최대 30 / 실제 {summary['epochs_completed']}에폭; 후보 best epoch {summary['best_epoch']} |
| 입력 / 배치 / 계산 | 640 × 640 / 8 / FP32 |
| 최적화 | 새 AdamW, lr 0.0003, cosine, lrf 0.1 |
| Warmup | 3에폭, bias lr 0 |
| 증강 | mosaic 0.5, epoch {scheduled_mosaic_close_epoch} 시작 시 종료 예정; 회전 10°, scale 0.5, translate 0.1, HSV·좌우 반전 유지 |
| 조기 종료 / seed | patience 10 / {plan['config']['seed']} |
| 실제 optimizer 갱신 | {summary['actual_optimizer_steps']}회 |
| 학습 시간 / 최대 할당 메모리 | {summary['training_seconds'] / 60:.1f}분 / {summary['peak_cuda_memory_gib']:.2f} GiB |
| 환경 | {state['gpu']}; PyTorch {state['torch']}; Ultralytics {state['ultralytics']} |

{mosaic_outcome}

![학습 곡선](learning_curves.png)

## 같은 검증 조건에서의 선택

프로토콜은 val 150장, FP32, `imgsz=640`, `batch=8`, `conf=0.001`, `iou=0.7`, `max_det=300`, GPU 0입니다. 검증 사진·라벨 바이트와 평가 프로토콜이 같음을 확인해 원래 best와 이전 미세 조정 모델의 기존 standalone validation JSON을 재사용했습니다. baseline 재추론은 하지 않았습니다. 후보는 같은 조건으로 평가했습니다.

학습 전에 네 가지 승격 기준을 고정했습니다. 원래 best는 초기값 및 개선 비교 기준이며, 이전 미세 조정 선택 모델은 전체 AP 하락 방지 기준과 승격 실패 시 유지하는 모델입니다. 모든 선택 기준은 validation만 사용합니다.

| 지표 | 원래 50에폭 best | 이전 선택 모델 | 이번 균형 보정 후보 |
|---|---:|---:|---:|
{metrics_table}

| 사전 승격 기준 | 판정 | 관측값 |
|---|---|---:|
{gate_table}

| 클래스 | val 정답 박스 | 원래 AP50–95 | 이전 선택 AP50–95 | 이번 후보 AP50–95 | 원래 대비 |
|---|---:|---:|---:|---:|---:|
{table}

![세 실험의 검증 비교](validation_three_experiment_comparison.png)

승격 기준은 작은 재사용 val에 적용한 실무 규칙이며 통계적 유의성을 뜻하지 않습니다. Jetson val은 TX2 14장 + Nano 2장, 2개 연결 그룹뿐입니다. 재사용 검증셋에 반복 적응할 가능성이 있어, 향후 새 보드·배경·조명으로 고정한 독립 평가가 필요합니다. D455 실촬영 성능은 아직 검증하지 않았습니다.

## Test 기록

{new_test_text}

| 이전 모델의 역사적 개발 평가 | mAP50 | mAP50–95 | Jetson AP50–95 | 기록 |
|---|---:|---:|---:|---|
{history_table}

이전 미세 조정 모델은 전체 개발 test mAP50–95가 소폭 높지만, Jetson AP50–95는 원래 모델의 {jetson_test_ap(original_test):.2f}%에서 {jetson_test_ap(fine_test):.2f}%로 {jetson_test_ap(original_test) - jetson_test_ap(fine_test):.2f}%p 낮아졌습니다. 이 한계를 공개해 보존하며, 현재 모델 유지 결정은 사전에 정한 validation 승격 규칙을 따른 것입니다.

[원래 test 기록](evidence/original50/test-evaluation.json) · [이전 미세 조정 test 기록](evidence/previous_finetune/test-evaluation.json). test 200장은 이전 test 106장·이전 val 86장·이전 train 8장에서 만든 재사용 개발 평가셋입니다. 새 최종시험이 아니며, test 점수는 이번 후보의 승격 여부에 사용하지 않았습니다.

## 파일과 복원 근거

- [현재 선택 모델](selected-best.pt): `{selection['selected_checkpoint_sha256']}`
- [이번 후보 best](candidate-best.pt) · [이번 후보 last](candidate-last.pt)
- [원래 초기값·비교 baseline](baseline-best.pt): `{ORIGINAL_SHA}`
- [이전 선택 모델](previous-selected-best.pt): `{FINE_SHA}`
- [설정과 사전 승격 정책](evidence/experiment_plan.json) · [승격 결정](evidence/selection.json)
- [실제 실행 기록](evidence/training-summary.json) · [epoch CSV](evidence/results.csv) · [적용 설정](evidence/args.yaml)
- [기존 baseline 검증](evidence/baseline_validation.json) · [이전 선택 검증](evidence/current_selected_validation.json) · [후보 검증](evidence/candidate_validation.json)
- [클래스 비교 CSV](validation_three_experiment_comparison.csv) · [분할 라벨 수](split_class_counts.csv)
- [데이터 검증](evidence/dataset/verification.json) · [교체 검증](evidence/dataset/data_correction_verification.json) · [라벨·사진 검토](evidence/dataset/visual_review.json)
- [새 데이터 인덱스](evidence/dataset/dataset_index.json) · [분할·교체 근거](evidence/dataset/derivation.json) · [원본 라벨](evidence/dataset/source_annotations.json)
- [학습 코드](code/run_boards1000_balanced_jetson.py) · [데이터 준비 코드](code/prepare_boards1000_balanced_jetson.py)
- [기계 판독 요약](PACKAGE_SUMMARY.json) · [패키지 SHA-256 목록](ARTIFACT_MANIFEST.json){outcome_audit_link}

새 데이터 fingerprint: `{EXPECTED_FINGERPRINT}`. 모델 파일은 SHA-256으로 연결해 기록했습니다. 패키지에는 데이터 원본을 중복 복사하지 않았고 기존 학습 폴더에 정확한 1,000장과 라벨이 있습니다. 코드와 data.yaml에는 로컬 경로가 있으므로 다른 PC에서는 복원한 경로로 조정해야 합니다. 이전 아카이브 환경 기록은 `evidence/baseline_environment`이며, 이번 라이브 버전은 실행 상태 JSON에 기록했습니다.

```python
from ultralytics import YOLO
model = YOLO("selected-best.pt")
model.predict(source="board_photo.jpg", imgsz=640, conf=0.25, save=True)
```

`conf=0.25`는 예시이며 test로 최적화한 운영 임계값이 아닙니다.

## 최종 Nano 라벨 검토 사진

{sheets}
'''
    (OUT / 'README.md').write_text(report, encoding='utf-8')
    assert sha(OUT / 'selected-best.pt') == selection['selected_checkpoint_sha256']
    assert sha(OUT / 'candidate-best.pt') == summary['best_sha256']
    write(OUT / 'ARTIFACT_MANIFEST.json', {'scope': 'All packaged files except this manifest',
          'files': [{'path': path.relative_to(OUT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
                    for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'ARTIFACT_MANIFEST.json']})
    print(json.dumps({'output': str(OUT), 'promoted': promoted, 'balanced_phase_epochs': summary['epochs_completed'],
                      'selected_checkpoint_sha256': selection['selected_checkpoint_sha256'], 'new_test_invocations': state['test_invocations'],
                      'validation_gain_vs_original_pp': gain * 100, 'validation_gain_vs_previous_pp': current_gain * 100}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
