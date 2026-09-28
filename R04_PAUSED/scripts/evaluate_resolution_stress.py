"""Separate, prespecified synthetic resolution sensitivity; CPU preparation by default.

--run is forbidden until the main R04 suite has completed four holdout jobs.
This file does not modify main protocol, native GT, selections, or evaluator.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '7afc529c3ddb6131ba5ee2db9281f8ff8a851407f165799cbdd5b6840d9fbb1d'
NAMES = ['resistor', 'capacitor', 'ic', 'connector']
REQUIRED = ['metrics.json', 'native_ground_truth.json', 'bbox_predictions.json', 'samples.json',
            'native_input_inventory.json', 'inference_profiles.json']
INTERPRETATION_LIMITS = [
    'This measures downsampled public board-image/crop sensitivity after evaluator LetterBox upsampling to imgsz1024; it does not recreate a board occupying the same pixel count inside a D455 1280x800 frame with surrounding background.',
    'Accurate board-region acquisition is assumed; a board detector/ROI-to-component-model chain is not evaluated.',
    'Resizing changes available image context and tile count: each small image has one native tile or two duplicate full-image crops in the frozen dual pipeline, unlike larger original images.',
    'Nominal3.2/2.1px/mm scores must not be used to predict actual D455 camera performance.'
]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def verify_protocol(root):
    p = root/'stress_protocol.json'
    if sha(p) != PROTOCOL_SHA:
        raise ValueError('Frozen auxiliary stress protocol SHA changed')
    protocol = read(p)
    for path, expected in [(root/'protocol.json', protocol['main_protocol_sha256']),
                           (root/'data/native_manifest.json', protocol['source_manifest_sha256'])]:
        if sha(path) != expected:
            raise ValueError(f'Main protected input changed: {path}')
    return protocol


def scaled_bbox(box, original_wh, saved_wh):
    ow, oh = original_wh
    nw, nh = saved_wh
    sx, sy = nw/ow, nh/oh
    raw = [float(box[0])*sx, float(box[1])*sy, float(box[2])*sx, float(box[3])*sy]
    value = [min(max(raw[0], 0.), float(nw)), min(max(raw[1], 0.), float(nh)),
             min(max(raw[2], 0.), float(nw)), min(max(raw[3], 0.), float(nh))]
    correction = max(abs(x-y) for x, y in zip(raw, value))
    if correction > 1e-9:
        raise ValueError('Clipping beyond floating-point boundary roundoff is forbidden')
    if not (0 <= value[0] < value[2] <= nw and 0 <= value[1] < value[3] <= nh):
        raise ValueError('Scaled bbox invalid; tiny boxes must not be discarded')
    restored = [value[0]/sx, value[1]/sy, value[2]/sx, value[3]/sy]
    return value, correction, max(abs(float(a)-b) for a, b in zip(box, restored))


def scale_object(obj, original_wh, saved_wh):
    result = copy.deepcopy(obj)
    result['original_source_bbox_xyxy'] = list(obj['bbox_xyxy'])
    result['bbox_xyxy'], correction, error = scaled_bbox(obj['bbox_xyxy'], original_wh, saved_wh)
    result['annotation_status'] = 'source_bbox_scaled_synthetic'
    if 'bbox_voc_raw' in result:
        result['bbox_voc_raw_scope'] = 'Unchanged original-source coordinates, not scaled-image coordinates'
    return result, correction, error


def verify_prepared(root, protocol):
    folder = root/'data/resolution_stress'
    preparation = read(folder/'preparation.json')
    if preparation['stress_protocol_sha256'] != PROTOCOL_SHA or preparation['source_manifest_sha256'] != protocol['source_manifest_sha256']:
        raise ValueError('Existing preparation belongs to different protocol')
    original = {r['id']: r for r in read(root/'data/native_manifest.json')['records'] if r.get('cohort') == 'pi_test'}
    for condition in preparation['conditions']:
        path = Path(condition['manifest_path'])
        if sha(path) != condition['manifest_sha256']:
            raise ValueError('Prepared stress manifest changed')
        manifest = read(path)
        if len(manifest['records']) != 2 or sum(len(r['objects']) for r in manifest['records']) != 189:
            raise ValueError('Stress data must retain both images and all189 target objects')
        for row in manifest['records']:
            src = original[row['id']]
            if sha(row['source_image']) != row['sha256'] or sha(src['source_image']) != src['sha256']:
                raise ValueError('Original/derived image SHA mismatch')
            if row['group_id'] != src['group_id'] or row['split'] != src['split'] or row['cohort'] != src['cohort']:
                raise ValueError('Stress source grouping changed')
            for key in ['objects', 'excluded_source_objects']:
                if len(row[key]) != len(src[key]):
                    raise ValueError('Source object or exclusion removed')
                for target, source in zip(row[key], src[key]):
                    expected, _, _ = scale_object(source, (src['width'], src['height']), (row['width'], row['height']))
                    if target != expected:
                        raise ValueError('Scaled object, source class/type or metadata differs')
    verify_protocol(root)
    return preparation


def prepare(root):
    from PIL import Image, __version__ as pillow_version
    protocol = verify_protocol(root)
    folder = root/'data/resolution_stress'
    if (folder/'preparation.json').exists():
        return verify_prepared(root, protocol)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Partial stress preparation retained; do not overwrite')
    source = read(root/'data/native_manifest.json')
    rows = [r for r in source['records'] if r.get('cohort') == 'pi_test']
    if len(rows) != 2 or sum(len(r['objects']) for r in rows) != 189 or len({r['group_id'] for r in rows}) != 1:
        raise ValueError('Unexpected original Pi cohort')
    source_counts = dict(Counter(NAMES[o['class_id']] for r in rows for o in r['objects']))
    source_excluded = dict(Counter(o['source_type'] for r in rows for o in r['excluded_source_objects']))
    record = {'schema':'r04-resolution-stress-preparation-v1', 'prepared_utc':now(),
              'stress_protocol_sha256':PROTOCOL_SHA, 'source_manifest_sha256':protocol['source_manifest_sha256'],
              'pillow_version':pillow_version, 'evaluator_sha256':sha(root/'scripts/evaluate_r04.py'),
              'training_eligible':False, 'd455_verified':False, 'mask':None, 'conditions':[]}
    for condition in protocol['conditions']:
        result = copy.deepcopy(source)
        result.update(schema='r04-synthetic-resolution-bbox-v1', training_eligible=False,
                      scope='Synthetic downsampled public Pi photographs with geometrically scaled original source bbox GT; not original-resolution evaluation or D455 capture',
                      annotation_status='source_bbox_scaled_synthetic',
                      annotation_limitations=protocol['limitations']+[
                          'Every target and excluded bbox is scaled by actual output-width/height ratio; original IDs/classes/source ontology preserved.',
                          'Evaluator ground_truth_policy is a generic source-ontology template: here coordinates ARE scaled derivatives. No evaluator metric file is rewritten.',
                          'Tiny/subpixel labels are retained even if visually unresolved; not human-validated D455 labels.'],
                      stress_protocol_sha256=PROTOCOL_SHA, synthetic_condition=condition,
                      records=[])
        image_records = []
        max_clip = max_roundtrip = 0.
        for row in rows:
            if sha(row['source_image']) != row['sha256']:
                raise ValueError('Original Pi image SHA mismatch')
            estimate = protocol['source_scale_approx_px_per_mm'][row['id']]
            factor = condition['target_nominal_px_per_mm']/estimate
            nw, nh = max(1, round(row['width']*factor)), max(1, round(row['height']*factor))
            dest = folder/'images'/condition['id']/(row['id']+'.png')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(row['source_image']) as image:
                if image.size != (row['width'], row['height']):
                    raise ValueError('Original image dimensions differ from source manifest')
                image.convert('RGB').resize((nw, nh), Image.Resampling.LANCZOS).save(dest, format='PNG')
            out = copy.deepcopy(row)
            out.update(original_source_image=row['source_image'], original_source_image_sha256=row['sha256'],
                       original_source_width=row['width'], original_source_height=row['height'],
                       source_image=str(dest.resolve()), image=str(dest.resolve()), width=nw, height=nh,
                       sha256=sha(dest), annotation_status='source_bbox_scaled_synthetic')
            for key in ['objects','excluded_source_objects']:
                out[key] = []
                for obj in row[key]:
                    scaled, clip, error = scale_object(obj, (row['width'],row['height']), (nw,nh))
                    out[key].append(scaled)
                    max_clip = max(max_clip, clip)
                    max_roundtrip = max(max_roundtrip, error)
            out['synthetic_transform'] = {'resample':'Pillow.LANCZOS', 'source_approx_px_per_mm':estimate,
                'target_nominal_px_per_mm':condition['target_nominal_px_per_mm'], 'nominal_factor':factor,
                'actual_sx':nw/row['width'], 'actual_sy':nh/row['height'],
                'approx_realized_px_per_mm_x':estimate*nw/row['width'],
                'approx_realized_px_per_mm_y':estimate*nh/row['height']}
            result['records'].append(out)
            image_records.append({'id':row['id'],'source_sha256':row['sha256'],'derived_sha256':out['sha256'],
                'original_wh':[row['width'],row['height']], 'saved_wh':[nw,nh], **out['synthetic_transform'],
                'target_count':len(out['objects']), 'excluded_count':len(out['excluded_source_objects']),
                'subpixel_target_bbox_count':sum(min(o['bbox_xyxy'][2]-o['bbox_xyxy'][0],o['bbox_xyxy'][3]-o['bbox_xyxy'][1])<1 for o in out['objects']),
                'subpixel_excluded_bbox_count':sum(min(o['bbox_xyxy'][2]-o['bbox_xyxy'][0],o['bbox_xyxy'][3]-o['bbox_xyxy'][1])<1 for o in out['excluded_source_objects'])})
        counts = dict(Counter(NAMES[o['class_id']] for r in result['records'] for o in r['objects']))
        excluded = dict(Counter(o['source_type'] for r in result['records'] for o in r['excluded_source_objects']))
        if counts != source_counts or excluded != source_excluded:
            raise ValueError('Original target or excluded-source types were dropped')
        result['counts'] = {'test':{'images':2,'groups':1,'label_instances':189,'per_class':counts}}
        result['native_split_counts'] = {'pi_test':{'images':2,'board_groups':1,'unique_instances':counts}}
        result['validation_policy'] = 'No validation or selection; reuse frozen main selection verbatim'
        manifest_path = folder/(condition['id']+'_manifest.json')
        save(manifest_path,result)
        record['conditions'].append({'condition':condition, 'manifest_path':str(manifest_path.resolve()),
            'manifest_sha256':sha(manifest_path), 'target_count':189, 'class_counts':counts,
            'excluded_source_type_counts':excluded, 'images':image_records,
            'max_boundary_roundoff_clip_px':max_clip, 'max_original_coordinate_roundtrip_error_px':max_roundtrip})
    save(folder/'preparation.json',record)
    return verify_prepared(root,protocol)


def main_gate(root, protocol):
    suite = root/'reports/evaluation_suite'
    aggregate_path, selection_path = suite/'aggregate.json', suite/'selections_frozen.json'
    if not aggregate_path.exists() or not selection_path.exists():
        raise ValueError('Main evaluation has not completed; no stress inference allowed')
    aggregate, selection = read(aggregate_path), read(selection_path)
    if aggregate['status'] != 'COMPLETED_DEVELOPMENT_HOLDOUT_NOT_D455_VALIDATED':
        raise ValueError('Main evaluation is not complete; no parallel stress inference')
    selection_sha = sha(selection_path)
    if selection_sha != aggregate['selection_sha256'] or selection.get('test_results_used_for_selection') is not False:
        raise ValueError('Main frozen-selection binding invalid')
    if datetime.fromisoformat(protocol['frozen_at_utc']) >= datetime.fromisoformat(selection['frozen_utc']):
        raise ValueError('Stress protocol was not frozen before main model selection')
    count = 0
    for arm in protocol['models']:
        tests = aggregate['arms'][arm]['tests']
        if set(tests) != {'general_test','pi_test'}:
            raise ValueError('Exactly four main holdout jobs must complete first')
        if aggregate['arms'][arm]['selection'] != selection['selections'][arm]:
            raise ValueError('Aggregate selection differs from frozen selection')
        for test in tests.values():
            job = read(test['job_path'])
            if job['status'] != 'completed' or job['binding']['selection_sha256'] != selection_sha:
                raise ValueError('Main holdout job incomplete or different selection')
            if sha(test['job_path']) != test['job_sha256'] or sha(test['metrics_path']) != test['metrics_sha256']:
                raise ValueError('Main evaluation evidence changed')
            if datetime.fromisoformat(protocol['frozen_at_utc']) >= datetime.fromisoformat(job['started_utc']):
                raise ValueError('Stress protocol must precede each main holdout test')
            count += 1
    if count != 4:
        raise ValueError('Unexpected main test job count')
    return aggregate, selection, {'selection_path':str(selection_path),'selection_sha256':selection_sha,
        'aggregate_path':str(aggregate_path),'aggregate_sha256':sha(aggregate_path)}


def check_job_metric(metric, binding):
    for key in ['checkpoint_sha256','manifest_sha256','evaluator_sha256','pipeline']:
        if metric[key] != binding[key]:
            raise ValueError('Stress result binding mismatch: '+key)
    if metric['images'] != 2 or metric['groups'] != 1 or metric['native_gt_instances'] != 189:
        raise ValueError('Stress metric changed sample/GT count')
    if metric['split'] != 'test' or metric['cohort'] != 'pi_test' or metric['task'] != 'detect':
        raise ValueError('Stress evaluation scope changed')
    if metric['d455_verified'] is not False or metric['mask'] is not None or metric['group_bootstrap_recall95'] is not None:
        raise ValueError('Stress metric mislabeled D455/mask/groupCI')
    if metric['threshold_source'] != 'supplied_frozen_confidence' or abs(metric['operating']['confidence']-binding['confidence'])>1e-12:
        raise ValueError('Stress confidence retuned or changed')


def run(root, python):
    protocol = verify_protocol(root)
    prepared = verify_prepared(root, protocol)
    aggregate, selection, reuse = main_gate(root, protocol)
    evaluator = root/'scripts/evaluate_r04.py'
    evaluator_sha = sha(evaluator)
    if evaluator_sha != prepared['evaluator_sha256']:
        raise ValueError('Evaluator changed after stress preparation')
    folder = root/'reports/resolution_stress'
    jobs = []
    for arm in protocol['models']:
        chosen = selection['selections'][arm]
        candidate = chosen['candidate']
        for condition in prepared['conditions']:
            cid = condition['condition']['id'];jid=arm+'__'+cid
            output=folder/'evaluations'/jid;control=folder/'jobs'/(jid+'.json');log=folder/'logs'/(jid+'.log')
            binding={'stress_protocol_sha256':PROTOCOL_SHA, 'selection_sha256':reuse['selection_sha256'],
                'checkpoint_sha256':chosen['checkpoint_sha256'], 'manifest_sha256':condition['manifest_sha256'],
                'evaluator_sha256':evaluator_sha, 'pipeline':candidate['pipeline'],
                'confidence':chosen['operating_confidence'], 'arm':arm, 'condition':cid,
                'device':'0', 'bootstrap':0, 'source_gt_count':189}
            if sha(candidate['weights']) != binding['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed')
            verify_protocol(root)
            if sha(condition['manifest_path']) != binding['manifest_sha256'] or sha(evaluator) != evaluator_sha:
                raise ValueError('Stress manifest/evaluator changed before job')
            if sha(reuse['selection_path']) != reuse['selection_sha256'] or sha(reuse['aggregate_path']) != reuse['aggregate_sha256']:
                raise ValueError('Main frozen evidence changed before stress job')
            if control.exists():
                record=read(control)
                if record['status']!='completed' or record['binding']!=binding:
                    raise ValueError('Existing partial/different stress job retained without automatic rerun')
                if set(record['output_sha256']) != set(REQUIRED):
                    raise ValueError('Completed stress job lacks required output inventory')
                for name,value in record['output_sha256'].items():
                    if sha(output/name)!=value:raise ValueError('Completed stress output changed')
            else:
                if output.exists() and any(output.iterdir()):
                    raise ValueError('Nonempty untracked stress output preserved')
                command=[str(python),'-B',str(evaluator),'--manifest',condition['manifest_path'],
                    '--weights',candidate['weights'],'--output',str(output),'--split','test','--cohort','pi_test',
                    '--pipeline',candidate['pipeline'],'--operating-conf',format(chosen['operating_confidence'],'.17g'),
                    '--bootstrap','0','--device','0']
                record={'status':'running','started_utc':now(),'binding':binding,'command':command}
                save(control,record);log.parent.mkdir(parents=True,exist_ok=True)
                print('STRESS RUN '+jid,flush=True)
                try:
                    with log.open('w',encoding='utf-8') as handle:
                        completed=subprocess.run(command,cwd=root,stdout=handle,stderr=subprocess.STDOUT,check=False)
                    if completed.returncode:raise RuntimeError('Evaluator failed; see '+str(log))
                    metric=read(output/'metrics.json');check_job_metric(metric,binding)
                    verify_protocol(root)
                    if sha(reuse['selection_path'])!=reuse['selection_sha256'] or sha(reuse['aggregate_path'])!=reuse['aggregate_sha256']:
                        raise ValueError('Frozen main evaluation changed during stress')
                    if sha(candidate['weights'])!=binding['checkpoint_sha256'] or sha(evaluator)!=evaluator_sha:
                        raise ValueError('Checkpoint/evaluator changed')
                    record.update(status='completed',finished_utc=now(),output_sha256={n:sha(output/n) for n in REQUIRED})
                    save(control,record)
                except BaseException as error:
                    record.update(status='failed',finished_utc=now(),error=repr(error));save(control,record);raise
            metric=read(output/'metrics.json');check_job_metric(metric,binding)
            jobs.append({'job_id':jid,'binding':binding,'metrics_path':str(output/'metrics.json'),
                'metrics_sha256':sha(output/'metrics.json'),'job_path':str(control),'job_sha256':sha(control),
                'bbox':metric['bbox'],'operating':metric['operating'],'latency':metric['latency'],
                'd455_verified':False,'mask':None,'group_bootstrap_recall95':None,
                'ground_truth_scope':'189 original-source target boxes scaled by actual sx/sy; synthetic derived-image coordinates, no object deletion'})
            print('STRESS DONE '+jid,flush=True)
    if len(jobs)!=4:raise ValueError('Expected exactly four stress jobs')
    table=[]
    for arm in protocol['models']:
        original=aggregate['arms'][arm]['tests']['pi_test']
        base={'arm':arm,'condition':'original_reference','pipeline':selection['selections'][arm]['candidate']['pipeline'],
              'confidence':original['operating']['confidence'],'ap50_95_max300':original['bbox']['ap50_95_max300'],
              'ap50_95_max100':original['bbox']['ap50_95_max100'],'precision':original['operating']['precision'],
              'recall':original['operating']['recall'],'tp':original['operating']['tp'],'fp':original['operating']['fp'],
              'fn':original['operating']['fn'],'delta_ap300_percentage_points':0.,'new_inference':False}
        table.append(base)
        for job in [j for j in jobs if j['binding']['arm']==arm]:
            m=job;op=m['operating'];table.append({'arm':arm,'condition':job['binding']['condition'],
                'pipeline':job['binding']['pipeline'],'confidence':job['binding']['confidence'],
                'ap50_95_max300':m['bbox']['ap50_95_max300'],'ap50_95_max100':m['bbox']['ap50_95_max100'],
                'precision':op['precision'],'recall':op['recall'],'tp':op['tp'],'fp':op['fp'],'fn':op['fn'],
                'delta_ap300_percentage_points':100*(m['bbox']['ap50_95_max300']-base['ap50_95_max300']),
                'new_inference':True})
    summary={'status':'COMPLETED_SYNTHETIC_RESOLUTION_SENSITIVITY_NOT_D455','completed_utc':now(),
        'stress_protocol_sha256':PROTOCOL_SHA,'stress_protocol_frozen_utc':protocol['frozen_at_utc'],
        'main_reuse':reuse,'selected_models_or_thresholds_changed':False,'additional_training':False,
        'main_holdout_jobs':4,'separate_stress_jobs':4,'d455_verified':False,'mask':None,
        'group_bootstrap_recall95':None,'prepared_data_sha256':sha(root/'data/resolution_stress/preparation.json'),
        'limitations':protocol['limitations']+INTERPRETATION_LIMITS,'metric_template_note':'Unmodified evaluator metrics contain its generic source-GT-policy text. In this experiment all coordinates are scaled synthetic derivatives; wrapper scope and manifest are authoritative. Raw metrics are not rewritten.',
        'conditions':prepared['conditions'],'jobs':jobs,'comparison_rows':table}
    save(folder/'summary.json',summary)
    with (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    text=['# 공개 Pi 합성 해상도 민감도 시험','',
        '**실제 D455 촬영·화질·실사용 검증이 아니다.** 명목 픽셀 배율로 축소한 같은 공개 Pi 앞·뒤 사진 2장, 한 보드 그룹의 진단 결과다. 성능 상한이나 하한으로 해석하지 않는다.',
        '', '주 평가 완료 후 선택된 두 모델의 checkpoint·native/dual 방식·confidence를 그대로 재사용했다. 모델이나 threshold를 이 결과로 다시 고르지 않았다. 보조 protocol은 '+protocol['frozen_at_utc']+'에 동결됐고 주 모델 선택·시험보다 먼저 작성됐음을 검사했다.',
        '', '| 모델 | 조건 | AP50–95 max300 | AP50–95 max100 | P | R | TP / FP / FN |', '|---|---|---:|---:|---:|---:|---|']
    for r in table:
        text.append('| {} | {} | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {} / {} / {} |'.format(r['arm'],r['condition'],100*r['ap50_95_max300'],100*r['ap50_95_max100'],100*r['precision'],100*r['recall'],r['tp'],r['fp'],r['fn']))
    text += ['', 'original_reference는 주 평가의 저장 결과이며 재추론하지 않았다. 추가 추론은 두 모델×두 축소 조건의 총4개 job이다. 결과의 bbox 정답189개와 unknown·비대상 원본 유형을 보존하고, 저장된 실제 가로·세로 비율 sx/sy로 좌표를 변환했다. 1px 미만 라벨도 삭제하지 않는다.', '',
        '| 조건 | 사진 | 원본 → 저장 크기 | sx / sy |', '|---|---|---|---|']
    for c in prepared['conditions']:
        for im in c['images']:
            text.append('| {} | {} | {} → {} | {:.8f} / {:.8f} |'.format(c['condition']['id'],im['id'],im['original_wh'],im['saved_wh'],im['actual_sx'],im['actual_sy']))
    text += ['', 'Top22.57px/mm·Bottom14.23px/mm는 공개 사진의 헤더 간격에서 얻은 근사 배율이다. 목표3.2/2.1px/mm도 카메라에서 실측한 값이 아니다. 광학 해상도·초점·노이즈·반사·노출을 재현하지 않는다. 저장 이미지를 원래 크기로 확대하지 않았지만, 선택된 평가 파이프라인 내부의 imgsz1024 LetterBox 확대는 그대로 적용된다.', '',
        '따라서 이 시험은 저해상도 보드 영역 crop을 확보한 뒤 1024 입력으로 확대하는 상황에 가깝다. D455의 1280×800 전체 프레임 안에서 보드가 같은 픽셀 수만 차지하고 주변 배경이 포함되는 상황과 다르다. 정확한 보드 영역 확보를 가정하며, 보드 검출기·ROI 추출·부품 검출기의 연결 성능은 평가하지 않았다. 명목3.2/2.1px/mm 결과를 실제 카메라 성능 예측으로 쓰면 안 된다.', '',
        '축소로 입력 context와 타일 수도 바뀐다. 작은 각 사진은 native에서 전체 사진 한 crop, dual에서는 1024·2048 요청 모두 전체 사진이 되어 같은 영역 두 crop을 처리한다. 원본의 여러 타일 평가와 이 조건이 다르므로 해상도만 고립시킨 광학 시험으로 해석하지 않는다. 선택된 pipeline을 그대로 유지한 입력 민감도 결과다.', '',
        '기본 evaluator의 ground_truth_policy 문구는 source 정답 정의를 보존한다는 공통 템플릿이다. 여기서는 좌표가 sx/sy로 변환된 파생 GT이며 원본 해상도 평가가 아니다. 이 차이는 manifest와 본 wrapper에 명시하고 원 metrics를 덮어쓰지 않았다.', '',
        '같은 한 그룹이므로 보드 간 신뢰구간은 계산하지 않았다. mask 지표는 null, D455 검증은 false다. 작은 부품이 사람이 읽을 수 있는지 또는 윤곽을 그릴 수 있는지는 이 기하 기반 시험에서 검증하지 않았다. 원본 모델 선택과 주 평가4개 job은 변경하지 않았다.', '',
        '세부 수치와 선택 재사용 SHA는 summary.json, 비교표는 summary.csv, 각 원본 평가 파일은 evaluations/에 있다.']
    (folder/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('STRESS COMPLETE '+str(folder/'summary.json'),flush=True)


def self_test():
    # Meaningful corner cases: retain subpixel boxes and undo anisotropic rounding.
    source=[0.,0.,1.,1.]
    box,clip,error=scaled_bbox(source,(2000,1350),(186,126))
    assert box[2]<1 and box[3]<1 and box[2]>0 and box[3]>0 and error<1e-10 and clip==0
    edge,clip,error=scaled_bbox([1999.,1349.,2000.,1350.],(2000,1350),(284,191))
    assert edge[2]<=284 and edge[3]<=191 and error<1e-9
    obj={'instance_id':'a','source_type':'unknown','bbox_xyxy':[1,2,3,4],'bbox_voc_raw':[2,3,3,4]}
    out,_,_=scale_object(obj,(20,20),(3,4))
    assert obj['bbox_xyxy']==[1,2,3,4] and out['source_type']=='unknown' and out['instance_id']=='a'
    assert out['original_source_bbox_xyxy']==[1,2,3,4] and out['bbox_voc_raw']==obj['bbox_voc_raw']
    print('PASS CPU geometry tests: subpixel retention, boundary rounding, anisotropic inverse, original metadata preservation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--run',action='store_true',help='Run four jobs only after main evaluation completes; default is CPU preparation only')
    parser.add_argument('--self-test',action='store_true',help='CPU-only geometry checks, no file or model operations')
    args=parser.parse_args()
    if args.self_test:return self_test()
    root=args.root.resolve()
    prepared=prepare(root)
    print('CPU PREPARATION VERIFIED: '+str(len(prepared['conditions']))+' conditions;189 GT each',flush=True)
    if args.run:run(root,args.python.resolve())


if __name__=='__main__':main()
