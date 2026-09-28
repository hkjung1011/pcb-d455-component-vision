"""CPU-only saved validation/selection audit, separate from the frozen final audit."""
from pathlib import Path
import argparse, importlib.util, json, os, traceback
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'r04'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
spec=importlib.util.spec_from_file_location('r04_frozen_audit',ROOT/'scripts/audit_r04_results.py')
auditmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(auditmod)

def run():
    a=auditmod.Audit(ROOT); suite=ROOT/'reports/evaluation_suite'
    a.report['schema']='r04-pre-holdout-saved-validation-audit-v1'
    a.report['source_sha256'][str(Path(__file__).resolve())]=auditmod.sha(__file__)
    try:
        freeze=a.load(HERE/'freeze_record.json')
        a.check_hash(HERE/'ANALYSIS_ADDENDUM_v1.1.md',freeze['document_sha256'])
        for relative,sha in freeze['core_sha256'].items(): a.check_hash(ROOT/relative,sha)
        jobs=[a.load(p) for p in (suite/'jobs').glob('*.json')]
        a.require(not any(j['binding']['split']=='test' for j in jobs),'No holdout job exists before audit')
        a.require(not any(p.name.endswith(('__general_test','__pi_test')) for p in (suite/'evaluations').iterdir()),'No holdout evaluation output directory exists')
        protocol=a.load(ROOT/'protocol.json'); manifest=a.source_truth()
        a.check_hash(ROOT/'data/native_manifest.json',protocol['native_manifest_sha256'])
        a.check_hash(ROOT/'data/summary.json',protocol['data_summary_sha256'])
        a.training(protocol)
        aggregate=a.load(suite/'aggregate.json'); frozen=a.load(suite/'selections_frozen.json'); evaluation=a.load(suite/'protocol.json')
        a.require(aggregate['status']=='SELECTIONS_FROZEN_DEVELOPMENT_HOLDOUT_PENDING','Suite stopped after selections, before holdout')
        a.check_hash(suite/'selections_frozen.json',aggregate['selection_sha256'])
        a.require(frozen['test_results_used_for_selection'] is False,'Selection never used holdout metrics')
        a.require(frozen['protocol_sha256']==evaluation['protocol_sha256']==aggregate['protocol_sha256'],'Selection protocol fingerprint matches')
        frozen_time=auditmod.stamp(frozen['frozen_utc']); winners=[]
        for arm in auditmod.ARMS:
            entry=aggregate['arms'][arm]; choices=[]
            expected={f'{arm}__epoch_{e:02d}__{p}' for e in protocol['save_epoch_candidates'] for p in ['native','dual']}
            a.require(len(entry['validation_candidates'])==len(expected) and {c['candidate_id'] for c in entry['validation_candidates']}==expected,f'{arm}: all20 validation candidates present')
            a.require(entry['tests']=={},f'{arm}: no holdout entries')
            for candidate in entry['validation_candidates']:
                metric,job=a.evaluation_job(candidate,manifest,validate_grid=True)
                a.require(auditmod.stamp(job['finished_utc'])<=frozen_time,f'{candidate["candidate_id"]}: validation completed before selection freeze')
                a.require(all(auditmod.stamp(t['finished_utc'])<=auditmod.stamp(job['started_utc']) for t in a.report['training'].values()),f'{candidate["candidate_id"]}: both training arms finished first')
                key=(-metric['bbox']['ap50_95_max300'],-metric['bbox']['ap50_95_max100'],metric['latency']['offline_predict_and_postprocess_ms_p50'],candidate['candidate_id'])
                choices.append((key,candidate,metric))
            key,winner,metric=min(choices,key=lambda x:x[0]); winners.append((key,arm,winner['candidate_id']))
            selected=frozen['selections'][arm]
            a.require(selected==entry['selection'],f'{arm}: aggregate equals frozen selection')
            a.require(selected['candidate']==winner['candidate'] and selected['validation_metrics_sha256']==winner['metrics_sha256'],f'{arm}: independent rank picks same checkpoint and pipeline')
            a.require(selected['operating_confidence']==metric['operating']['confidence'],f'{arm}: frozen confidence equals independent micro-F1 grid selection')
        _,arm,candidate=min(winners,key=lambda x:x[0])
        a.require(arm==frozen['recommended_arm']==aggregate['recommended_arm'] and candidate==frozen['recommended_candidate_id']==aggregate['recommended_candidate_id'],'Independent overall validation recommendation matches')
        a.require(freeze['holdout_jobs_at_freeze']==0 and auditmod.stamp(freeze['frozen_utc'])<datetime.now(timezone.utc),'Analysis addendum fixed before any holdout')
        a.require(len(jobs)==40 and all(j['status']=='completed' and j['binding']['split']=='val' for j in jobs),'Exactly40 completed validation jobs, zero holdout jobs')
        a.report.update(status='PASS_PRE_HOLDOUT_AUDIT',selection_sha256=aggregate['selection_sha256'],recommended_arm=arm,analysis_addendum_sha256=freeze['document_sha256'],analysis_addendum_frozen_utc=freeze['frozen_utc'],class_weight_note='training_summary unit list is declared metadata; start.json class_weights=null with cls_pw=0 means no class-specific multiplier')
    except Exception as exc:
        a.report['status']='FAIL_PRE_HOLDOUT_AUDIT'; a.report['failures'].append(f'{type(exc).__name__}: {exc}'); traceback.print_exc()
    a.report['finished_utc']=datetime.now(timezone.utc).isoformat()
    output=HERE/'reports'; output.mkdir(exist_ok=True)
    (output/'pre_holdout_audit.json').write_text(json.dumps(a.report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (output/'pre_holdout_audit.md').write_text('# R04 holdout 실행 전 저장 결과 감사\n\n'+a.report['status']+'\n\n'+f"검사 {len(a.report['checks_passed'])}개 통과, 실패 {len(a.report['failures'])}개. GPU 추론 없이 source GT, 학습 budget/20 checkpoint, val40 AP·매칭·19 confidence grid·선택 순위와 해시를 확인했다.\n\n"+'\n'.join(a.report['failures'])+'\n',encoding='utf-8')
    print(json.dumps({'status':a.report['status'],'checks':len(a.report['checks_passed']),'failures':a.report['failures']}))
    if a.report['failures']: raise SystemExit(1)
if __name__=='__main__': run()
