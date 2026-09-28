"""Preserve a user-requested shutdown checkpoint; never stop/start any process."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from run_r04_evaluation import preflight, validate_completed_files, fingerprint, read

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parents[1]/'outputs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def safe_path(p):
    p=p.resolve()
    if not p.is_relative_to(ROOT.resolve()):raise RuntimeError(f'Path outside R04: {p}')
    return p
def main():
    suite=ROOT/'reports/evaluation_suite'
    evidence=preflight(ROOT,Path(sys.executable))
    existing=read(suite/'protocol.json')
    if fingerprint(evidence)!=existing['protocol_sha256']:raise RuntimeError('Evaluation resume preflight mismatch')
    when=datetime.now(timezone.utc)
    archive=ROOT/'paused_evaluations'/when.strftime('%Y%m%dT%H%M%SZ')
    completed=[];interrupted=[]
    for path in sorted((suite/'jobs').glob('*.json')):
        record=read(path);name=path.stem
        if record['status']=='completed':
            validate_completed_files(record,suite/'evaluations'/name)
            completed.append({'job':name,'job_sha256':sha(path),'metrics_sha256':record['output_sha256']['metrics.json']})
        else:
            if record['binding']['split']!='val':raise RuntimeError('Do not automatically retry an interrupted holdout')
            archived=[]
            for source in [path,suite/'logs'/f'{name}.log',suite/'evaluations'/name]:
                if source.exists():
                    source=safe_path(source)
                    target=safe_path(archive/source.relative_to(suite))
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():raise RuntimeError('Paused archive collision')
                    shutil.move(str(source),str(target))
                    archived.append(str(target))
            interrupted.append({'job':name,'original_status':record['status'],'archive':archived,
                                'reason':'User requested safe save before PC shutdown; repeat only this unfinished validation job on resume'})
    training={arm:read(ROOT/'runs'/arm/'training_summary.json') for arm in ['baseline','improved']}
    if any(r['status']!='COMPLETE' or r['actual_direct_optimizer_calls']!=1165 for r in training.values()):
        raise RuntimeError('Both completed training summaries are required')
    candidate_count=sum(len(read(ROOT/'runs'/arm/'candidates.json')) for arm in training)
    source=read(ROOT/'data/native_manifest.json')
    missing=[r['image'] for r in source['records'] if not Path(r['image']).is_file()]
    if missing:raise RuntimeError(f'Missing native source files {missing}')
    snapshot={'status':'USER_PAUSED_SAFE_TO_SHUT_DOWN','saved_utc':when.isoformat(),'workspace':str(ROOT.parents[1]),
              'r04_root':str(ROOT),'python':sys.executable,'training':training,'saved_candidate_checkpoints':candidate_count,
              'checkpoint_hashes_verified':True,'completed_validation_jobs':len(completed),'planned_validation_jobs':40,
              'completed_jobs':completed,'interrupted_jobs_preserved':interrupted,
              'native_source_images_exist':len(source['records']),'completed_general_pi_holdout_jobs':0,
              'final_selection_frozen':False,'evaluation_protocol_sha256':existing['protocol_sha256'],
              'native_manifest_sha256':sha(ROOT/'data/native_manifest.json'),
              'remaining':['Resume remaining validation only; completed jobs reused by hashes','Freeze all selections before general/Pi holdout',
                           'Run four main holdout jobs','Run resolution stress --run with frozen choices','Independent saved-result audit',
                           'Prepare inference registry and CLI parity','Error analysis and final 9 figures','Package R04 and publish final private GitHub record'],
              'do_not':['Do not retrain either model','Do not edit evaluator/orchestrator/training protocol/candidate files',
                        'Do not delete work/r03 raw caches or work/r04','Do not claim unfinished mAP results or D455 success'],
              'd455_verified':False,'target4_instance_masks_trained':False}
    save(ROOT/'PAUSED_STATE.json',snapshot)
    save(OUT/'PCB_D455_R04_중간저장.json',snapshot)
    with (ROOT.parent/'r04_review/progress.log').open('a',encoding='utf-8') as f:
        f.write(f'[{when.isoformat()}] USER PAUSED: both trainings complete; {len(completed)}/40 validation jobs verified, {len(interrupted)} interrupted job preserved for retry; no active GPU task.\n')
    print(json.dumps({'status':snapshot['status'],'validation_done':len(completed),'checkpoints':candidate_count,'paused_jobs':len(interrupted)}))
if __name__=='__main__':main()
