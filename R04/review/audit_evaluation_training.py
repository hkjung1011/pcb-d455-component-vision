"""Read-only R03 evaluation/training audit; CPU only, no model inference."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R03 = HERE.parent / 'r03'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evidence = {}
    def track(path):
        evidence[str(Path(path).resolve())] = sha(path)
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'CPU metadata and saved prediction audit only; no training, inference, source mutation or remote operation',
             'status': 'R03_REVIEW_VERIFIED_WITH_QUALIFICATIONS_NOT_D455_VALIDATED'}
    manifest_path = R03 / 'component_assets/native_manifest.json'
    manifest = read(manifest_path)
    track(manifest_path)
    names = manifest['names']
    split_stats = {}
    dense = []
    group_stats = {}
    for cohort in ['train', 'val', 'test', 'pi_test']:
        rows = [r for r in manifest['records'] if r['cohort'] == cohort]
        bins = Counter()
        classes = Counter()
        cross = defaultdict(Counter)
        for r in rows:
            counts = Counter()
            for obj in r['objects']:
                x1,y1,x2,y2 = obj['bbox_xyxy']
                short = min(x2-x1,y2-y1)
                bucket = 'lt8' if short<8 else '8to16' if short<16 else '16to32' if short<32 else 'ge32'
                name = names[obj['class_id']]
                bins[bucket] += 1
                classes[name] += 1
                cross[name][bucket] += 1
                counts[name] += 1
            for name,count in counts.items():
                if count>100:
                    dense.append({'cohort':cohort,'image':r['id'],'class':name,'gt_count':count,
                                  'max100_recall_ceiling_for_image_class':100/count})
            if cohort in ['train','val']:
                g = group_stats.setdefault(r['group_id'], {'original_split':cohort,'images':0,'class_counts':Counter(),'shortside_counts':Counter()})
                g['images']+=1
                for obj in r['objects']:
                    x1,y1,x2,y2=obj['bbox_xyxy']; short=min(x2-x1,y2-y1)
                    bucket='lt8' if short<8 else '8to16' if short<16 else '16to32' if short<32 else 'ge32'
                    g['class_counts'][names[obj['class_id']]]+=1
                    g['shortside_counts'][bucket]+=1
        total=sum(classes.values())
        split_stats[cohort]={'images':len(rows),'groups':len({r['group_id'] for r in rows}),'objects':total,
                             'classes':dict(classes),'shortside_bins':dict(bins),
                             'class_x_shortside':{k:dict(v) for k,v in cross.items()},
                             'shortside_8to16_fraction':bins['8to16']/total}
    audit['split_distribution'] = split_stats
    audit['trainval_group_inventory_for_restratification_only'] = group_stats
    audit['dense_image_class_counts'] = dense
    training = {}
    for name in ['parts_yolo11n','parts_yolo11s','board_yolo11n','board_yolo11n_seg']:
        p=R03/'runs'/name
        envpath=p/'environment.json'; evpath=p/'epoch_updates.json'; csvpath=p/'fit/results.csv'
        for path in [envpath,evpath,csvpath]: track(path)
        env=read(envpath); events=read(evpath); cfg=env['configuration']['train']
        csvrows=list(csv.DictReader(csvpath.open(encoding='utf-8-sig')))
        actual_epochs={int(row['epoch']) for row in csvrows}
        events_train=[e for e in events if e['epoch'] in actual_epochs]
        n=env['label_counts']['train']['images']; nb=math.ceil(n/cfg['batch'])
        nw=round(min(cfg['warmup_epochs'],cfg['epochs']-1)*nb)
        count=0;last=-1;acc=max(round(cfg['nbs']/cfg['batch']),1)
        for ni in range(nb*cfg['epochs']):
            if ni<nw: acc=max(1,int(np.interp(ni,[0,nw],[1,cfg['nbs']/cfg['batch']]).round()))
            if ni-last>=acc:count+=1;last=ni
        ckpath=p/'fit/weights/best.pt';track(ckpath)
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        cm=ck['train_metrics']
        matching=[int(row['epoch']) for row in csvrows
                  if all(abs(float(row[k])-float(v))<5e-6*max(1,abs(float(v)))
                         for k,v in cm.items() if k in row)]
        apkey='metrics/mAP50-95(B)'
        maxrow=max(csvrows,key=lambda row:float(row[apkey]))
        training[name]={'epochs_completed':len(csvrows),'train_images_or_tiles':n,'batch':cfg['batch'],
                       'state_steps_min':max(e['optimizer_steps_min'] for e in events_train),
                       'state_steps_max':max(e['optimizer_steps_max'] for e in events_train),
                       'optimizer_step_call_counter_recorded':False,
                       'deterministically_reconstructed_calls_if_uninterrupted':count,
                       'reconstruction_assumptions':['No loader-length or batch change/restart','Current installed trainer matches training version','All scheduled batches completed'],
                       'checkpoint_epoch_after_stripping':ck.get('epoch'),
                       'checkpoint_ema_updates_after_stripping':ck.get('updates'),
                       'best_checkpoint_train_metrics_matching_csv_epochs':matching,
                       'highest_training_loader_bbox_ap_epoch':int(maxrow['epoch']),
                       'highest_training_loader_bbox_ap':float(maxrow[apkey]),
                       'train_metrics':cm,
                       'terminal_callback_epochs_excluded':[e['epoch'] for e in events if e['epoch'] not in actual_epochs]}
    audit['training_accounting']=training
    spec=importlib.util.spec_from_file_location('r03_native_eval_readonly',R03/'scripts/native_evaluate.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    recalculated={}
    for cohort in ['general_test','pi_test']:
        p=R03/'reports/evaluation_suite/evaluations'/f'parts_yolo11s__{cohort}'
        paths=[p/'metrics.json',p/'native_ground_truth.json',p/'bbox_predictions.json',p/'samples.json']
        for path in paths:track(path)
        metric=read(paths[0]); co=evaluator.coco_score(read(paths[1]),read(paths[2]))
        op=evaluator.operate(read(paths[3]),metric['operating']['confidence'],4)
        assert abs(co['ap50_95_max100']-metric['bbox']['ap50_95_max100'])<1e-12
        assert abs(co['ap50_95_max300']-metric['bbox']['ap50_95_max300'])<1e-12
        assert op['per_class_counts_tp_fp_fn']==metric['operating']['per_class_counts_tp_fp_fn']
        recalculated[cohort]={'ap100':co['ap50_95_max100'],'ap300':co['ap50_95_max300'],
                             'confidence':op['confidence'],'per_class_tp_fp_fn':op['per_class_counts_tp_fp_fn'],
                             'precision':op['precision'],'recall':op['recall'],
                             'group_bootstrap_recall95':metric.get('group_bootstrap_recall95'),
                             'saved_predictions_recalculation_matches':True}
    audit['cpu_saved_prediction_recalculation']=recalculated
    selections=read(R03/'reports/evaluation_suite/selection.json')
    candidates={}
    for name in ['parts_yolo11n','parts_yolo11s']:
        candidates[name]={}
        for candidate in ['best_whole','last_whole','best_tile1024','last_tile1024']:
            path=R03/'reports/evaluation_suite/evaluations'/f'{name}__val__{candidate}'/'metrics.json';track(path)
            m=read(path)
            candidates[name][candidate]={'ap100':m['bbox']['ap50_95_max100'],'ap300':m['bbox']['ap50_95_max300'],
                                         'confidence':m['operating']['confidence']}
    audit['validation_candidates']=candidates
    audit['claim_verdicts']=[
        {'claim':'1165/2615/1477 are measured optimizer.step call counts','verdict':'QUALIFY',
         'finding':'They are measured maxima of per-parameter AdamW state.step. No independent call hook was recorded. Deterministic scheduler reconstruction gives the same values; use inferred scheduled calls, not directly measured calls.'},
        {'claim':'s best checkpoint is epoch18','verdict':'SUPPORTED_BY_RECONSTRUCTION',
         'finding':'Stripped checkpoint epoch=-1; its train_metrics uniquely match CSV epoch18, highest whole-loader fitness, and installed save_model saves fitness maxima. Record reconstructed epoch18, not a surviving epoch tag.'},
        {'claim':'val small-object coverage is poor','verdict':'CONFIRMED',
         'finding':'Native 8<=shortside<16 counts: val24/875, general446/1348, Pi77/189. This is a selection representativeness limitation, not a COCO arithmetic error.'},
        {'claim':'best checkpoint selection differs from tiled deployment','verdict':'CONFIRMED_WITH_SCOPE',
         'finding':'Training best is whole-image1024 fitness. R03 later correctly chose among best/last x whole/tile on native val, but never evaluated every epoch in deployed tiled mode; this restricts candidates, not test leakage.'},
        {'claim':'AP300 should be available for dense scenes','verdict':'CONFIRMED',
         'finding':'COCO maxDets truncates per image/category here. R03 computed both correctly. Keep AP100 historical result and prespecify AP300 for R04; do not replace R03 metric definition retroactively.'},
        {'claim':'Graph05 does not mention training-loader versus final native tiled evaluation','verdict':'PARTLY_INCORRECT',
         'finding':'Graph source footer already says VAL monitor comes from training loader and final native/tiled evaluation is separate. Add explicit whole-image downsample1024 wording and chosen-checkpoint epoch to make it clearer.'},
        {'claim':'s better than n is established','verdict':'NOT_ESTABLISHED',
         'finding':'s is the validation-selected default. One Pi board group and broad general-test group recall interval do not establish significant model superiority; distinct thresholds also confound precision comparisons.'},
        {'claim':'Synthetic degraded-image performance is a lower bound on D455','verdict':'UNSUPPORTED',
         'finding':'Neither resolution reduction nor assumed fx proves a lower bound for real optics, focus, noise, exposure, aliasing or domain shift. Use sensitivity probe only.'}
    ]
    audit['r04_recommended_protocol']={
        'test_status':'R03 general/Pi test are now development holdouts because diagnostics informed R04. Preserve their image membership and original source GT; do not claim fresh final test. New independent D455 test remains unavailable.',
        'split':'Restratify only the previous train+val union, preserving source board groups. Freeze labels/manifest SHA before fitting; objective uses class and native-size coverage, never predictions or R03 test performance. Compare R04 baseline and improved arm under identical new split.',
        'initialization':'Start both arms from identical official pretrained checkpoint SHA, not the R03 fine-tuned model after repartitioning (it has seen some new-val groups).',
        'budget':'At most20 epochs per arm plus prespecified equal optimizer-call budget, image/tile exposure and walltime records. Equal epochs alone are not equal effort when adding scales/oversampling; nbs8 does not guarantee same warmup updates with batch2 versus8.',
        'counter':'Log optimizer.step pre/post hooks, per-parameter step min/max, EMA updates and global batches separately; count actual finite parameter change if needed. Preserve unstripped periodic metadata so best epoch survives.',
        'checkpoint_selection':'Prespecify every2 completed epochs plus final checkpoint, label 1-based epoch explicitly (Ultralytics save_period names use0-based epoch). Evaluate each with final native/tile/multiscale pipeline on frozen val. Choose native AP300 then AP100 then prespecified tie-break. Freeze checkpoint SHA and class/global thresholds before development-holdout run.',
        'metrics':'Native AP100+AP300, class-by-size AP/recall, PR curves; one confidence policy fixed on val. Report paired changes descriptively; preserve raw source-GT main metric and separate reviewed-label sensitivity analysis.',
        'annotation_limits':'Known excluded categories are hard negatives for target4 only if ontology is reliable; unknown may include target4 and remains incomplete-label limitation. Filling gray is image masking, not mathematically ignored loss/COCO GT. Genuine ignore needs explicit supported loss/evaluation logic.',
        'd455':'Real camera pixel dimensions/intrinsics/focus still need measurement. Synthetic down/up sampling is diagnostic, not D455 evidence or lower bound.'
    }
    for path in [R03/'scripts/train_model.py',R03/'scripts/native_evaluate.py',R03/'scripts/run_evaluation_suite.py',R03/'scripts/make_r03_result_figures.py',R03/'reports/evaluation_suite/selection.json']:
        track(path)
    import ultralytics
    engine=Path(ultralytics.__file__).parent/'engine/trainer.py'
    metrics=Path(ultralytics.__file__).parent/'utils/metrics.py'
    track(engine);track(metrics)
    audit['source_sha256']=evidence
    (HERE/'evaluation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'audit':str(HERE/'evaluation_audit.json'),'training':training,'saved_prediction_recalculation':recalculated},ensure_ascii=False))


if __name__=='__main__':
    main()
