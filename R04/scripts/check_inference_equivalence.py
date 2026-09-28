"""Compare one offline CLI replay to frozen predictions; never retune or rescore a holdout."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from evaluate_r04 import sha, utc

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    aggregate=read(ROOT/'reports/evaluation_suite/aggregate.json')
    arm=aggregate['recommended_arm']
    test=aggregate['arms'][arm]['tests']['pi_test']
    samples=read(Path(test['metrics_path']).with_name('samples.json'))
    sample=next(row for row in samples if row['id']=='RPI3B_Top')
    manifest=read(ROOT/'data/native_manifest.json')
    image=next(row['image'] for row in manifest['records'] if row['id']==sample['id'])
    # Source-photo overlays remain outside packaged reports and outputs.
    destination=ROOT.parent/'r04_runtime_smoke'
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/infer_r04.py'),'--input',str(image),
                    '--output',str(destination),'--selection',str(ROOT/'selected_models.json'),'--device','0'],check=True)
    files=list(destination.glob('*/predictions.json'))
    if len(files)!=1:raise RuntimeError('Expected exactly one inference result')
    replay=read(files[0])
    expected=[p for p in sample['predictions'] if p['score']>=test['operating']['confidence']]
    actual=replay['predictions']
    if len(actual)!=len(expected):raise RuntimeError('Offline inference detection count differs')
    deltas=[]
    for a,b in zip(actual,expected):
        if a['class_id']!=b['class_id']:raise RuntimeError('Offline inference class/order differs')
        delta=max(abs(a['score']-b['score']),float(np.max(np.abs(np.array(a['bbox_xyxy'])-b['bbox_xyxy']))))
        deltas.append(delta)
    if max(deltas,default=0)>1e-6:raise RuntimeError('Offline coordinates/scores differ')
    report={'status':'PASS_OFFLINE_REPLAY_PARITY','utc':utc(),'arm':arm,'sample_id':sample['id'],
            'count':len(actual),'max_absolute_coordinate_or_score_difference':max(deltas,default=0),
            'selection_registry_sha256':sha(ROOT/'selected_models.json'),
            'frozen_selection_sha256':aggregate['selection_sha256'],'inference_script_sha256':sha(ROOT/'scripts/infer_r04.py'),
            'expected_samples_sha256':sha(Path(test['metrics_path']).with_name('samples.json')),
            'actual_predictions_sha256':sha(files[0]),'raw_output_not_packaged':str(destination),
            'scope':'One deployment-CLI replay of fixed predictions for software equivalence; no metrics, threshold search, training or model reselection; not a new evaluation or D455 observation.'}
    (ROOT/'reports/inference_equivalence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))
if __name__=='__main__':main()
