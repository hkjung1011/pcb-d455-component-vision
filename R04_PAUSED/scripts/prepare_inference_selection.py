"""Export only the already frozen validation selections as a portable registry."""
from pathlib import Path
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    suite=ROOT/'reports/evaluation_suite'
    aggregate=json.loads((suite/'aggregate.json').read_text(encoding='utf-8'))
    if aggregate['status']!='COMPLETED_DEVELOPMENT_HOLDOUT_NOT_D455_VALIDATED':
        raise RuntimeError('Finish the frozen evaluation suite first')
    if sha(suite/'selections_frozen.json')!=aggregate['selection_sha256']:
        raise RuntimeError('Frozen selections changed')
    registry={'schema':'r04-inference-selection-v1','recommended_arm':aggregate['recommended_arm'],
              'selection_sha256':aggregate['selection_sha256'],'selected_on':'validation_only',
              'd455_verified':False,'component_instance_masks_trained':False,'arms':{}}
    for arm,payload in aggregate['arms'].items():
        selected=payload['selection'];candidate=selected['candidate']
        source=Path(candidate['weights'])
        if sha(source)!=candidate['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
        registry['arms'][arm]={'checkpoint':source.relative_to(ROOT).as_posix(),'sha256':candidate['checkpoint_sha256'],
            'pipeline':candidate['pipeline'],'confidence':selected['operating_confidence'],
            'epoch':candidate['epoch'],'optimizer_calls':candidate['optimizer_calls'],
            'imgsz':1024,'tile_overlap':.2,'tile_nms_iou':.5}
    target=ROOT/'selected_models.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=registry:
        raise RuntimeError('Existing registry differs; do not overwrite')
    target.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Portable registry: {target}; recommended by validation={registry["recommended_arm"]}')
if __name__=='__main__':main()
