"""Derive source/board-family tables from frozen predictions; no model selection."""
from collections import defaultdict
import json
from pathlib import Path
from native_evaluate import coco_score, operate

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    suite=read(ROOT/'reports/evaluation_suite/aggregate.json')
    results={}
    for mid,folder in [('board_yolo11n','board_detect_v2'),('board_yolo11n_seg','board_segment_v2')]:
        test=suite['models'][mid]['tests']['general_test']
        evaluation=Path(test['metrics_path']).parent
        manifest=read(ROOT/'data'/folder/'manifest.json')
        native=read(evaluation/'native_ground_truth.json')
        predictions=read(evaluation/'bbox_predictions.json')
        samples=read(evaluation/'samples.json')
        rows={r['id']:r for r in manifest['records'] if r['split']=='test'}
        if len(rows)!=len(samples):raise ValueError('Test image count mismatch')
        groups=defaultdict(list)
        for index,sample in enumerate(samples,1):
            row=rows[sample['id']]
            # Source subsets include their negative photos. Variant subsets are positive only.
            groups['source:'+row['source']].append(index)
            if row['objects']:
                variant=row.get('source_model') or row.get('source_model_code') or 'unknown'
                groups['positive_variant:'+row['source']+':'+str(variant)].append(index)
        output={}
        for name,indices in groups.items():
            keep=set(indices)
            gt=dict(native,images=[im for im in native['images'] if im['id'] in keep],annotations=[a for a in native['annotations'] if a['image_id'] in keep])
            preds=[p for p in predictions if p['image_id'] in keep]
            operating=operate([samples[i-1] for i in indices],test['operating']['confidence'],1)
            operating.pop('per_image',None)
            output[name]={'images':len(indices),'instances':len(gt['annotations']),'bbox':coco_score(gt,preds),'operating':operating}
        results[mid]={'fixed_confidence_from_val':test['operating']['confidence'],'subsets':output}
    out={'scope':'Descriptive post-hoc test subsets from fixed predictions, not model selection. Variant groups contain positive images only; see source subsets for negatives. Source model names are source metadata, micro exact variant not independently established.','models':results}
    (ROOT/'reports/board_subsets.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
