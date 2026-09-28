"""Check materialized R03 board views against native manifest geometry and global groups."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path

from prepare_board_data import ROOT,digest,label_text,make_edges,save


def main():
    manifests={task:json.loads((ROOT/'data'/name/'manifest.json').read_text(encoding='utf-8'))
               for task,name in [('detect','board_detect_v2'),('segment','board_segment_v2')]}
    detect=manifests['detect']['records']; segment=manifests['segment']['records']
    fingerprints=json.loads((ROOT/'reports'/'board_fingerprints_v2.json').read_text(encoding='utf-8'))
    assert len(fingerprints)==5983
    assert len({r['sha256'] for r in detect})==len(detect)
    def verify(r):
        assert Path(r['image']).exists() and Path(r['label']).exists(),r['id']
        assert digest(r['image'])==r['sha256'],r['id']
        assert Path(r['label']).read_text(encoding='utf-8')==label_text(r,r['task']),r['id']
        return True
    with ThreadPoolExecutor(max_workers=6) as pool: assert all(pool.map(verify,detect+segment))
    edges,counts=make_edges(detect)
    for a,b in edges: assert detect[a]['split']==detect[b]['split'],(detect[a]['id'],detect[b]['id'])
    hashes={};groups={}
    for row in detect:
        for key,tracking in [(row['sha256'],hashes),(row['group_id'],groups)]:
            assert key not in tracking or tracking[key]==row['split'],row['id']
            tracking[key]=row['split']
        if row['source']=='micro_pcb': assert bool(row['objects'])==(row['source_model_code'] in 'GHM')
        assert len(row['objects'])==len([line for line in Path(row['label']).read_text().splitlines() if line.strip()])
    by_id={r['id']:r for r in detect}
    for row in segment:
        original=by_id[row['id']]
        assert row['source']=='iotkits_v1'
        assert (row['sha256'],row['split'],row['group_id'],row['objects'])==(original['sha256'],original['split'],original['group_id'],original['objects'])
        assert all(o['annotation_type']=='native_polygon' and len(o['polygon_parts_xy'])==1 for o in row['objects'])
    expected={r['id'] for r in detect if r['source']=='iotkits_v1' and all(o.get('polygon_parts_xy') for o in r['objects'])}
    assert {r['id'] for r in segment}==expected
    report={'status':'PASS_STRUCTURAL_AND_NATIVE_GEOMETRY_BINDING',
        'detect_images':len(detect),'segment_images':len(segment),
        'actual_materialized_sha256_verified':len(detect)+len(segment),
        'labels_exactly_match_native_geometry_normalization':True,
        'remaining_cross_split_sha256_pairs':0,'remaining_cross_split_source_family_or_phash_edges':0,
        'retained_edge_counts':counts,'micro_positive_codes':dict(Counter(r['source_model_code'] for r in detect if r['source']=='micro_pcb' and r['objects'])),
        'same_global_assignment_for_detect_and_segment':True,
        'annotation_scope':'Source provided geometry; reviewed per-code binary class override; sampling is not exhaustive per-image semantic review',
        'not_accuracy_or_d455_validation':True}
    save(ROOT/'reports'/'board_v2_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
