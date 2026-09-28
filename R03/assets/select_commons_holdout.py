"""Freeze 24 visually screened photos before creating labels or predictions."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
SELECTION={
 'pi4':[88249711,88249714,80430835,122664660,114604365,97431420],
 'pi5':[145630798,144264832,144264831,145417518,149565043,144264833],
 'zero':[145417507,178528806,153836405,112083087,86471022,67172993],
 'pi3':[112083120,67173800,75417829,67384198,69775241,48775492]}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    all_path=ROOT/'commons_records_all.json'
    if not all_path.exists():all_path.write_bytes((ROOT/'commons_records.json').read_bytes())
    all_rows=json.loads(all_path.read_text(encoding='utf-8'))
    selected_ids={number for values in SELECTION.values() for number in values}
    rows=[r for r in all_rows if r['pageid'] in selected_ids]
    assert len(rows)==24
    old=Path('C:/Users/hkjun/Documents/mcu-vision/data/manifests/raspberry_pi_sbc.sources.jsonl')
    old_ids={json.loads(line)['source_page_id'] for line in old.read_text(encoding='utf-8').splitlines() if line.strip()}
    assert not selected_ids.intersection(old_ids)
    for row in rows:
        assert row['nearest_iotkits_phash_distance']>8
        assert row['objects'] is None
        assert sha(Path(row['image']))==row['sha256']
        row['holdout_candidate_status']='VISUALLY_SCREENED_PENDING_MANUAL_BBOX'
        row['visual_screening']='Whole Raspberry Pi board(s) visible in source photo. Nearest-IoTKIT contact sheets inspected; detailed crop-level source-independence audit remains open.'
        row['old_commons_collection_pageid_overlap']=False
        row['objects']=None
    write(ROOT/'commons_records.json',rows)
    folder=ROOT/'commons_holdout'
    decisions=[]
    for row in all_rows:
        status='selected_24' if row['pageid'] in selected_ids else 'reserve_not_in_frozen_24'
        reason='Diverse viewpoints/backgrounds with visible board; four filename/query families balanced six each'
        if row['nearest_iotkits_phash_distance']<=8:status='excluded';reason='Conservative pHash <=8 source-overlap screen; not proof of same image'
        elif row['pageid'] in {112083117,112083119}:status='excluded';reason='Closed product packaging, no board visible'
        elif row['pageid']==159094209:status='excluded';reason='Board obscured by case on network switch'
        elif row['pageid']==67384195:status='excluded';reason='Tight component crop, whole-board evaluation ambiguous'
        elif row['pageid']==162901180:status='excluded';reason='Similar stock-view appearance to nearest IoTKIT sample; excluded conservatively pending detailed crop audit'
        decisions.append({'id':row['id'],'status':status,'reason':reason})
    write(folder/'selection_review.json',decisions)
    frozen={'schema':'commons-unlabelled-holdout-candidates-v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'records_file':'../commons_records.json','records_sha256':sha(ROOT/'commons_records.json'),
        'image_count':24,'source_page_ids':sorted(selected_ids),'generation_query_counts':dict(Counter(r['model_generation_candidate'] for r in rows)),
        'photographer_group_count':len({r['group_id'] for r in rows}),
        'annotation_status':'UNLABELLED; parent must manually annotate native photos before metrics',
        'predictions_used_for_annotation_or_selection':False,
        'overlap_screen':'No exact IoTKIT hash match; nearest whole-image dihedral pHash distance >8. Distinct from five previous Commons page IDs.',
        'physical_or_crop_source_independence':'NOT_FULLY_VERIFIED; internet-derived IoTKIT origins are incomplete',
        'assets':[{'id':r['id'],'image':r['image'],'sha256':r['sha256'],'license':r['license'],'source_url':r['source_url']} for r in rows]}
    write(folder/'frozen_holdout_candidates.json',frozen)
    canvas=Image.new('RGB',(1600,1200),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        x=(i%4)*400;y=(i//4)*200
        with Image.open(row['image']) as im:
            im=im.convert('RGB');im.thumbnail((385,165));canvas.paste(im,(x,y+30))
        draw.text((x+5,y+3),row['id']+' '+row['model_generation_candidate'],fill='black')
        draw.text((x+5,y+16),'IoT pHash distance '+str(row['nearest_iotkits_phash_distance']),fill='black')
    canvas.save(folder/'selected_24_contactsheet.jpg',quality=92)
    print(json.dumps({'selected_images':len(rows),'photographer_groups':frozen['photographer_group_count'],'records_sha256':frozen['records_sha256'],'annotations_created':False,'predictions_used':False},indent=2))

if __name__=='__main__':main()
