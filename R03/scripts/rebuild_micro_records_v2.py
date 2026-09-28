"""Build R03-only binary board labels from reviewed code G/H/M, not erroneous README."""
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('C:/Users/hkjun/Documents/mcu-vision/data/raw/curated/micro_pcb_images')
README_MODELS={
 'A':'Raspberry Pi A+','B':'Arduino Mega 2560 (Blue)','C':'Arduino Mega 2560 (Black)',
 'D':'Arduino Mega 2560 (Black and Yellow)','E':'Arduino Due','F':'Beaglebone Black',
 'G':'Arduino Uno (Green)','H':'Raspberry Pi 3 B+','I':'Raspberry Pi 1 B+',
 'J':'Arduino Uno Camera Shield','K':'Arduino Uno (Black)','L':'Arduino Uno WiFi Shield','M':'Arduino Leonardo'}
OBSERVED={
 'A':'ELEGOO MEGA2560 text; blue Mega-shaped board; no Raspberry Pi identity',
 'B':'MEGA text on black long microcontroller board',
 'C':'MEGA text; yellow headers on black long microcontroller board',
 'D':'Arduino DUE text and logo on blue board',
 'E':'BeagleBone text and dual long expansion headers',
 'F':'ARDUINO UNO text and Arduino infinity logo',
 'G':'Raspberry Pi logo on green SBC with Ethernet and four USB ports',
 'H':'Raspberry Pi logo on green SBC with Ethernet and four USB ports',
 'I':'ESP8266 Wireless Module UNO text on Arduino-shaped board',
 'J':'ELEGOO UNO R3 text on black board',
 'K':'WiFi/ESP8266 module on blue UNO-shaped board; no Raspberry Pi identity',
 'L':'ARDUINO LEONARDO text and Arduino infinity logo',
 'M':'Raspberry Pi logo on compact green SBC with single USB and no Ethernet'}

def read_csv(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 rows=[];by_code=defaultdict(lambda:defaultdict(list))
 for serialsplit in ['train','test']:
  sizes={r['Image']:r for r in read_csv(RAW/f'{serialsplit}_sizes.csv')}
  boxes={r['Image']:r for r in read_csv(RAW/f'{serialsplit}_bboxes.csv')}
  for filename,s in sizes.items():
   b=boxes[filename];x,y,w,h=(int(b[k]) for k in ['Left','Top','Width','Height'])
   by_code[filename[0]][filename[:4]].append({'filename':filename,'image':RAW/f'{serialsplit}_coded'/f'{serialsplit}_coded'/filename,
    'source_serial_split':serialsplit,'width':int(s['Width']),'height':int(s['Height']),'bbox_xyxy':[x,y,x+w,y+h]})
 for code,groups in sorted(by_code.items()):
  positive=code in 'GHM'
  conditions=sorted(groups) if positive else sorted(random.Random(42+ord(code)).sample(sorted(groups),20))
  for condition in conditions:
   source_rows=sorted(groups[condition],key=lambda r:r['filename'])
   assert len(source_rows)==5
   for r in source_rows:
    rows.append({'id':'micro_v2_'+Path(r['filename']).stem,'source':'micro_pcb','image':str(r['image']),
     'source_image':str(r['image']),'label':None,'source_model':f'micro_code_{code}_'+('raspberry_pi_sbc' if positive else 'other_board'),
     'source_model_code':code,'source_model_label':README_MODELS[code],
     'visual_binary_class':'raspberry_pi_sbc' if positive else 'other_board',
     'source_metadata_conflict':(positive != (code in 'AHI')),
     'source_metadata_override':'R03 binary image-identity review; native README names not used as class ground truth',
     'visual_evidence':OBSERVED[code],
     'class_review_evidence':'reports/contact_sheets/micro_ALL_CODES_index.json',
     'class_review_scope':'Three different conditions per source code independently reviewed by parent and audit agents; source-code binary mapping applied; not exhaustive per-image adjudication',
     'group_id':'micro_condition_'+condition,'condition_group_id':condition,'split':None,'is_rpi':positive,
     'bbox_xyxy':r['bbox_xyxy'],'bbox_semantics':'raspberry_pi_whole_board' if positive else 'source_non_rpi_board_bbox_NOT_an_rpi_target',
     'width':r['width'],'height':r['height'],'source_capture_serial':int(r['filename'][4]),
     'source_serial_split':r['source_serial_split'],'physical_specimen_independence_verified':False,
     'existing_split_immutable':False,'source_url':'https://www.kaggle.com/datasets/frettapper/micropcb-images','license':'CC BY 4.0',
     'annotation_status':'native_bbox_with_reviewed_binary_source_code_override'})
 cached={r['sha256']:r['phash64'] for r in json.loads((ROOT/'reports'/'board_fingerprints.json').read_text(encoding='utf-8'))}
 with ThreadPoolExecutor(max_workers=6) as pool: hashes=list(pool.map(lambda r:sha(Path(r['image'])),rows))
 for r,h in zip(rows,hashes):
  r['sha256']=h
  if h in cached:r['phash64_from_hash_verified_prior_audit']=cached[h]
 review={'status':'REVIEWED_R03_BINARY_OVERRIDE','raspberry_pi_codes':['G','H','M'],
  'reviewers':'Parent agent and independent local audit agent; both visually inspected 39 cropped source photographs',
  'coverage':'3 different conditions for each of 13 source codes; not all images individually adjudicated',
  'source_readme':str(RAW/'README.txt'),'source_readme_sha256':sha(RAW/'README.txt'),
  'code_map':[{'code':code,'source_model_label':README_MODELS[code],'is_raspberry_pi':code in 'GHM','observed_evidence':OBSERVED[code]} for code in README_MODELS],
  'split_reset_reason':'Earlier A/H/I binary ontology was visually invalid; no old split is considered a valid Raspberry Pi reference. New global source-family + SHA + pHash grouping uses seed42 70/15/15.',
  'legacy_model_status':'QUARANTINED_CLASS_MAPPING_CONFLICT; must not report historical RPi AP as a valid accuracy baseline',
  'source_images_original_repository_changed':False}
 (ROOT/'reports'/'micro_binary_class_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
 content={'schema':'micro-pcb-expansion-records-v2','semantic_mapping_status':review,
  'source_url':'https://www.kaggle.com/datasets/frettapper/micropcb-images','license':'CC BY 4.0',
  'counts':{'images':len(rows),'positive':sum(r['is_rpi'] for r in rows),'negative':sum(not r['is_rpi'] for r in rows),'by_code':dict(Counter(r['source_model_code'] for r in rows))},'records':rows}
 (ROOT/'micro_records_v2.json').write_text(json.dumps(content,indent=2),encoding='utf-8')
 text=['# Micro-PCB R03 binary class correction','',
  'Native README names contradict visible photos. Both reviewers inspected three different conditions for each code (39 images). Raspberry Pi identity is visible for G/H/M. Exact Raspberry Pi variants were not required or asserted for this single-class task. The override applies to this R03 workspace only.','',
  '| Code | Native README claim | Reviewed RPi? | Visible evidence |','|---|---|---|---|']
 text += [f'| {c} | {README_MODELS[c]} | {"Yes" if c in "GHM" else "No"} | {OBSERVED[c]} |' for c in README_MODELS]
 text += ['',review['coverage'], '', review['split_reset_reason'], '',review['legacy_model_status'], '',
  'Review evidence: `contact_sheets/micro_ALL_CODES_1.png`, `_2.png`, `_3.png`; exact source coordinates/hashes in `micro_ALL_CODES_index.json`.',
  '',f'Prepared records: {content["counts"]}. Other-board native boxes remain provenance only and do not become RPi labels.']
 (ROOT/'reports'/'micro_binary_class_review.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
 print(json.dumps(content['counts']),flush=True)

if __name__=='__main__':main()
