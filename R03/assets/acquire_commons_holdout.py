"""Download CC-licensed Commons photos and freeze UNLABELLED holdout candidates.
No detector/model outputs are used. Whole-image perceptual hashes are screening
candidates only; they do not prove physical-board or source independence.
"""
import hashlib,html,json,re,time,urllib.request
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'commons_holdout'
IDS={
 'pi4':[88249711,88249714,88249722,80430835,122664660,114604365,80140656,97431420,159094209],
 'pi5':[138594630,138594632,138594631,138514068,145630798,144264842,144264832,162901180,144264831,145417518,149565043,144264841,144264843,144264833],
 'zero':[111935985,111935983,111935982,145417507,178528806,69660497,67172993,178528459,153836402,153836405,86470705,112083087,112083094,67172980,86471022,86471024],
 'pi3':[112083117,112083119,112083120,112083121,67173800,78854507,75417829,67384198,67384195,69775241,48775492,47246408,112305535]}

def plain(text):return html.unescape(re.sub('<[^>]+>','',text or '')).strip()
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def phash(image):
    gray=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),np.float32)
    low=cv2.dct(gray)[:8,:8]
    bits=low>np.median(low)
    return int.from_bytes(np.packbits(bits.flatten()).tobytes(),'big')

def acquire(item):
    model,page=item
    info=page['imageinfo'][0];meta=info.get('extmetadata',{})
    license_name=plain(meta.get('LicenseShortName',{}).get('value',''))
    if not (license_name.startswith('CC BY') or license_name in {'CC0','Public domain'}):
        return {'excluded_pageid':page['pageid'],'reason':'license_not_allowlisted','license':license_name}
    # Commons-generated thumbnail preserves source composition while limiting bytes.
    url=info.get('thumburl',info['url'])
    extension='.png' if '.png' in url.lower() else '.jpg'
    path=FOLDER/'images'/f"commons_{page['pageid']}{extension}"
    if not path.exists():
        req=urllib.request.Request(url,headers={'User-Agent':'PCBBoardDatasetResearch/1.0'})
        try:
            with urllib.request.urlopen(req,timeout=40) as response:data=response.read(25_000_001)
            if len(data)>25_000_000:raise RuntimeError('Per-file size cap')
            path.write_bytes(data)
        except Exception as e:
            return {'excluded_pageid':page['pageid'],'reason':'download_error','error':str(e)[:200]}
    with Image.open(path) as image:
        image.load();width,height=image.size
        hashes=[phash(image.rotate(degrees,expand=True)) for degrees in (0,90,180,270)]
        hashes += [phash(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(degrees,expand=True)) for degrees in (0,90,180,270)]
    artist=plain(meta.get('Artist',{}).get('value','UNKNOWN'))
    return {'source':'wikimedia_commons','id':f"commons_{page['pageid']}",'pageid':page['pageid'],
        'title':page['title'],'source_url':info.get('descriptionurl'),
        'original_url':info['url'],'download_url':url,'image':str(path.resolve()),
        'width':width,'height':height,'sha256':sha(path.read_bytes()),
        'original_publisher_sha1':info.get('sha1'),'download_is_commons_thumbnail':url!=info['url'],
        'license':license_name,'license_url':meta.get('LicenseUrl',{}).get('value'),
        'artist':artist,'credit':plain(meta.get('Credit',{}).get('value')),
        'copyrighted':meta.get('Copyrighted',{}).get('value'),
        'attribution_required':meta.get('AttributionRequired',{}).get('value'),
        'original_date':plain(meta.get('DateTimeOriginal',{}).get('value')),
        'description':plain(meta.get('ImageDescription',{}).get('value')),
        'per_file_origin':'Commons file page attribution and license metadata recorded',
        'model_generation_candidate':model,'group_id':'commons_author_'+sha(artist.encode())[:16],
        'group_basis':'Conservative photographer grouping; physical board/session identity not independently verified',
        'originalsplit':None,'objects':None,'annotation_status':'UNLABELLED_DO_NOT_USE_AS_GROUND_TRUTH',
        'model_predictions_used':False,'phash_dihedral_hex':[f'{h:016x}' for h in hashes]}

def main():
    FOLDER.mkdir(exist_ok=True);(FOLDER/'images').mkdir(exist_ok=True)
    pages=[]
    for model,ids in IDS.items():
        discovery=json.loads((FOLDER/f'discovery_{model}.json').read_text(encoding='utf-8'))
        mapping={int(p['pageid']):p for p in discovery['query']['pages'].values()}
        for number in ids:
            if number in mapping:pages.append((model,mapping[number]))
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(acquire,pages))
    records=[r for r in results if 'image' in r]
    excluded=[r for r in results if 'image' not in r]
    iot=json.loads((ROOT/'board_records.json').read_text(encoding='utf-8'))
    reference=[]
    for row in iot:
        with Image.open(row['image']) as image:reference.append((row['id'],phash(image),row['sha256'],row['image']))
    for row in records:
        hashes=[int(h,16) for h in row['phash_dihedral_hex']]
        distances=[(min((h^ref[1]).bit_count() for h in hashes),ref) for ref in reference]
        distance,nearest=min(distances,key=lambda x:x[0])
        row['nearest_iotkits_phash_distance']=distance
        row['nearest_iotkits_id']=nearest[0]
        row['nearest_iotkits_image']=nearest[3]
        row['exact_iotkits_hash_overlap']=any(row['sha256']==ref[2] for ref in reference)
        row['holdout_candidate_status']='EXCLUDE_NEAR_DUPLICATE_SCREEN' if distance<=8 or row['exact_iotkits_hash_overlap'] else 'PENDING_VISUAL_REVIEW_AND_MANUAL_BBOX'
    # Within Commons, group all near-duplicate candidates before manual selection.
    pairs=[]
    for i,left in enumerate(records):
        lh=[int(h,16) for h in left['phash_dihedral_hex']]
        for right in records[i+1:]:
            rh=int(right['phash_dihedral_hex'][0],16)
            distance=min((h^rh).bit_count() for h in lh)
            if distance<=8:pairs.append({'left':left['id'],'right':right['id'],'distance':distance})
    write(ROOT/'commons_records.json',records)
    write(FOLDER/'screening_summary.json',{'frozen_before_model_predictions_utc':datetime.now(timezone.utc).isoformat(),
        'images':len(records),'status_counts':dict(Counter(r['holdout_candidate_status'] for r in records)),
        'license_counts':dict(Counter(r['license'] for r in records)),
        'generation_counts':dict(Counter(r['model_generation_candidate'] for r in records)),
        'within_commons_near_duplicate_pairs':pairs,'download_exclusions':excluded,
        'phash_method':'PIL LANCZOS gray32 + cv2 DCT 8x8 median 64bit; 8 dihedral transforms on Commons query; Hamming <=8 screen',
        'reference_images':len(reference),'annotations_created':False,'model_predictions_used':False,
        'limitations':['pHash alone cannot certify independent photography; crop/edits may evade it','Original Commons photographer/license metadata must accompany redistributions','Images remain unlabeled until visual manual box review','Generation labels here come from discovery queries and filenames, not verified hardware inspection']})
    # Side-by-side nearest-source pairs for manual leakage adjudication.
    for page_number,start in enumerate(range(0,len(records),12),1):
        chunk=records[start:start+12];canvas=Image.new('RGB',(1280,180*len(chunk)),(255,255,255));draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(chunk):
            y=j*180
            for x,path in [(0,row['image']),(400,row['nearest_iotkits_image'])]:
                with Image.open(path) as image:
                    image=image.convert('RGB');image.thumbnail((380,158));canvas.paste(image,(x,y+20))
            draw.text((800,y+15),row['id']+' '+row['model_generation_candidate'],fill='black')
            draw.text((800,y+40),'IoT distance '+str(row['nearest_iotkits_phash_distance']),fill='black')
            draw.text((800,y+65),row['holdout_candidate_status'][:45],fill='black')
            draw.text((800,y+90),row['nearest_iotkits_id'],fill='black')
        canvas.save(FOLDER/f'nearest_source_contactsheet_{page_number:02d}.jpg',quality=85)
    print(json.dumps({'images':len(records),'status':dict(Counter(r['holdout_candidate_status'] for r in records)),'records':str((ROOT/'commons_records.json').resolve())},indent=2))

if __name__=='__main__':main()
