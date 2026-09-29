"""Check the packaged inference/display path; this is not an accuracy benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from live_view import Detector, View, HEADER, ROOT, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='0')
    parser.add_argument('--output',type=Path,default=ROOT/'sessions/verification.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    registry=json.loads((ROOT/'selected_models.json').read_text())
    weights=ROOT/registry['local_prototype']['checkpoint']
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==registry['local_prototype']['sha256']
    image=cv2.imread(str(ROOT/'evaluation/functional-input.png'))
    assert image is not None
    detector=Detector(args.device)
    result=detector.predict(image)
    baseline=json.loads((ROOT/'evaluation/r04-baseline.json').read_text())
    assert len(result['parts'])==len(baseline['parts'])
    errors=[]
    for actual,expected in zip(result['parts'],baseline['parts']):
        assert actual['class_id']==expected['class_id']
        assert abs(actual['score']-expected['score'])<1e-4
        error=float(np.abs(np.array(actual['bbox_xyxy'])-expected['bbox_xyxy']).max())
        assert error<.01
        errors.append(error)
    view=View()
    combined=view.describe(result)
    assert sum(combined['counts'].values())==len(combined['parts'])
    assert combined['board_counts']=={'raspberry_pi_5':1,'raspberry_pi_4':1}
    assert {p['source'] for p in combined['parts']}=={'R04','local_board_parts_prototype'}
    stats={'camera_fps':0,'inference_fps':0}
    view.render(image,result,stats)
    for selected_class in [2,4,5,6,7,8,9]:
        target=next(p for p in combined['parts'] if p['class_id']==selected_class)
        x1,y1,x2,y2=target['bbox_xyxy']
        view.click(cv2.EVENT_LBUTTONDOWN,int((x1+x2)/2),int((y1+y2)/2)+HEADER,0,None)
        assert view.selection[0]==selected_class
        view.render(image,result,stats)
    view.expanded=False
    reference=view.describe(result)
    assert reference['counts']=={'ic':2}
    assert reference['boards']==result['boards']
    view.render(image,result,stats)
    view.diagnostic=True
    assert len(view.describe(result)['parts'])>len(reference['parts'])
    view.render(image,result,stats)
    view.expanded=True
    view.diagnostic=False
    blank=detector.predict(np.zeros_like(image))
    empty=view.describe(blank)
    assert not empty['parts'] and not empty['boards']
    view.render(np.zeros_like(image),blank,stats)
    shifted=cv2.warpAffine(image,np.float32([[1,0,-160],[0,1,0]]),(image.shape[1],image.shape[0]))
    shifted_result=detector.detail.predict(shifted,imgsz=1024,conf=.6,iou=.5,quantize=None,device=args.device,verbose=False)[0]
    shift_errors=[]
    for part in result['additional_parts']+result['detail_boards']:
        if part['score']<.6:
            continue
        expected=np.array(part['bbox_xyxy'])+[-160,0,-160,0]
        candidates=[b.numpy() for b,c in zip(shifted_result.boxes.xyxy.cpu(),shifted_result.boxes.cls.cpu()) if int(c)==part['model_class_id']]
        assert candidates
        shift_errors.append(float(min(np.abs(b-expected).max() for b in candidates)))
    assert max(shift_errors)<20
    report={
        'scope':'Functional and synthetic translation checks, not independent accuracy evaluation',
        'passed':True,'checkpoint_sha256':registry['local_prototype']['sha256'],
        'input_sha256':hashlib.sha256((ROOT/'evaluation/functional-input.png').read_bytes()).hexdigest(),
        'counts':combined['counts'],'board_counts':combined['board_counts'],
        'r04_max_coordinate_difference':max(errors),'blank_detections':0,
        'translation_x_pixels':-160,'translation_box_max_error_pixels':max(shift_errors),
        'click_classes_checked':[2,4,5,6,7,8,9],'reference_and_diagnostic_modes_checked':True,
        'camera_opened':False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
