"""Attribute external Commons predictions; no ground-truth scoring or model selection."""
from pathlib import Path
from collections import Counter
import html
import json
import re
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def plain(s):return html.unescape(re.sub('<[^>]*>','',str(s)))


def main():
    source={r['id']:r for r in read(ROOT/'assets/commons_records.json')}
    folder=ROOT/'reports/qualitative_external24'
    index=read(folder/'index.json')
    if len(index['images'])!=24:raise ValueError('Expected fixed 24-image external set')
    cells=[]; entries=[]; notices=[]
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',13)
    for item in index['images']:
        identity=Path(item['image']).stem
        original=source[identity]
        target=Path(item['output'])
        prediction=read(target/'prediction.json')
        if prediction['input_sha256']!=original['sha256']:raise ValueError('External photo hash changed')
        attribution={'source_title':original['title'],'source_url':original['source_url'],
          'artist':plain(original['artist']),'credit':plain(original['credit']),
          'original_license':original['license'],'license_url':original['license_url'],
          'modification':'Added R03 model-predicted boxes and whole-board mask overlays. Predictions are not ground truth.',
          'derivative_license':'Same as the original photo license; source attribution retained',
          'quantitative_accuracy_measured':False}
        display_artist=attribution['artist']
        if identity=='commons_86471022':
            # The source explicitly lacks an author; retain that fact rather
            # than equating the upload account with the photographer.
            attribution.update(uploader='Ubahnverleih',author_status='Source page lacks author information',
                author_note_source='https://commons.wikimedia.org/wiki/File:Raspberry_Pi_Zero_W_vs_Zero.jpeg',
                author_note_checked_date='2026-09-28')
            display_artist='Author unspecified; uploader Ubahnverleih'
        (target/'ATTRIBUTION.json').write_text(json.dumps(attribution,ensure_ascii=False,indent=2),encoding='utf-8')
        counts=dict(Counter(p['class_name'] for p in prediction['component_detections']))
        entries.append({'id':identity,'output':target.name,'source':attribution,'boards':item['boards'],
                        'board_masks':item['board_masks'],'parts_by_predicted_class':counts})
        cell=Image.new('RGB',(500,340),'#f4f7fb')
        with Image.open(target/'annotated.png') as im:
            panel=ImageOps.contain(im.convert('RGB'),(484,260))
            cell.paste(panel,((500-panel.width)//2,30))
        draw=ImageDraw.Draw(cell)
        draw.text((8,4),identity,font=font,fill='#25344b')
        draw.text((8,295),f"예측: 보드 {item['boards']} / mask {item['board_masks']} / 부품 {item['components']}",font=font,fill='#25344b')
        draw.text((8,321),(display_artist+' | '+original['license'])[:66],font=small,fill='#65758a')
        cells.append(cell)
        notices.append(f"- **{identity}**: [{original['title']}]({original['source_url']}) · {display_artist} · [{original['license']}]({original['license_url']}). 변경: 모델 예측 상자·보드 마스크 표시. [예측 이미지]({target.name}/annotated.png), [상세 JSON]({target.name}/prediction.json).")
    sheet=Image.new('RGB',(2000,6*340+110),'#f4f7fb')
    draw=ImageDraw.Draw(sheet)
    draw.text((20,15),'학습 미사용 공개 사진 24장 · 모델 예측 예시',font=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',34),fill='#25344b')
    draw.text((20,63),'정답 라벨과 대조한 정확도 평가가 아닙니다. 보드 mask와 부품 bbox를 구분해 표시합니다.',font=font,fill='#9a3a4d')
    for i,cell in enumerate(cells):sheet.paste(cell,((i%4)*500,110+(i//4)*340))
    sheet.save(folder/'contact_sheet.jpg',quality=90)
    (folder/'qualitative_summary.json').write_text(json.dumps({'images':24,'quantitative_accuracy_measured':False,
       'source_photos_not_used_in_r03_finetuning':True,'physical_identity_independence_not_verified':True,
       'draft_annotations_used_as_ground_truth':False,'records':entries},ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# 공개 사진 24장 추론 예시','',
      'R03 학습에 쓰지 않은 Commons 사진에 검증셋에서 고정한 모델과 confidence를 적용했다. 정답 라벨과 대조한 mAP/recall 시험이 아니다. 표시된 검출 수를 정답 수나 성공률로 해석하지 않는다. 물리 보드의 독립성과 사전학습 자료 중복 여부는 미검증이다.','',
      '![24장 예측](contact_sheet.jpg)','',
      'assistant가 작성한 28 bbox 초안은 별도 검토 자료이며 이번 정확도 계산에 사용하지 않았다. 원본 사진의 작성자·라이선스와 예측 표시 변경 내역은 각 폴더 ATTRIBUTION.json에 있다.','']+notices
    (folder/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':main()
