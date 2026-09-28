"""Korean R03 figures from audited manifests and recorded model architectures.

No training/inference is run. Quarantined legacy board counts and AP are never
used. Missing corrected manifests or architecture counts remain pending.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
BG='#f5f7fb';INK='#1d2d44';MUTED='#516478';BLUE='#2468ba';TEAL='#008875';ORANGE='#db8b22';RED='#b34453'
NAMES=['resistor','capacitor','ic','connector']
KOREAN=['저항','콘덴서','IC','커넥터']

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def configure():
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    else:plt.rcParams['font.family']='sans-serif'
    plt.rcParams.update({'axes.unicode_minus':False,'svg.fonttype':'none','font.size':12,'axes.titleweight':'bold','figure.facecolor':BG,'savefig.facecolor':BG})

def save(fig,out,name,provenance):
    for suffix in ['png','svg']:fig.savefig(out/f'{name}.{suffix}',dpi=180,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)
    provenance['figures'].append(name)

def node(ax,x,y,w,h,title,body='',color=BLUE,title_size=12,body_size=10):
    box=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.013',facecolor='white',edgecolor=color,linewidth=1.7)
    ax.add_patch(box)
    ax.text(x+w/2,y+h*.72 if body else y+h/2,title,ha='center',va='center',fontsize=title_size,color=color,fontweight='bold')
    if body:ax.text(x+w/2,y+h*.33,body,ha='center',va='center',fontsize=body_size,color=INK,linespacing=1.45)

def arrow(ax,start,end,color=MUTED,style='-'):ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,linewidth=1.4,color=color,linestyle=style,connectionstyle='arc3'))
def title(fig,heading,subheading):
    fig.text(.045,.965,heading,fontsize=23,fontweight='bold',color=INK,va='top')
    fig.text(.045,.916,subheading,fontsize=12,color=MUTED,va='top')

def architecture_files(root):
    result={}
    for path in sorted((root/'runs').glob('*/architecture.json')):
        try:
            a=read(path);names=a.get('classes',[]);task=a.get('task');modelid=a.get('model_id',path.parent.name)
            if names==NAMES and task=='detect':key='parts_s' if a.get('yaml',{}).get('scale')=='s' or '11s' in modelid else 'parts_n'
            elif names==['raspberry_pi_sbc'] and task=='detect':key='board_detect'
            elif names==['raspberry_pi_sbc'] and task=='segment':key='board_segment'
            else:continue
            result[key]=(path,a)
        except (ValueError,OSError):continue
    return result

def corrected_board_manifests(root,explicit_detect=None,explicit_segment=None):
    """Require positive eligibility and exact G/H/M corrected audit evidence."""
    candidates=[]
    for explicit in [explicit_detect,explicit_segment]:
        if explicit:candidates.append(Path(explicit))
    candidates+=list((root/'data').glob('*/manifest.json'))
    selected={};rejections=[]
    for path in sorted(set(candidates)):
        if not path.is_file():continue
        try:m=read(path)
        except (ValueError,OSError):continue
        if m.get('task') not in {'detect','segment'} or m.get('names')!=['raspberry_pi_sbc']:continue
        if m.get('training_eligible') is not True:
            rejections.append({'path':str(path),'reason':'training_eligible is not true'});continue
        review=m.get('split_protocol',{}).get('source_class_review',{})
        ghm=sorted(review.get('raspberry_pi_codes',[]))==['G','H','M']
        corrected=review.get('status')=='REVIEWED_R03_BINARY_OVERRIDE'
        # Segmentation may carry the global corrected split audit rather than micro labels.
        if not (ghm and corrected):
            rejections.append({'path':str(path),'reason':'Corrected G/H/M mapping audit not found; pass audited manifest with recorded correction evidence'});continue
        selected[m['task']]=(path,m)
    return selected,rejections

def pipeline_figure(out,p):
    fig,ax=plt.subplots(figsize=(18,9.6));ax.set(xlim=(-.012,1.012),ylim=(0,1));ax.axis('off')
    title(fig,'보드와 부품을 나누어 학습하는 구조','현재 모델의 역할과 추론 연결 설계 · D455 실제 촬영 성능과 통합 실측은 별도 검증 대상')
    ax.set_position([.035,.10,.93,.75])
    node(ax,.005,.48,.135,.22,'RGB 사진','공개 보드 사진\n향후 D455 입력',title_size=14,body_size=12)
    node(ax,.19,.64,.18,.20,'① YOLO11n 검출','보드 1종 bbox\n수정 micro + IoTKITs',body_size=11)
    node(ax,.425,.64,.14,.20,'보드 ROI','관심 보드 영역\n좌표 보존',body_size=11)
    node(ax,.62,.64,.145,.20,'타일 분할','1024 px / 20% 겹침\n원본 좌표로 복원',body_size=11)
    node(ax,.815,.61,.17,.26,'③·④ 부품 검출','YOLO11n / YOLO11s\n저항·콘덴서·IC·커넥터\n출력: 부품별 bbox',color=TEAL,body_size=11)
    arrow(ax,(.14,.64),(.19,.74));arrow(ax,(.37,.74),(.425,.74));arrow(ax,(.565,.74),(.62,.74));arrow(ax,(.765,.74),(.815,.74))
    node(ax,.19,.25,.18,.20,'② YOLO11n-seg','IoTKITs native polygon\n보드 전체 외곽 학습',color=ORANGE,body_size=11)
    node(ax,.44,.25,.275,.20,'출력: 보드 영역 mask','보드 경계를 픽셀 단위로 표시\n부품별 mask는 이 모델의 출력이 아님',color=ORANGE,body_size=11)
    arrow(ax,(.14,.54),(.19,.35),ORANGE);arrow(ax,(.37,.35),(.44,.35),ORANGE)
    ax.text(.81,.38,'공통 후처리\n원본 좌표 결합\n타일 간 중복 검출 정리',fontsize=12,color=MUTED,ha='center',va='center')
    ax.text(.5,.115,'보드 mask 학습과 부품 mask 학습은 다릅니다. 이번 신규 4종 부품 모델은 bbox 검출입니다.',ha='center',fontsize=13,fontweight='bold',color=RED)
    ax.text(.5,.055,'R02의 IC·전해콘덴서·커넥터 semantic-CC 모델은 별도 보조 실험이며, 신규 4종 부품 mask 모델로 취급하지 않습니다.',ha='center',fontsize=10.5,color=MUTED)
    save(fig,out,'01_task_pipeline',p)

def model_figure(out,p,architectures,root):
    fig,axes=plt.subplots(2,2,figsize=(18,12));fig.subplots_adjust(left=.04,right=.97,bottom=.07,top=.84,hspace=.19,wspace=.10)
    title(fig,'모델별 구조와 출력','공통 YOLO11 backbone·다중 스케일 feature fusion / 파라미터 수는 architecture.json 실측 기록, 미생성 모델은 PENDING 표시')
    specs=[('board_detect','① 보드 위치 · YOLO11n','detect',1,'n',BLUE),('board_segment','② 보드 외곽 · YOLO11n-seg','segment',1,'n',ORANGE),
           ('parts_n','③ 부품 4종 · YOLO11n','detect',4,'n',TEAL),('parts_s','④ 부품 4종 · YOLO11s','detect',4,'s',TEAL)]
    for ax,(key,label,task,nc,scale,color) in zip(axes.flat,specs):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(.015,.965,label,fontsize=17,fontweight='bold',color=color,va='top')
        record=architectures.get(key);a=record[1] if record else None
        params=f"{a['parameters']:,} parameters" if a and isinstance(a.get('parameters'),int) else 'parameters: PENDING — 실제 모델 기록 대기'
        ax.text(.015,.87,params,fontsize=11,color=MUTED)
        scales=a.get('yaml',{}).get('scales',{}) if a else {}
        # The recorded YOLO11 n architecture includes the entire scale table.
        if not scales and architectures:scales=next(iter(architectures.values()))[1].get('yaml',{}).get('scales',{})
        factor=scales.get(scale)
        factors=f'깊이 배율 {factor[0]} / 채널 폭 배율 {factor[1]}' if factor else '스케일 상세: PENDING'
        ax.text(.015,.805,factors,fontsize=10,color=MUTED)
        config_name={'board_detect':'board_yolo11n','board_segment':'board_yolo11n_seg','parts_n':'parts_yolo11n','parts_s':'parts_yolo11s'}[key]
        config_path=root/'configs'/f'{config_name}.yaml'
        if config_path.is_file():
            cfg=yaml.safe_load(config_path.read_text(encoding='utf-8-sig'));train=cfg.get('train',{})
            ax.text(.015,.75,f"설정 예산 {train.get('epochs','PENDING')} epoch · 입력 {train.get('imgsz','PENDING')} px · batch {train.get('batch','PENDING')}",fontsize=10,color=MUTED)
            p['inputs'][str(config_path)]=sha(config_path)
        node(ax,.025,.43,.28,.29,'Backbone','Conv → C3k2\nSPPF → C2PSA\n다중 해상도 특징 추출',color=color,title_size=12,body_size=10.5)
        node(ax,.36,.43,.28,.29,'Feature fusion','Upsample + Concat\nC3k2 + downsample\nP3 / P4 / P5',color=color,title_size=12,body_size=10.5)
        head='Segment head' if task=='segment' else 'Detect head'
        body=f'{nc}개 클래스\nbox + mask 계수\nprototype mask 결합' if task=='segment' else f'{nc}개 클래스\n클래스 점수 + box\nP3·P4·P5에서 예측'
        node(ax,.695,.43,.28,.29,head,body,color=color,title_size=12,body_size=10.5)
        arrow(ax,(.305,.575),(.36,.575));arrow(ax,(.64,.575),(.695,.575))
        if a:
            modules=Counter(x['module'].split('.')[-1] for x in a.get('layers',[]))
            actual_head=a.get('layers',[{}])[-1].get('module','').split('.')[-1]
            ax.text(.025,.31,f"저장된 최상위 블록 {len(a.get('layers',[]))}개 · 실제 head: {actual_head}",fontsize=10.5,color=INK)
            ax.text(.025,.23,' / '.join(f'{k} {v}' for k,v in modules.items() if k in {'Conv','C3k2','SPPF','C2PSA','Concat','Upsample'}),fontsize=9.5,color=MUTED)
            p['inputs'][str(record[0])]=sha(record[0])
        else:ax.text(.025,.28,'설계 예정 구조 · 실제 model_id/블록 수는 학습 기록 생성 후 자동 갱신',fontsize=10,color=MUTED)
        ax.text(.025,.10,'출력: 보드 bbox + 보드 mask' if task=='segment' else ('출력: Raspberry Pi 보드 bbox' if nc==1 else '출력: 저항·콘덴서·IC·커넥터 bbox'),fontsize=11,fontweight='bold',color=color)
    fig.text(.045,.025,'P3/P4/P5는 stride 8/16/32 특징맵입니다. n과 s는 채널 폭이 다르며, 부품 mask branch는 이번 신규 학습에 포함되지 않습니다.',fontsize=11,color=MUTED)
    save(fig,out,'02_model_architectures',p)

def component_counts_figure(out,p,path):
    if not path.is_file():p['skipped'].append({'figure':'03_component_label_counts','reason':'component_data_manifest missing'});return
    m=read(path);p['inputs'][str(path)]=sha(path)
    if m.get('class_names')!=NAMES:raise ValueError('Unexpected component classes')
    splits=m['native_split_counts'];native=np.array([splits['train']['unique_instances'].get(k,0) for k in NAMES]);tiles=np.array([m['train_tiling']['class_exposure_counts'].get(k,0) for k in NAMES])
    fig,(left,right)=plt.subplots(1,2,figsize=(18,9.5),gridspec_kw={'width_ratios':[1.08,1]});fig.subplots_adjust(left=.07,right=.95,top=.81,bottom=.27,wspace=.22)
    title(fig,'라벨 수를 세는 기준: 원본 라벨과 타일 노출','WACV 원래 bbox를 사용 · 타일에 반복 등장한 같은 annotation을 새 원본 라벨로 세지 않습니다')
    x=np.arange(4);w=.34
    a=left.bar(x-w/2,native,w,label='학습 원본 고유 annotation',color=TEAL);b=left.bar(x+w/2,tiles,w,label='타일에서의 라벨 노출',color=BLUE)
    left.bar_label(a,fmt='%d',padding=5,fontsize=11);left.bar_label(b,fmt='%d',padding=5,fontsize=11)
    left.set_xticks(x,KOREAN);left.set_ylim(0,max(tiles)*1.22);left.set_ylabel('개수');left.set_title(f"학습 원본과 {m['train_tiling']['tile_count']}개 타일의 라벨 수",fontsize=15,pad=16);left.legend(loc='upper left',frameon=False,fontsize=10);left.grid(axis='y',alpha=.17);left.set_axisbelow(True)
    labels=['학습','검증','일반 test','Pi3B holdout'];keys=['train','val','test','pi_test']
    totals=[sum(splits[k]['unique_instances'].values()) for k in keys];images=[splits[k]['images'] for k in keys];groups=[splits[k]['board_groups'] for k in keys]
    bars=right.barh(np.arange(4),totals,color=[TEAL,BLUE,ORANGE,RED]);right.bar_label(bars,labels=[f'{v:,}개 / {im}장 / {gr}그룹' for v,im,gr in zip(totals,images,groups)],padding=6,fontsize=11)
    right.set_yticks(np.arange(4),labels);right.invert_yaxis();right.set_xlim(0,max(totals)*1.48);right.set_xlabel('원본 고유 annotation 개수');right.set_title('보드 그룹별로 분리한 원본 정답',fontsize=15,pad=16);right.grid(axis='x',alpha=.17);right.set_axisbelow(True)
    tile=m['train_tiling'];all_images=sum(images);all_inst=sum(totals)
    fig.text(.07,.19,f"전체: 원본 {all_images}장 · 고유 annotation {all_inst:,}개 | 학습: {splits['train']['images']}장 → {tile['tile_count']}타일 ({tile['negative_tiles']}개는 target4 배경)",fontsize=12,color=INK,fontweight='bold')
    fig.text(.07,.145,f"타일: {tile['tile_size']}px · overlap {int(tile['overlap']*100)}% · 중복 노출 {sum(tiles):,}회 · 경계에서 잘린 노출 {tile['clipped_exposures']:,}회",fontsize=11,color=MUTED)
    fig.text(.07,.10,'원본 고유 수는 이미지별 native annotation ID 기준입니다. 동일 물리 부품의 다각도 중복 여부는 별도 미검증이며 Pi3B 2장/189개는 학습에서 제외했습니다.',fontsize=10.7,color=MUTED)
    fig.text(.07,.055,'주의: 이 자료는 bbox 정답입니다. 부품 mask 정답이 아니며, 알려지지 않은 부품을 저항·콘덴서로 추정해 채우지 않았습니다.',fontsize=11,color=RED)
    save(fig,out,'03_component_label_counts',p)

def board_counts_figure(out,p,manifests,rejections):
    if set(manifests)!={'detect','segment'}:
        p['skipped'].append({'figure':'04_board_dataset_counts','reason':'Corrected eligible detect+segment manifests are not both available','rejections':rejections});return
    fig,axes=plt.subplots(1,2,figsize=(18,9.3));fig.subplots_adjust(left=.075,right=.95,top=.80,bottom=.27,wspace=.26)
    title(fig,'수정한 보드 데이터: 영상 수와 원본 라벨','G/H/M 실제 사진 확인을 반영한 manifest만 사용 · source별 loss factor 1.0 / 별도 oversampling 없음')
    for ax,task,name in zip(axes,['detect','segment'],['보드 bbox · 수정 micro + IoTKITs','보드 mask · IoTKITs native polygon']):
        path,m=manifests[task];p['inputs'][str(path)]=sha(path);counts=m['counts'];keys=['train','val','test'];x=np.arange(3)
        pos=[counts[s]['positive_images'] for s in keys];neg=[counts[s]['negative_images'] for s in keys]
        inst=[counts[s]['instances'] for s in keys]
        ax.bar(x,pos,color=TEAL,label='RPi 보드 포함 영상');ax.bar(x,neg,bottom=pos,color='#b9c8d8',label='다른 보드 등 배경 영상')
        for j,s in enumerate(keys):ax.text(j,pos[j]+neg[j]+max(pos+neg)*.035,f"{pos[j]+neg[j]:,}장\n정답 {inst[j]:,}개",ha='center',fontsize=11,color=INK)
        ax.set_xticks(x,['학습','검증','내부 test']);ax.set_ylim(0,max(a+b for a,b in zip(pos,neg))*1.24);ax.set_title(name,fontsize=14,pad=18);ax.set_ylabel('이미지 수');ax.legend(frameon=False,fontsize=10,loc='upper right');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.text(.075,.18,'micro README의 A/H/I 표기는 사진과 충돌했습니다. 코드별 3장씩 총 39장 표본을 확인해 R03을 G/H/M 코드로 다시 분류했습니다.',fontsize=11.6,color=RED,fontweight='bold')
    fig.text(.075,.135,'기존 A/H/I 기반 실험의 AP는 RPi baseline으로 비교하지 않습니다. 출처 그룹·exact SHA·pHash로 분리해도 물리 보드 독립성을 증명하지는 않습니다.',fontsize=11,color=MUTED)
    fig.text(.075,.09,'보드 mask 수는 원본 polygon 개체 수입니다. bbox를 사각형 mask로 바꾸지 않았습니다. Commons 24장/28 bbox 초안은 정량 평가에 포함하지 않습니다.',fontsize=11,color=MUTED)
    fig.text(.075,.045,'데이터셋 수치는 eligibility=true인 수정 manifest에서 읽습니다. D455 실촬영 성능은 이 내부 test 수치와 별도로 검증해야 합니다.',fontsize=11,color=MUTED)
    save(fig,out,'04_board_dataset_counts',p)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--out',type=Path);parser.add_argument('--board-detect-manifest',type=Path);parser.add_argument('--board-segment-manifest',type=Path)
    args=parser.parse_args();root=args.root.resolve();out=(args.out or root/'reports/figures').resolve();out.mkdir(parents=True,exist_ok=True);configure()
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Static Korean explanatory figures; no training, inference or evaluation performed','inputs':{},'figures':[],'skipped':[],
        'legacy_wrong_mapping_ap_used':False,'part_mask_training_claimed':False,'commons_draft_used_as_gt':False}
    architectures=architecture_files(root);boards,rejections=corrected_board_manifests(root,args.board_detect_manifest,args.board_segment_manifest)
    pipeline_figure(out,provenance);model_figure(out,provenance,architectures,root);component_counts_figure(out,provenance,root/'component_assets/component_data_manifest.json');board_counts_figure(out,provenance,boards,rejections)
    (out/'figure_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'figures':provenance['figures'],'skipped':provenance['skipped'],'architecture_records':list(architectures)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
