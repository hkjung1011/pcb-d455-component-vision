"""CPU-only R04 figures from existing recorded data; missing results are skipped.

PNG is the display/export artifact, SVG retains editable vector text.  The JSON
inventory binds each figure to exact input-file SHA-256 and plotted numbers.
This script never trains, predicts, selects a model or modifies source evidence.
Re-run after training/evaluation to replace provisional snapshot figures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

NAMES = ["resistor", "capacitor", "ic", "connector"]
KO = ["저항", "커패시터", "IC", "커넥터"]
ARMS = ["baseline", "improved"]
ARM_LABEL = {"baseline": "기준안", "improved": "개선안"}
COLORS = {"baseline": "#66758C", "improved": "#147D92"}
BIN_NAMES = ["lt8", "8to16", "16to32", "ge32"]
BIN_LABEL = ["<8 px", "8–<16 px", "16–<32 px", "≥32 px"]
HOLDOUT_NOTE = "R03에서 이미 검토한 개발 holdout 재사용 · 새로운 최종 시험 아님 · 실제 D455 미검증"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def percentage(value):
    if value is None or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid recorded rate: {value!r}")
    return float(value) * 100


def setup_style():
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        raise RuntimeError("Malgun Gothic is required to render this Korean artifact")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "svg.fonttype": "none",
                         "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
                         "savefig.facecolor": "#FFFFFF", "legend.frameon": False})


class Figures:
    def __init__(self, root):
        self.root = root.resolve()
        self.out = self.root / "reports/figures"
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache, self.used = {}, {}
        self.inventory = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                          "scope": "Recorded R04 data only; CPU plotting, no training/inference or invented results",
                          "development_holdout_reused": True, "d455_verified": False,
                          "figures": [], "skipped": [], "errors": []}

    def read(self, relative, optional=False):
        path = Path(relative)
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        key = str(path)
        if key not in self.cache:
            if optional and not path.is_file():
                return None
            raw = path.read_bytes()
            self.cache[key] = (json.loads(raw.decode("utf-8-sig")), digest(raw))
        value, sha = self.cache[key]
        self.used[key] = sha
        return value

    def skip(self, name, reason):
        self.inventory["skipped"].append({"id": name, "reason": reason,
                                          "existing_older_artifact": any((self.out / (name + ext)).exists() for ext in [".png", ".svg"])})

    def emit(self, fig, name, title, caption, values):
        fig.suptitle(title, x=.055, y=.975, ha="left", fontsize=18, fontweight="bold", color="#193044")
        fig.text(.055, .025, "\n".join(textwrap.wrap(caption, 112)), ha="left", va="bottom", fontsize=9, color="#485462")
        artifacts = {}
        for ext in ["png", "svg"]:
            path = self.out / f"{name}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            artifacts[ext] = {"file": path.name, "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
        plt.close(fig)
        self.inventory["figures"].append({"id": name, "title": title, "caption": caption,
                                          "artifacts": artifacts, "input_sha256": dict(self.used), "plotted_values": values})

    def aggregate(self):
        aggregate = self.read("reports/evaluation_suite/aggregate.json", optional=True)
        if aggregate is None:
            return None
        frozen = self.read("reports/evaluation_suite/selections_frozen.json")
        path = str((self.root / "reports/evaluation_suite/selections_frozen.json").resolve())
        if self.used[path] != aggregate["selection_sha256"]:
            raise ValueError("Frozen selection SHA does not match aggregate")
        if frozen.get("test_results_used_for_selection") is not False or aggregate.get("d455_verified") is not False:
            raise ValueError("Evaluation scope or selection assertion mismatch")
        return aggregate

    def checked_metric(self, record):
        metric = self.read(record["metrics_path"])
        path = str(Path(record["metrics_path"]).resolve())
        if self.used[path] != record["metrics_sha256"]:
            raise ValueError(f"Evaluation metric SHA mismatch: {path}")
        if metric["bbox"] != record["bbox"] or metric["operating"]["confidence"] != record["operating"]["confidence"]:
            raise ValueError("Aggregate metric differs from original metrics.json")
        return metric

    def complete_tests(self, figure_id):
        aggregate = self.aggregate()
        if aggregate is None or any(cohort not in aggregate["arms"][arm]["tests"] for arm in ARMS for cohort in ["general_test", "pi_test"]):
            self.skip(figure_id, "두 arm의 일반/Pi 개발 holdout 평가 4개가 모두 완료되지 않음")
            return None, None
        metrics = {arm: {cohort: self.checked_metric(aggregate["arms"][arm]["tests"][cohort])
                         for cohort in ["general_test", "pi_test"]} for arm in ARMS}
        return aggregate, metrics

    def pipeline(self):
        name = "01_algorithm_pipeline"
        protocol = self.read("protocol.json")
        summary = self.read("data/summary.json")
        fig, ax = plt.subplots(figsize=(15, 9))
        fig.subplots_adjust(left=.03, right=.98, top=.90, bottom=.13)
        ax.set(xlim=(0, 15), ylim=(0, 10)); ax.axis("off")
        def box(x, y, w, h, content, color="#EFF4F8", edge="#A4B6C7", fontsize=11):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.10,rounding_size=.12",facecolor=color,edgecolor=edge,lw=1.5))
            ax.text(x+w/2,y+h/2,content,ha="center",va="center",fontsize=fontsize,linespacing=1.55,color="#20364B")
        def arrow(start, end):
            ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="-|>",lw=1.8,color="#526B80",shrinkA=3,shrinkB=3))
        box(.2,7.9,3.5,1.45,"공개 원본 47장 · 4종 bbox\n그룹 단위 분할을 먼저 고정\n새 인간 라벨·부품 mask = 0")
        box(4.5,8.25,4.7,1.15,f"기준안: 원본 1024px 타일 {summary['arms'][0]['views']}개\n조각 라벨 유지 · 비대상/unknown 배경",color="#EDF0F5")
        box(4.5,6.45,4.7,1.15,f"개선안: 다중 크기·객체 중심 view {summary['arms'][1]['views']}개\n작은 조각·unknown → 실제 음성 loss 제외",color="#E5F4F3")
        arrow((3.8,8.65),(4.35,8.8)); arrow((3.3,7.8),(4.35,7.0))
        box(10.1,6.6,4.35,2.7,f"공통 학습 조건\n공식 YOLO11s 초기 가중치\n최대 {protocol['max_epochs_per_arm']} epoch / optimizer {protocol['max_direct_optimizer_calls_per_arm']:,}회\nbatch {protocol['batch']} · FP32 · 같은 sampler 정책\n검증 후보: 2 epoch 간격",color="#ECF4FB")
        arrow((9.3,8.8),(10,8.3)); arrow((9.3,7.0),(10,7.6))
        box(9.7,3.85,4.7,1.8,"모든 arm × 10 checkpoint × 2 추론 방식\n검증 후보 40개: native / dual\nval AP@300 → AP@100 → 지연으로 선택\nconfidence도 val에서 결정",color="#E8F2FA")
        arrow((12.3,6.4),(12.3,5.8))
        box(5.45,4.0,3.2,1.45,"모든 선택·confidence 동결\n선택 파일 SHA-256 기록\n그 다음 holdout 평가",color="#FFF4DA",edge="#D1B767")
        arrow((9.5,4.75),(8.8,4.75))
        box(.2,3.65,4.15,2.1,"일반 11장 / Pi3B 2장\n원본 좌표 bbox AP@100·300\n클래스·크기별 recall / TP·FP·FN\nR03에서 본 개발 holdout 재사용",color="#F4ECF6",edge="#B59DC0")
        arrow((5.3,4.7),(4.5,4.7))
        box(.2,.35,14.2,1.8,"후속 실제 D455 단계 — 현재 공개자료 시험과 별도\n실촬영·intrinsics·픽셀 크기 → 판독 가능성 확인 → 검수된 라벨/instance mask → 새 실물·세션·설계 시험\n현재 결과로 4종 부품 세그멘테이션 또는 실제 D455 성능을 주장하지 않음",color="#F6F7F8",edge="#B2BCC5")
        self.emit(fig,name,"R04 알고리즘과 검증 순서", "구조도는 protocol과 데이터 준비 기록을 표현한다. 후보 40개는 사전 계획 수이며 실제 완료 여부는 05 그림과 inventory에서 확인한다. " + HOLDOUT_NOTE, {"training_protocol":protocol,"view_counts":{x['arm']:x['views'] for x in summary['arms']}})

    def labels(self):
        name = "02_original_labels_and_view_exposures"
        summary = self.read("data/summary.json")
        split = summary["split"]
        fig, axes = plt.subplots(1,2,figsize=(15,6.8))
        fig.subplots_adjust(left=.07,right=.97,bottom=.24,top=.82,wspace=.25)
        x = np.arange(4)
        cohort_labels = ["학습", "검증", "일반 holdout", "Pi holdout"]
        cohort_colors = ["#165D75", "#5AA9A3", "#AD8AC7", "#D9A84F"]
        original = {}
        for i,(cohort,label,color) in enumerate(zip(["train","val","test","pi_test"],cohort_labels,cohort_colors)):
            values = [split[cohort]["class_counts"].get(k,0) for k in NAMES]
            original[cohort]=values
            bars=axes[0].bar(x+(i-1.5)*.19,values,.19,label=f"{label} {split[cohort]['images']}장",color=color)
            axes[0].bar_label(bars,fontsize=8,padding=3)
        axes[0].set(xticks=x,xticklabels=KO,ylabel="고유 원본 bbox 수 (개)",title="원본 정답: split별 수량")
        axes[0].legend(ncol=2,fontsize=9,loc="upper center",bbox_to_anchor=(.5,-.11)); axes[0].set_ylim(0,max(max(v) for v in original.values())*1.2)
        values_by_kind={"학습 원본":original['train']}
        for arm_summary in summary['arms']:
            values_by_kind[ARM_LABEL[arm_summary['arm']]+" 저장 view"]=[arm_summary['supervised_label_exposures'].get(k,0) for k in NAMES]
        for i,(label,values) in enumerate(values_by_kind.items()):
            bars=axes[1].bar(x+(i-1)*.24,values,.24,label=label,color=["#C4A775",COLORS['baseline'],COLORS['improved']][i])
            axes[1].bar_label(bars,fontsize=8,padding=3)
        axes[1].set(xticks=x,xticklabels=KO,ylabel="객체 / 저장 라벨 노출 수 (회)",title="같은 원본을 여러 view에서 반복 사용")
        axes[1].legend(fontsize=9,loc="upper center",bbox_to_anchor=(.5,-.11)); axes[1].set_ylim(0,max(max(v) for v in values_by_kind.values())*1.16)
        for ax in axes: ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        self.emit(fig,name,"고유 원본 라벨과 view 라벨 노출을 구분", "우측은 생성된 view를 각각 한 번 세었을 때의 라벨 총합이다. 실제 weighted draw 노출·새 라벨·gradient 비중과 다르다. 새 인간 라벨 및 instance mask는 0개. " + HOLDOUT_NOTE, {"original_by_split":original,"materialized_view_labels":values_by_kind})

    def coverage(self):
        name="03_complete_views_and_ignore_regions"
        summary=self.read("data/summary.json")
        totals=[summary['split']['train']['class_counts'][k] for k in NAMES]
        covered={}; ignores={}
        for arm in ARMS:
            views=self.read(f"data/{arm}/views.json")['records']
            seen=[set() for _ in NAMES]
            for view in views:
                for target in view['targets']:
                    if target['visible_fraction']>=1-1e-8:seen[target['class_id']].add(target['instance_id'])
            covered[arm]=[len(v) for v in seen]
            ignores[arm]=dict(Counter(r['reason'] for v in views for r in v['ignore_regions']))
        fig,axes=plt.subplots(1,2,figsize=(15,6.5));fig.subplots_adjust(left=.07,right=.97,bottom=.21,top=.82,wspace=.24)
        x=np.arange(4)
        for i,arm in enumerate(ARMS):
            values=100*np.array(covered[arm])/np.array(totals)
            bars=axes[0].bar(x+(i-.5)*.32,values,.32,label=ARM_LABEL[arm],color=COLORS[arm])
            axes[0].bar_label(bars,labels=[f"{v:.1f}%" for v in values],fontsize=9,padding=3)
            for bar,n,d in zip(bars,covered[arm],totals):
                axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()/2,f"{n:,}/{d:,}개",rotation=90,ha='center',va='center',fontsize=9,color='white')
        axes[0].set(xticks=x,xticklabels=KO,ylabel="원본 객체의 완전 bbox view 확보율 (%)",ylim=(0,125),title="한 view 이상에 bbox 전체가 포함된 원본 객체")
        axes[0].legend(ncol=2,loc="lower right")
        reason_keys=["target_fragment_lt50","source_unknown"]
        values=[ignores['improved'].get(k,0) for k in reason_keys]
        bars=axes[1].bar([0,1],values,.5,color=["#DBA953","#A68BBC"])
        axes[1].bar_label(bars,labels=[f"{v:,}회" for v in values],padding=5)
        axes[1].set(xticks=[0,1],xticklabels=["50% 미만 bbox 조각","원본 unknown"],ylabel="저장 view의 ignore 영역 수 (회)",title="개선안 true-ignore metadata",ylim=(0,max(values)*1.22 if max(values)>0 else 1))
        for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        self.emit(fig,name,"완전한 부품 모습 확보와 학습 ignore 범위", "확보율은 저장 view 기하 기준이며 sampler가 실제로 모든 객체를 뽑았다는 뜻은 아니다. ignore는 학습의 음성 분류 loss만 제외한다. 알려진 비대상은 배경 유지, 평가 GT에는 ignore를 추가하지 않았다.", {"source_train_counts":totals,"complete_source_bbox_views":covered,"materialized_ignore_regions":ignores})

    def training(self):
        name="04_actual_training_loss_and_updates"
        records={arm:self.read(f"runs/{arm}/epochs.json",optional=True) for arm in ARMS}
        records={arm:rows for arm,rows in records.items() if rows}
        if not records:return self.skip(name,"실제 epoch 기록이 아직 없음")
        completed={arm:self.read(f"runs/{arm}/training_summary.json",optional=True) for arm in records}
        fig,axes=plt.subplots(2,3,figsize=(15,9));fig.subplots_adjust(left=.07,right=.97,bottom=.16,top=.86,wspace=.26,hspace=.39)
        for arm,rows in records.items():
            epochs=[r['epoch'] for r in rows]
            if len(set(epochs))!=len(epochs) or any(e<1 or e>20 for e in epochs):raise ValueError("Invalid actual epoch rows")
            suffix="종료" if completed[arm] else "진행 중"
            partial=" 일부" if rows[-1]['partial_epoch'] else ""
            label=f"{ARM_LABEL[arm]} · {epochs[-1]}번째{partial} ({suffix})"
            for i,key in enumerate(['box_loss','cls_loss','dfl_loss']):
                axes[0,i].plot(epochs,[r['train_loss'][key] for r in rows],marker='o',ms=3,color=COLORS[arm],label=label)
                axes[0,i].set(title=f"학습 {key}",xlabel="실제 epoch",ylabel="loss")
            axes[1,0].plot(epochs,[r['optimizer_calls'] for r in rows],marker='o',ms=3,color=COLORS[arm],label=label)
            axes[1,1].plot(epochs,[percentage(r['full_image_loader_validation']['metrics/mAP50-95(B)']) for r in rows],marker='o',ms=3,color=COLORS[arm])
            axes[1,2].plot(epochs,[r['lr']['lr/pg0'] for r in rows],marker='o',ms=3,color=COLORS[arm])
            for r in rows:
                if r['partial_epoch']:axes[1,0].scatter(r['epoch'],r['optimizer_calls'],marker='*',s=125,color=COLORS[arm],zorder=4)
        protocol=self.read('protocol.json')
        axes[1,0].axhline(protocol['max_direct_optimizer_calls_per_arm'],color='#AC4E46',ls='--',lw=1)
        axes[1,0].set(title="직접 센 optimizer.step 호출",xlabel="실제 epoch",ylabel="누적 호출 수 (회)")
        axes[1,1].set(title="전체 이미지 축소 val: 진단용",xlabel="실제 epoch",ylabel="bbox mAP50–95 (%)")
        axes[1,2].set(title="학습률 (parameter group 0)",xlabel="실제 epoch",ylabel="learning rate")
        axes[1,2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlim(.5,20.5);ax.set_xticks([1,5,10,15,20])
        axes[0,0].legend(fontsize=8)
        state={arm:{"last_recorded_epoch":rows[-1]['epoch'],"training_complete":completed[arm] is not None,"rows":rows} for arm,rows in records.items()}
        self.emit(fig,name,"실제 학습 기록: loss·가중치 갱신·학습률", "저장된 epoch만 표시한다. 별표는 budget으로 중단된 부분 epoch다. 전체 이미지 축소 val은 checkpoint 최종 선택 기준이 아니며, 최종 선택은 별도 native/dual val이다. loss 수치 차이만으로 정확도 우열을 판단하지 않는다.",state)

    def validation(self):
        name="05_all_validation_candidates_ap300"
        aggregate=self.aggregate()
        if aggregate is None:return self.skip(name,"모든 validation 후보가 완료되어 aggregate가 생성되기 전")
        protocol=self.read('protocol.json');expected=len(protocol['save_epoch_candidates'])*len(protocol['inference_candidates'])
        if any(len(aggregate['arms'][arm]['validation_candidates'])!=expected for arm in ARMS):return self.skip(name,"사전 계획된 40개 validation 후보가 모두 없음")
        fig,ax=plt.subplots(figsize=(13,7));fig.subplots_adjust(left=.085,right=.97,bottom=.30,top=.85)
        values={}
        for arm in ARMS:
            for pipeline,linestyle in [('native','-'),('dual','--')]:
                rows=sorted([r for r in aggregate['arms'][arm]['validation_candidates'] if r['candidate']['pipeline']==pipeline],key=lambda r:r['candidate']['epoch'])
                numbers=[percentage(self.checked_metric(r)['bbox']['ap50_95_max300']) for r in rows]
                epochs=[r['candidate']['epoch'] for r in rows]
                ax.plot(epochs,numbers,color=COLORS[arm],ls=linestyle,marker='o',label=f"{ARM_LABEL[arm]} · {pipeline}")
                values[f'{arm}__{pipeline}']={'epochs':epochs,'ap300_percent':numbers}
            selected=aggregate['arms'][arm]['selection'];c=selected['candidate']
            ap=percentage(selected['validation_bbox']['ap50_95_max300'])
            ax.scatter(c['epoch'],ap,marker='*',s=240,color=COLORS[arm],edgecolor='white',zorder=5)
            ax.annotate(f"{ARM_LABEL[arm]} 선택: {c['epoch']}e / {c['pipeline']}\n{ap:.2f}%",(c['epoch'],ap),xytext=(8,18) if arm=='improved' else (-142,-74),textcoords='offset points',fontsize=9,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2},
                        arrowprops=None if arm=='improved' else {'arrowstyle':'->','color':COLORS[arm],'lw':.8})
        ax.set(xlabel='후보 checkpoint의 실제 epoch',ylabel='원본 좌표 bbox AP50–95 @ maxDets=300 (%)',xticks=protocol['save_epoch_candidates'])
        ax.grid(alpha=.2);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.12))
        low,high=ax.get_ylim();ax.set_ylim(max(0,low-2),high+3)
        self.emit(fig,name,"40개 검증 후보에서 checkpoint·추론 방식을 선택", "native = 원본 1024 타일, dual = 원본 1024 + 2048 타일을 1024 입력으로 처리. AP는 이미지 전체 최대1000개 제한 후 클래스별 최대300개 조건이다. 별표는 val 선택이며 holdout은 선택에 사용하지 않았다.",values)

    def tests(self):
        name="06_development_holdout_metrics"
        aggregate,metrics=self.complete_tests(name)
        if metrics is None:return
        fig,axes=plt.subplots(1,2,figsize=(15,7));fig.subplots_adjust(left=.07,right=.97,bottom=.25,top=.83,wspace=.22)
        data={};x=np.arange(4)
        for ax,cohort,title in zip(axes,['general_test','pi_test'],['일반 개발 holdout','Pi3B 개발 holdout']):
            data[cohort]={}
            for i,arm in enumerate(ARMS):
                m=metrics[arm][cohort];o=m['operating']
                values=[percentage(m['bbox'][key]) for key in ['ap50_95_max100','ap50_95_max300']]+[percentage(o['precision']),percentage(o['recall'])]
                bars=ax.bar(x+(i-.5)*.33,values,.33,color=COLORS[arm],label=f"{ARM_LABEL[arm]} · conf {o['confidence']:.2f}")
                ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=4,fontsize=9)
                data[cohort][arm]={'values_percent':values,'confidence':o['confidence'],'images':m['images'],'groups':m['groups']}
            m=metrics['baseline'][cohort]
            ax.set(title=f"{title}: {m['images']}장 / {m['groups']}그룹",xticks=x,xticklabels=['AP@100','AP@300','Precision','Recall'],ylabel='%',ylim=(0,112));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.11))
        self.emit(fig,name,"선택 동결 후 개발 holdout 성능 비교", "AP는 이미지 전체 최대1000개 제한 후 클래스별 최대100/300개, bbox IoU 0.50:0.95 평균이다. P/R은 각 arm의 val에서 고정한 confidence와 IoU≥0.5 기준이며 같은 threshold 비교가 아니다. " + HOLDOUT_NOTE,data)

    def pi_recall(self):
        name="07_pi_class_recall_and_counts"
        aggregate,metrics=self.complete_tests(name)
        if metrics is None:return
        fig,ax=plt.subplots(figsize=(13,6.5));fig.subplots_adjust(left=.08,right=.97,bottom=.22,top=.83)
        data={};x=np.arange(4)
        for i,arm in enumerate(ARMS):
            m=metrics[arm]['pi_test'];counts=np.asarray(m['operating']['per_class_counts_tp_fp_fn']);total=counts[:,0]+counts[:,2]
            if (total==0).any():raise ValueError('Unexpected missing Pi target class')
            values=100*counts[:,0]/total
            bars=ax.bar(x+(i-.5)*.32,values,.32,color=COLORS[arm],label=f"{ARM_LABEL[arm]} · conf {m['operating']['confidence']:.2f}")
            ax.bar_label(bars,labels=[f"{v:.1f}%\n{tp}/{n}개" for v,tp,n in zip(values,counts[:,0],total)],padding=5,fontsize=10)
            data[arm]={'class_order':NAMES,'tp_fp_fn':counts.tolist(),'recall_percent':values.tolist(),'confidence':m['operating']['confidence']}
        ax.set(xticks=x,xticklabels=KO,ylabel='Recall (%) · 같은 클래스와 IoU≥0.5',ylim=(0,119));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.10))
        self.emit(fig,name,"Pi3B 부품별로 실제 몇 개를 찾았는가", "막대 위 숫자는 TP / 원본 GT 수다. 원본 정답의 미실장 footprint·unknown 문제를 임의로 재라벨링하거나 제외하지 않았다. Pi 앞뒤 2장·1그룹이므로 그룹 신뢰구간이나 일반적인 Raspberry Pi 성능으로 해석하지 않는다. " + HOLDOUT_NOTE,data)

    def class_size(self):
        name="08_class_by_native_size_recall"
        aggregate,metrics=self.complete_tests(name)
        if metrics is None:return
        fig,axes=plt.subplots(2,2,figsize=(14,11));fig.subplots_adjust(left=.09,right=.88,bottom=.15,top=.88,hspace=.37,wspace=.27)
        data={};cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#ECEFF1')
        for row,cohort in enumerate(['general_test','pi_test']):
            for col,arm in enumerate(ARMS):
                ax=axes[row,col];m=metrics[arm][cohort];table=m['operating']['class_by_native_shortside_recall']
                values=np.full((4,4),np.nan);annotations=[]
                for i,key in enumerate(NAMES):
                    annotation=[]
                    for j,bin_name in enumerate(BIN_NAMES):
                        cell=table[key][bin_name];tp,total=cell['detected'],cell['total']
                        if total:
                            values[i,j]=percentage(cell['recall']);label=f"{values[i,j]:.0f}%\n{tp}/{total}"
                        else:label="—\nGT 0"
                        annotation.append(label)
                    annotations.append(annotation)
                im=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=100,cmap=cmap,aspect='auto')
                for i in range(4):
                    for j in range(4):ax.text(j,i,annotations[i][j],ha='center',va='center',fontsize=10,color='white' if np.isfinite(values[i,j]) and values[i,j]>65 else '#243A49')
                ax.set(xticks=np.arange(4),xticklabels=BIN_LABEL,yticks=np.arange(4),yticklabels=KO,xlabel='원본 정답 bbox의 짧은 변',title=f"{'일반' if row==0 else 'Pi3B'} · {ARM_LABEL[arm]} · conf {m['operating']['confidence']:.2f}")
                data[f'{cohort}__{arm}']=table
        cax=fig.add_axes([.91,.23,.02,.57]);fig.colorbar(im,cax=cax,label='Recall (%)')
        self.emit(fig,name,"클래스와 원본 픽셀 크기를 함께 본 검출률", "각 셀은 recall 및 TP/GT 수다. 회색은 정답 0개로 평가할 수 없는 구간이며 0%를 뜻하지 않는다. 크기는 공개 원본 좌표 기준으로 D455 픽셀 크기나 물리 mm가 아니다. " + HOLDOUT_NOTE,data)

    def draw_balance(self):
        name="09_actual_group_sampling_balance"
        summary=self.read('data/summary.json');groups=summary['split']['training_groups']
        sampled={arm:self.read(f'runs/{arm}/sampled_exposures.json',optional=True) for arm in ARMS}
        sampled={k:v for k,v in sampled.items() if v and v['draws']>0}
        if not sampled:return self.skip(name,'실제 sampled_exposures 기록이 없음')
        fig,ax=plt.subplots(figsize=(15,7.6));fig.subplots_adjust(left=.075,right=.97,bottom=.35,top=.83)
        x=np.arange(len(groups));width=.7/len(sampled);data={}
        for i,(arm,rec) in enumerate(sampled.items()):
            if sum(rec['groups'].values())!=rec['draws']:raise ValueError('Actual group draw counts do not sum to recorded draws')
            counts=[rec['groups'].get(g,0) for g in groups];values=100*np.array(counts)/rec['draws']
            bars=ax.bar(x+(i-(len(sampled)-1)/2)*width,values,width,color=COLORS[arm],label=f"{ARM_LABEL[arm]} · 실제 {rec['draws']:,} draw")
            ax.bar_label(bars,labels=[str(v) for v in counts],fontsize=7,padding=3,rotation=90)
            data[arm]={'draws':rec['draws'],'group_order':groups,'counts':counts,'percent':values.tolist(),'actual_class_exposures':rec['class_exposures'],'actual_ignore_exposures':rec['ignore_exposures'],'unique_source_instances_seen':rec['unique_positive_source_instances_seen'],'unique_source_instances_seen_complete':rec['unique_positive_source_instances_seen_complete']}
        expected=100/len(groups);ax.axhline(expected,color='#B78534',ls='--',label=f'그룹 균등 기대값 {expected:.2f}%')
        ax.set(xticks=x,xticklabels=groups,ylabel='실제 학습 draw 중 해당 원본 보드 그룹의 비중 (%)')
        plt.setp(ax.get_xticklabels(),rotation=55,ha='right',fontsize=9)
        low,high=ax.get_ylim();ax.set_ylim(0,high*1.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(ncol=3,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.11))
        self.emit(fig,name,"실제로 학습한 보드 그룹별 draw 비중", "막대 위 숫자는 실제 draw 횟수다. 같은 view 재선택을 포함하며 고유 이미지·고유 라벨·gradient 비중과 다르다. 학습 중에는 마지막 저장된 누적 snapshot을 표시한다. 그룹 내 IC/커넥터 포함 view 우선도는 두 arm에 동일 적용한다.",data)

    def run(self):
        methods=[self.pipeline,self.labels,self.coverage,self.training,self.validation,self.tests,self.pi_recall,self.class_size,self.draw_balance]
        for method in methods:
            self.used={}
            try:method()
            except Exception as exc:
                self.inventory['errors'].append({'generator':method.__name__,'error':repr(exc)})
        self.inventory['generator_sha256']=digest(Path(__file__).read_bytes())
        (self.out/'figure_inventory.json').write_text(json.dumps(self.inventory,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        lines=['# R04 그래프','',f"생성: {self.inventory['generated_utc']}",'','기록된 실제 값만 사용했다. 학습 중 그래프는 부분 snapshot이며, 평가가 없는 그래프는 생성하지 않았다. 공개자료 개발 holdout은 R03에서 이미 검토한 자료이고 실제 D455 성능 또는 새로운 최종 시험이 아니다.','', '## 생성된 그림','']
        for row in self.inventory['figures']:
            lines += [f"### {row['title']}",'',f"![{row['title']}]({row['artifacts']['png']['file']})",'',row['caption'],'',f"[SVG 벡터 파일]({row['artifacts']['svg']['file']})",'']
        if self.inventory['skipped']:
            lines += ['## 대기 중인 그림','']
            for row in self.inventory['skipped']:lines.append(f"- {row['id']}: {row['reason']}")
        if self.inventory['errors']:
            lines += ['','## 확인이 필요한 생성 오류','']
            for row in self.inventory['errors']:lines.append(f"- {row['generator']}: {row['error']}")
        lines += ['','각 입력과 결과 파일의 SHA-256, 실제 plotted_values는 [figure_inventory.json](figure_inventory.json)에 기록했다.','']
        (self.out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
        print(json.dumps({'generated':[r['id'] for r in self.inventory['figures']],'skipped':self.inventory['skipped'],'errors':self.inventory['errors']},ensure_ascii=False,indent=2))
        if self.inventory['errors']:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();setup_style();Figures(args.root).run()


if __name__=='__main__':main()
