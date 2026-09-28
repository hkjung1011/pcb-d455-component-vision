"""CPU-only paired R04 GT error analysis from saved predictions, never inference.

Requires completed aggregate.json, the unchanged native_manifest.json, and the
four saved samples.json files.  Uses each arm's already frozen global confidence.
No class/threshold/model selection or annotation correction is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unittest

NAMES = ["resistor", "capacitor", "ic", "connector"]
ARMS = ["baseline", "improved"]
STATES = ["both_found", "improved_only", "baseline_only", "both_missed"]
STATE_KO = {"both_found": "둘 다 찾음", "improved_only": "개선안만 찾음",
            "baseline_only": "기준안만 찾음", "both_missed": "둘 다 놓침"}


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def box_iou(a, b):
    """Independent scalar implementation of the frozen half-open xyxy IoU."""
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    overlap = width * height
    union = max(0., a[2]-a[0]) * max(0., a[3]-a[1]) + max(0., b[2]-b[0]) * max(0., b[3]-b[1]) - overlap
    return overlap / union if union > 0 else 0.


def match_sample(sample, confidence):
    """Score-ordered class-same IoU>=.5 one-to-one greedy matching.

    Stable prediction-score ordering and larger-GT-index IoU tie resolution
    deliberately match evaluate_r04.operate.  Ignore is never applied to GT.
    """
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid frozen confidence")
    truths = sample["truth"]
    found, false_positives = {}, []
    counts = [[0, 0, 0] for _ in NAMES]
    retained = []
    for index, pred in enumerate(sample["predictions"]):
        if pred["class_id"] not in range(4) or not math.isfinite(pred["score"]) or not 0 <= pred["score"] <= 1:
            raise ValueError("Invalid saved prediction")
        if pred["score"] >= confidence:
            retained.append((index, pred))
    for index, pred in sorted(retained, key=lambda x: -x[1]["score"]):
        candidates = [(box_iou(pred["bbox_xyxy"], gt["bbox_xyxy"]), gt_index)
                      for gt_index, gt in enumerate(truths)
                      if gt_index not in found and gt["class_id"] == pred["class_id"]]
        quality, gt_index = max(candidates, default=(0., -1))
        if quality >= .5:
            found[gt_index] = {"prediction_index": index, "score": pred["score"],
                               "iou": quality, "bbox_xyxy": pred["bbox_xyxy"], "class_id": pred["class_id"]}
            counts[pred["class_id"]][0] += 1
        else:
            false_positives.append(index)
            counts[pred["class_id"]][1] += 1
    for index, gt in enumerate(truths):
        if index not in found:
            counts[gt["class_id"]][2] += 1
    return {"found": found, "false_positive_prediction_indices": false_positives,
            "per_class_counts_tp_fp_fn": counts}


def candidate_for(prediction, prediction_index, gt):
    return {"prediction_index": prediction_index, "class_id": prediction["class_id"],
            "class_name": NAMES[prediction["class_id"]], "score": prediction["score"],
            "iou": box_iou(prediction["bbox_xyxy"], gt["bbox_xyxy"]),
            "bbox_xyxy": prediction["bbox_xyxy"]}


def candidate_diagnosis(gt, predictions, confidence, match, gt_index):
    same, wrong = [], []
    for index, pred in enumerate(predictions):
        candidate = candidate_for(pred, index, gt)
        (same if pred["class_id"] == gt["class_id"] else wrong).append(candidate)
    same_overlap = [row for row in same if row["iou"] > 0]
    wrong_overlap = [row for row in wrong if row["iou"] > 0]
    same_feasible = [row for row in same if row["iou"] >= .5]
    wrong_feasible = [row for row in wrong if row["iou"] >= .5]
    best_iou = max(same, key=lambda row: (row["iou"], row["score"], -row["prediction_index"]), default=None)
    best_score = max(same_feasible, key=lambda row: (row["score"], row["iou"], -row["prediction_index"]), default=None)
    best_wrong = max(wrong_overlap, key=lambda row: (row["iou"], row["score"], -row["prediction_index"]), default=None)
    best_wrong_score = max(wrong_feasible, key=lambda row: (row["score"], row["iou"], -row["prediction_index"]), default=None)
    assignment = match["found"].get(gt_index)
    if assignment is not None:
        explanation = "found_at_frozen_confidence"
    elif best_score is not None and best_score["score"] < confidence:
        explanation = "same_class_iou_ge05_candidate_below_frozen_confidence"
    elif best_score is not None:
        explanation = "same_class_qualified_candidate_lost_in_one_to_one_assignment"
    elif best_wrong_score is not None and best_wrong_score["score"] >= confidence:
        explanation = "wrong_class_iou_ge05_candidate_at_frozen_confidence"
    else:
        explanation = "no_retained_same_class_iou_ge05_candidate"
    # A high-confidence same-class candidate can already belong to a neighbour.
    assigned_to = {record["prediction_index"]: idx for idx, record in match["found"].items()}
    if best_score is not None:
        best_score = dict(best_score, assigned_truth_index=assigned_to.get(best_score["prediction_index"]))
    return {"matched": assignment is not None, "matched_prediction": assignment,
            "frozen_confidence": confidence, "diagnostic_category": explanation,
            "best_iou_same_class": best_iou,
            "highest_score_same_class_iou_ge05": best_score,
            "highest_score_same_class_positive_overlap": max(same_overlap, key=lambda row: (row["score"], row["iou"]), default=None),
            "largest_iou_wrong_class_positive_overlap": best_wrong,
            "highest_score_wrong_class_iou_ge05": best_wrong_score,
            "retained_same_class_candidates": len(same), "same_class_iou_ge05_candidates": len(same_feasible),
            "candidate_scope": "Saved post-crop-NMS and post-global-NMS candidates, raw confidence floor .001 and global cap 1000; not all neural-network outputs"}


def source_refdes(source_name):
    match = re.match(r'^\s*(?:"[^"]+"|\S+)\s*(.*)$', source_name or "")
    return match.group(1).strip() or None if match else None


class Analysis:
    def __init__(self, root):
        self.root = root.resolve()
        self.hashes = {}

    def read(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        raw = path.read_bytes()
        self.hashes[str(path.resolve())] = sha_bytes(raw)
        return json.loads(raw.decode("utf-8-sig"))

    def run(self):
        aggregate = self.read("reports/evaluation_suite/aggregate.json")
        if aggregate.get("status") != "COMPLETED_DEVELOPMENT_HOLDOUT_NOT_D455_VALIDATED":
            raise ValueError("All frozen development-holdout evaluations must finish before error analysis")
        manifest = self.read("data/native_manifest.json")
        manifest_sha = self.hashes[str((self.root / "data/native_manifest.json").resolve())]
        if manifest["names"] != NAMES:
            raise ValueError("Unexpected source ontology")
        native = {row["id"]: row for row in manifest["records"]}
        cohorts, pi_details, parity = {}, [], []
        # Import only CPU functions from the original evaluator, never main().
        from evaluate_r04 import operate
        for cohort, expected_gt in [("general_test", 1348), ("pi_test", 189)]:
            selected_native = {key: row for key, row in native.items() if row["split"] == "test" and row.get("cohort") == cohort}
            if sum(len(row["objects"]) for row in selected_native.values()) != expected_gt:
                raise ValueError("Frozen native cohort denominator changed")
            samples_by_arm, matches_by_arm, confidence_by_arm = {}, {}, {}
            for arm in ARMS:
                payload = aggregate["arms"][arm]
                record = payload["tests"][cohort]
                confidence = payload["selection"]["operating_confidence"]
                if record["operating"]["confidence"] != confidence or record["manifest_sha256"] != manifest_sha:
                    raise ValueError("Holdout confidence/native manifest differs from frozen aggregate")
                if record["checkpoint_sha256"] != payload["selection"]["checkpoint_sha256"]:
                    raise ValueError("Holdout checkpoint differs from frozen selection")
                path = Path(record["metrics_path"]).with_name("samples.json")
                samples = self.read(path)
                sample_map = {sample["id"]: sample for sample in samples}
                if len(sample_map) != len(samples) or set(sample_map) != set(selected_native):
                    raise ValueError("Saved sample IDs do not match native cohort")
                totals = [[0, 0, 0] for _ in NAMES]
                matches = {}
                for image_id, sample in sample_map.items():
                    native_row = selected_native[image_id]
                    if sample["group_id"] != native_row["group_id"]:
                        raise ValueError("Saved source group differs from native cohort")
                    expected = [{"class_id": obj["class_id"], "bbox_xyxy": obj["bbox_xyxy"]} for obj in native_row["objects"]]
                    if sample["truth"] != expected:
                        raise ValueError("Saved GT order/classes/coordinates differ from original native GT")
                    match = match_sample(sample, confidence)
                    matches[image_id] = match
                    for cls in range(4):
                        totals[cls] = [a+b for a,b in zip(totals[cls],match["per_class_counts_tp_fp_fn"][cls])]
                original_result = operate(samples, confidence)
                if totals != original_result["per_class_counts_tp_fp_fn"] or totals != record["operating"]["per_class_counts_tp_fp_fn"]:
                    raise ValueError("Independent one-to-one counts disagree with original evaluator or recorded metrics")
                summed = [sum(row[i] for row in totals) for i in range(3)]
                if summed != [record["operating"][key] for key in ["tp", "fp", "fn"]]:
                    raise ValueError("Independent global TP/FP/FN differs from aggregate")
                parity.append({"arm": arm, "cohort": cohort, "confidence": confidence,
                               "independent_tp_fp_fn": summed, "per_class_tp_fp_fn": totals,
                               "matches_original_evaluator": True, "matches_recorded_metrics": True,
                               "native_gt_order_and_coordinates_unchanged": True})
                samples_by_arm[arm], matches_by_arm[arm], confidence_by_arm[arm] = sample_map, matches, confidence
            comparison, state_counts = [], Counter()
            by_class = {name: Counter() for name in NAMES}
            for image_id, row in selected_native.items():
                for index, gt in enumerate(row["objects"]):
                    baseline = index in matches_by_arm["baseline"][image_id]["found"]
                    improved = index in matches_by_arm["improved"][image_id]["found"]
                    state = "both_found" if baseline and improved else "baseline_only" if baseline else "improved_only" if improved else "both_missed"
                    item = {"instance_id": gt["instance_id"], "image_id": image_id, "group_id": row["group_id"],
                            "source_name": gt.get("source_name"), "source_refdes": source_refdes(gt.get("source_name")),
                            "class_id": gt["class_id"], "class_name": NAMES[gt["class_id"]],
                            "bbox_xyxy": gt["bbox_xyxy"], "state": state,
                            "baseline_match": matches_by_arm["baseline"][image_id]["found"].get(index),
                            "improved_match": matches_by_arm["improved"][image_id]["found"].get(index)}
                    comparison.append(item);state_counts[state] += 1;by_class[item["class_name"]][state] += 1
                    if cohort == "pi_test" and gt["class_id"] in [2, 3]:
                        details = dict(item)
                        details["arms"] = {arm:candidate_diagnosis(gt, samples_by_arm[arm][image_id]["predictions"],
                                            confidence_by_arm[arm], matches_by_arm[arm][image_id], index) for arm in ARMS}
                        if image_id == "RPI3B_Bottom" and details["source_refdes"] == "J9":
                            details["prior_review_context"] = "Previous human review referred to this connector J9 as the microSD socket. This analysis reads no photograph and does not independently verify the subtype."
                        pi_details.append(details)
            counts = {state:state_counts[state] for state in STATES}
            if sum(counts.values()) != expected_gt:
                raise ValueError("Paired outcome states do not partition original GT")
            cohorts[cohort] = {"images": len(selected_native), "gt_instances": expected_gt,
                               "confidence_by_arm": confidence_by_arm, "state_counts": counts,
                               "by_class": {name:{state:counter[state] for state in STATES} for name,counter in by_class.items()},
                               "source_instance_comparisons": comparison}
        confusion_examples, low_score_examples = [], []
        for item in pi_details:
            for arm, diagnostic in item['arms'].items():
                if diagnostic['matched']:
                    continue
                common = {'arm':arm, 'instance_id':item['instance_id'], 'source_name':item['source_name'],
                          'gt_class':item['class_name'], 'frozen_confidence':diagnostic['frozen_confidence']}
                wrong = diagnostic['highest_score_wrong_class_iou_ge05']
                if wrong is not None and wrong['score'] >= diagnostic['frozen_confidence']:
                    confusion_examples.append({**common, 'wrong_class_candidate':wrong})
                same = diagnostic['highest_score_same_class_iou_ge05']
                if same is not None and same['score'] < diagnostic['frozen_confidence']:
                    low_score_examples.append({**common, 'same_class_candidate':same})
        result = {"schema":"r04-saved-prediction-paired-error-analysis-v1", "status":"PASS_MATCH_PARITY_DIAGNOSTIC_ONLY",
                  "generated_utc":datetime.now(timezone.utc).isoformat(), "cohorts":cohorts,
                  "pi_ic_connector_details":pi_details, "count_parity":parity,
                  "pi_operating_class_confusion_examples":confusion_examples,
                  "pi_low_score_same_class_examples":low_score_examples,
                  "selection_unchanged":True, "recommended_arm_unchanged":aggregate["recommended_arm"],
                  "frozen_selection_sha256":aggregate["selection_sha256"], "thresholds_changed":False,
                  "gt_changed":False, "new_inference_calls":0, "raw_photo_reads":0,
                  "d455_verified":False, "fresh_final_test":False,
                  "interpretation":"Different arm-specific val-frozen thresholds. This descriptive comparison changes no selected model or GT. Saved low-score candidates can diagnose misses; their existence is not proof that a threshold change would improve accuracy.",
                  "source_sha256":self.hashes, "script_sha256":sha_bytes(Path(__file__).read_bytes())}
        # Refuse to interpret metrics if any input changed while CPU recount ran.
        for path, expected_sha in self.hashes.items():
            if sha_bytes(Path(path).read_bytes()) != expected_sha:
                raise ValueError("Source evidence changed during error analysis")
        output = self.root / "reports"
        output.mkdir(parents=True, exist_ok=True)
        (output / "error_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (output / "error_analysis.md").write_text(markdown(result),encoding="utf-8")
        print(json.dumps({"status":result["status"], "cohorts":{key:value["state_counts"] for key,value in cohorts.items()},
                          "pi_ic_connector_details":len(pi_details), "count_parity_records":len(parity)},ensure_ascii=False,indent=2))
        return result


def diagnostic_cell(row):
    if row is None:return "저장 후보 없음"
    return f"{row['class_name']} · score {row['score']:.4f} / IoU {row['iou']:.3f}"


def markdown(result):
    lines=["# R04 저장 예측의 객체별 오류 분석", "",
           "새 추론 없이 저장된 예측과 원본 GT를 비교했다. 두 arm의 기존 전역 confidence를 각각 그대로 사용했다. 독립적인 score순·같은 클래스·IoU≥0.5·1:1 매칭 결과가 원 평가 함수 및 저장된 TP/FP/FN과 모두 일치한 뒤 분석을 작성했다.", "",
           "모델 선택, threshold, 원본 GT는 변경하지 않았다. R03에서 이미 본 개발 holdout을 재사용한 진단이며 새로운 최종시험 또는 D455 실측이 아니다.", "",
           "## 두 모델이 같은 객체를 찾았는가", "",
           "| 자료 | 원본 GT | 둘 다 찾음 | 개선안만 | 기준안만 | 둘 다 놓침 |", "|---|---:|---:|---:|---:|---:|"]
    for cohort,payload in result["cohorts"].items():
        c=payload['state_counts'];lines.append(f"| {cohort} | {payload['gt_instances']} | " + " | ".join(str(c[state]) for state in STATES)+" |")
    lines += ["", "분모는 고유 원본 GT이며 오검출 수를 이 표의 분모에 섞지 않았다. 개선안만 찾은 수와 기준안만 찾은 수는 바뀐 학습 구성·선택 checkpoint·추론 방식·confidence를 함께 반영하므로 특정 변경 하나의 효과로 해석하지 않는다.", ""]
    for cohort,payload in result['cohorts'].items():
        conf=payload['confidence_by_arm']
        lines += [f"### {cohort} 클래스별", "", f"고정 confidence: 기준안 {conf['baseline']:.2f}, 개선안 {conf['improved']:.2f}.", "",
                  "| 클래스 | 전체 GT | 둘 다 찾음 | 개선안만 | 기준안만 | 둘 다 놓침 |", "|---|---:|---:|---:|---:|---:|"]
        for name in NAMES:
            counter=payload['by_class'][name]
            lines.append(f"| {name} | {sum(counter.values())} | " + " | ".join(str(counter[state]) for state in STATES)+" |")
        lines.append("")
    lines += ["## Pi3B 혼동 사례 후보", "",
              "해당 GT를 놓쳤고, 다른 클래스의 IoU≥0.5 상자가 기존 운영 confidence 이상인 경우만 추렸다. 아래 사례는 서로 배타적인 오류 원인 분류가 아니다.", "",
              "| 모델 | 원본 객체 | GT 클래스 | 겹친 다른 클래스 후보 |", "|---|---|---|---|"]
    for item in result['pi_operating_class_confusion_examples']:
        lines.append(f"| {item['arm']} | {item['instance_id']} · {item['source_name']} | {item['gt_class']} | {diagnostic_cell(item['wrong_class_candidate'])} |")
    if not result['pi_operating_class_confusion_examples']:
        lines.append("| — | 이 조건의 저장 후보 없음 | — | — |")
    lines += ["", "## Pi3B IC·커넥터의 원본 객체별 후보", "",
              "같은 클래스 후보는 최고 IoU 및 IoU≥0.5 안에서의 최고 score를 따로 기록했다. 다른 클래스 후보는 양의 overlap 중 IoU가 가장 큰 것을 표시한다. 이 표의 후보는 이미 crop/global NMS와 최대 1,000개 제한을 통과한 저장 후보이며, 네트워크의 모든 출력은 아니다. raw confidence floor는 0.001이다.", "",
              "score는 보정된 정답 확률이 아니다. 높은 점수의 다른 클래스 후보가 겹친다는 것은 클래스 혼동의 사례 후보이며, 원본 GT의 의미가 항상 옳다고 입증하지 않는다. 기존 미실장/unknown 정답은 임의로 바꾸지 않았다.", ""]
    for detail in result['pi_ic_connector_details']:
        lines += [f"### {detail['instance_id']} · {detail['source_name']} · {STATE_KO[detail['state']]}", "",
                  "| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |", "|---|---|---|---|---|"]
        for arm in ARMS:
            row=detail['arms'][arm]
            lines.append(f"| {arm} | {'찾음' if row['matched'] else '놓침'} | {diagnostic_cell(row['best_iou_same_class'])} | {diagnostic_cell(row['highest_score_same_class_iou_ge05'])} | {diagnostic_cell(row['largest_iou_wrong_class_positive_overlap'])} |")
        if detail.get('prior_review_context'):
            lines += ["", "J9는 이전 검토에서 microSD 소켓으로 지목한 위치다. 이번 스크립트는 사진을 읽지 않아 subtype을 새로 검증한 것은 아니다."]
        lines.append("")
    lines += ["## 재현 기록", "", "원본 instance ID별 전체 일반/Pi 비교, 매칭된 예측 좌표·score, 오류 분류, 후보 정보 및 파일 SHA-256은 `error_analysis.json`에 있다. 픽셀 프리뷰·원본 이미지·새 마스크는 만들지 않았다.", ""]
    return "\n".join(lines)


def self_test():
    from evaluate_r04 import operate
    def gt(cls,box):return {"class_id":cls,"bbox_xyxy":box}
    def pred(cls,box,score):return {"class_id":cls,"bbox_xyxy":box,"score":score}
    def sample(truth,predictions):return {"id":"synthetic", "group_id":"synthetic", "truth":truth, "predictions":predictions}
    class Tests(unittest.TestCase):
        def parity(self,s,confidence):
            actual=match_sample(s,confidence)
            expected=operate([s],confidence)
            self.assertEqual(actual['per_class_counts_tp_fp_fn'],expected['per_class_counts_tp_fp_fn'])
            return actual
        def test_threshold_equality_and_duplicate_detection(self):
            s=sample([gt(2,[0,0,10,10])],[pred(2,[0,0,10,10],.35),pred(2,[0,0,10,10],.36)])
            m=self.parity(s,.35);self.assertEqual(m['found'][0]['prediction_index'],1);self.assertEqual(m['per_class_counts_tp_fp_fn'][2],[1,1,0])
        def test_tied_iou_prefers_larger_gt_index(self):
            s=sample([gt(3,[0,0,10,10]),gt(3,[0,0,10,10])],[pred(3,[0,0,10,10],.9)])
            self.assertEqual(set(self.parity(s,.5)['found']),{1})
        def test_wrong_class_is_fp_and_source_fn(self):
            s=sample([gt(3,[0,0,10,10])],[pred(2,[0,0,10,10],.9)])
            m=self.parity(s,.5);self.assertEqual(m['per_class_counts_tp_fp_fn'][3],[0,0,1]);self.assertEqual(m['per_class_counts_tp_fp_fn'][2],[0,1,0])
        def test_low_score_is_candidate_not_operating_tp(self):
            truth=gt(2,[0,0,10,10]);s=sample([truth],[pred(2,[0,0,10,10],.003)])
            m=self.parity(s,.35);d=candidate_diagnosis(truth,s['predictions'],.35,m,0)
            self.assertEqual(d['diagnostic_category'],'same_class_iou_ge05_candidate_below_frozen_confidence');self.assertFalse(d['matched'])
        def test_no_predictions_and_empty_truth(self):
            self.parity(sample([gt(1,[0,0,10,10])],[]),.5)
            self.parity(sample([],[pred(0,[0,0,10,10],.9)]),.5)
        def test_source_refdes_preserves_original_suffix(self):
            self.assertEqual(source_refdes('connector J9'),'J9')
            self.assertEqual(source_refdes('"electrolytic capacitor" C3'),'C3')
            self.assertEqual(source_refdes('connector unknown'),'unknown')
        def test_randomized_parity(self):
            rng=random.Random(10)
            for _ in range(30):
                truths=[gt(rng.randrange(4),[x,y,x+10,y+12]) for x,y in [(rng.randrange(20),rng.randrange(20)) for __ in range(12)]]
                predictions=[]
                for obj in truths:
                    shift=rng.choice([0,0,2,20]);b=obj['bbox_xyxy']
                    predictions.append(pred(rng.choice([obj['class_id'],obj['class_id'],rng.randrange(4)]),[b[0]+shift,b[1],b[2]+shift,b[3]],rng.choice([.001,.05,.35,.6,.9])))
                for threshold in [.05,.35,.6]:self.parity(sample(truths,predictions),threshold)
    results=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not results.wasSuccessful():raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--self-test',action='store_true',help='Synthetic CPU matcher parity only; no actual holdout analysis')
    args=parser.parse_args()
    if args.self_test:return self_test()
    Analysis(args.root).run()


if __name__=='__main__':main()
