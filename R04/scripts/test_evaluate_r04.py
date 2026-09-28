"""CPU tests for native/dual coordinates, fixed matching and dense COCO caps."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_r04 as e


class EvaluationTests(unittest.TestCase):
    def test_crop2048_is_translation_only(self):
        p = e.restore_prediction([100, 200, 300, 500], (819, 700, 2867, 2748), 2, .8)
        self.assertEqual(p['bbox_xyxy'], [919, 900, 1119, 1200])

    def test_tiny_crop_and_clip(self):
        self.assertEqual(e.tile_bounds(7, 5, 1024), [(0, 0, 5, 7)])
        p = e.restore_prediction([-3, -4, 7, 9], (20, 30, 25, 37), 0, .1)
        self.assertEqual(p['bbox_xyxy'], [20, 30, 25, 37])
        self.assertIsNone(e.restore_prediction([8, 1, 9, 2], (0, 0, 5, 7), 0, .1))

    def test_invalid_finite_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            e.restore_prediction([0, 0, float('nan'), 3], (0, 0, 5, 7), 0, .5)
        with self.assertRaises(ValueError):
            e.tile_bounds(20, 20, 1024, float('nan'))

    def test_edge_coverage_and_no_duplicate_bounds(self):
        bounds = e.tile_bounds(2049, 3001, 1024)
        self.assertEqual(len(bounds), len(set(bounds)))
        self.assertEqual(max(b[2] for b in bounds), 3001)
        self.assertEqual(max(b[3] for b in bounds), 2049)
        self.assertIn((819, 0, 1843, 1024), bounds)

    def test_dual_deduplicates_but_does_not_change_ground_truth(self):
        gt = {'id': 'one', 'objects': [{'class_id': 0, 'bbox_xyxy': [10, 20, 30, 40]}]}
        before = json.dumps(gt, sort_keys=True)
        p = e.restore_prediction([10, 20, 30, 40], (0, 0, 1024, 1024), 0, .8)
        q = e.restore_prediction([10, 20, 30, 40], (0, 0, 2048, 2048), 0, .7)
        merged = e.deduplicate([p, q])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['score'], .8)
        self.assertEqual(json.dumps(gt, sort_keys=True), before)
        self.assertEqual(len(e.inference_plan(600, 600, 'dual')), 2)

    def test_nms_is_class_aware_and_empty_safe(self):
        self.assertEqual(e.deduplicate([]), [])
        a = {'class_id': 0, 'score': .8, 'bbox_xyxy': [1, 1, 5, 5]}
        b = dict(a, class_id=1)
        self.assertEqual(len(e.deduplicate([a, b])), 2)

    def test_cross_table_and_one_to_one_matching(self):
        sample = {'id': 'a', 'group_id': 'g', 'truth': [
            {'class_id': 0, 'bbox_xyxy': [0, 0, 7, 20]},
            {'class_id': 2, 'bbox_xyxy': [50, 50, 82, 90]}], 'predictions': [
            {'class_id': 0, 'bbox_xyxy': [0, 0, 7, 20], 'score': .9},
            {'class_id': 0, 'bbox_xyxy': [0, 0, 7, 20], 'score': .8}]}
        result = e.operate([sample], .5)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 1))
        self.assertEqual(result['class_by_native_shortside_recall']['resistor']['lt8']['recall'], 1)
        self.assertEqual(result['class_by_native_shortside_recall']['ic']['ge32']['recall'], 0)
        self.assertIsNone(result['class_by_native_shortside_recall']['connector']['lt8']['recall'])

    def test_confidence_boundary_and_empty_pr(self):
        sample = {'id': 'a', 'group_id': 'g', 'truth': [], 'predictions': []}
        curve = e.threshold_curves([sample])
        self.assertEqual(curve['selected_global_confidence'], .95)
        self.assertFalse(curve['per_class_thresholds_applied'])
        self.assertEqual(len(curve['per_class_curves']['ic']), 19)

    def test_one_group_ci_refused_and_general_ci_allowed(self):
        row = {'id': 'a', 'group_id': 'g', 'counts_tp_fp_fn': [[1, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]}
        interval, reason = e.bootstrap_recall({'per_image': [row]}, 100, 'test', 'pi_test')
        self.assertIsNone(interval)
        self.assertIn('one', reason)
        row2 = copy.deepcopy(row); row2['group_id'] = 'other'
        interval, reason = e.bootstrap_recall({'per_image': [row, row2]}, 100, 'test', 'general_test')
        self.assertEqual(interval['lower'], .5)
        self.assertIsNone(reason)

    def test_ground_truth_conversion_is_immutable(self):
        row = {'id': 'a', 'objects': [{'class_id': 2, 'bbox_xyxy': [3, 5, 20, 30]}]}
        original = copy.deepcopy(row)
        annotations, truths = e.make_truth(row, 9, 0, 100, 100)
        self.assertEqual(annotations[0]['bbox'], [3, 5, 17, 25])
        self.assertEqual(truths[0]['bbox_xyxy'], row['objects'][0]['bbox_xyxy'])
        self.assertEqual(row, original)

    def test_dense_ap100_vs_ap300(self):
        gt = {'info': {}, 'images': [{'id': 1, 'width': 2000, 'height': 100}],
              'categories': [{'id': 1, 'name': 'resistor'}], 'annotations': []}
        preds = []
        for index in range(150):
            box = [index*10, 0, 5, 5]
            gt['annotations'].append({'id': index+1, 'image_id': 1, 'category_id': 1, 'bbox': box,
                                      'area': 25, 'iscrowd': 0})
            preds.append({'image_id': 1, 'category_id': 1, 'bbox': box, 'score': 1-index/1000})
        before = json.dumps([gt, preds], sort_keys=True)
        score = e.coco_score(gt, preds)
        self.assertGreater(score['ap50_95_max300'], .999)
        self.assertLess(score['ap50_95_max100'], .68)
        self.assertGreater(score['ap50_max300'], .999)
        self.assertEqual(json.dumps([gt, preds], sort_keys=True), before)

    def test_coco_empty_predictions(self):
        gt = {'info': {}, 'images': [{'id': 1, 'width': 20, 'height': 20}],
              'categories': [{'id': 1, 'name': 'resistor'}],
              'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [1, 1, 5, 5], 'area': 25, 'iscrowd': 0}]}
        score = e.coco_score(gt, [])
        self.assertEqual(score['ap50_max300'], 0)
        self.assertEqual(score['ap50_95_max100'], 0)

    def test_test_requires_frozen_confidence(self):
        args = ['--manifest', 'x.json', '--weights', 'x.pt', '--output', 'nonexistent-r04-test', '--split', 'test']
        with self.assertRaises(SystemExit):
            e.parse_args(args)
        parsed = e.parse_args(args+['--operating-conf', '.35'])
        self.assertEqual(parsed.operating_conf, .35)

    def test_cache_exact_iou_tie_and_float_confidence_boundary(self):
        samples = [{'id': 'ties', 'group_id': 'g', 'truth': [
            {'class_id': 0, 'bbox_xyxy': [0, 0, 8, 16]},
            {'class_id': 0, 'bbox_xyxy': [0, 0, 16, 32]}], 'predictions': [
            {'class_id': 0, 'bbox_xyxy': [0, 0, 16, 16], 'score': .35},
            {'class_id': 0, 'bbox_xyxy': [0, 0, 16, 16], 'score': .34}]}]
        cache = e._confidence_match_cache(samples)
        for threshold in [0, .05, .349999, .35, .35000000000000003, .95, 1.]:
            self.assertEqual(e._operate_from_cache(cache, threshold), e.operate(samples, threshold))
        chosen = e._operate_from_cache(cache, .35)
        self.assertEqual(chosen['size_recall']['8to16']['detected'], 0)
        self.assertEqual(chosen['size_recall']['16to32']['detected'], 1)

    def test_cache_empty_and_negative_images_full_result(self):
        samples = [{'id': 'empty', 'group_id': 'g', 'truth': [], 'predictions': []},
                   {'id': 'negative', 'group_id': 'g', 'truth': [], 'predictions': [
                       {'class_id': 3, 'bbox_xyxy': [1, 1, 2, 2], 'score': .1}]}]
        cache = e._confidence_match_cache(samples)
        for i in range(1, 20):
            self.assertEqual(e._operate_from_cache(cache, i/20), e.operate(samples, i/20))

    def test_cache_randomized_all_class_per_image_parity(self):
        rng = np.random.default_rng(204)
        truths, preds = [], []
        for i in range(30):
            origin = rng.uniform(0, 100, 2); side = rng.uniform(2, 40, 2)
            box = np.r_[origin, origin+side].tolist()
            truths.append({'class_id': i % 4, 'bbox_xyxy': box})
            preds.append({'class_id': i % 4, 'bbox_xyxy': box, 'score': float(rng.choice([.05, .35, .6, .95]))})
            preds.append({'class_id': (i+1) % 4, 'bbox_xyxy': box, 'score': float(rng.uniform())})
        samples = [{'id': 'random', 'group_id': 'g', 'truth': truths, 'predictions': preds}]
        cache = e._confidence_match_cache(samples)
        for i in range(1, 20):
            self.assertEqual(e._operate_from_cache(cache, i/20), e.operate(samples, i/20))


if __name__ == '__main__':
    unittest.main(verbosity=2)
