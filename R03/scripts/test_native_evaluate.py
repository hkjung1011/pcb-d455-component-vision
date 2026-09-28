"""CPU-only regression checks for native-coordinate evaluation, not model quality."""
import copy
import json
import unittest

import numpy as np
from pycocotools import mask as mu

from native_evaluate import bbox_iou, coco_score, encode_native_mask, operate


def ground_truth(categories=None):
    return {'info': {}, 'images': [{'id': 1, 'width': 120, 'height': 48}],
            'categories': categories or [{'id': 1, 'name': 'component'}], 'annotations': []}


def annotation(index=1, category=1, box=(10, 10, 30, 30)):
    x1, y1, x2, y2 = box
    return {'id': index, 'image_id': 1, 'category_id': category,
            'bbox': [x1, y1, x2-x1, y2-y1], 'area': (x2-x1)*(y2-y1),
            'iscrowd': 0, 'segmentation': [[x1,y1,x2,y1,x2,y2,x1,y2]]}


class CocoEvaluationTests(unittest.TestCase):
    def test_perfect_native_box_and_mask_are_one(self):
        gt = ground_truth(); gt['annotations'] = [annotation()]
        pred_box = [{'image_id': 1, 'category_id': 1, 'score': .9, 'bbox': [10,10,20,20]}]
        mask = np.zeros((48,120), dtype=np.float32); mask[10:30,10:30] = 1
        pred_mask = [{'image_id': 1, 'category_id': 1, 'score': .9,
                      'segmentation': encode_native_mask(mask,48,120)}]
        originals = copy.deepcopy((gt,pred_box,pred_mask))
        self.assertAlmostEqual(coco_score(gt,pred_box)['ap50_95_max100'],1.)
        self.assertAlmostEqual(coco_score(gt,pred_mask,'segm')['ap50_95_max100'],1.)
        self.assertEqual((gt,pred_box,pred_mask),originals)

    def test_unsorted_category_ids_and_missing_class_are_not_mislabelled(self):
        gt = ground_truth([{'id':5,'name':'present'},{'id':2,'name':'absent'}])
        gt['annotations'] = [annotation(category=5)]
        predictions = [{'image_id':1,'category_id':5,'score':.9,'bbox':[10,10,20,20]}]
        result = coco_score(gt,predictions)
        self.assertAlmostEqual(result['per_class']['present']['ap50_95_max100'],1.)
        self.assertIsNone(result['per_class']['absent']['ap50_95_max100'])

    def test_dense_150_instances_distinguish_max100_and_max300(self):
        gt = ground_truth(); gt['images'][0].update(width=160,height=100)
        box_predictions=[]; mask_predictions=[]
        for i in range(150):
            x,y = (i%16)*10,(i//16)*10
            gt['annotations'].append(annotation(i+1,box=(x,y,x+4,y+4)))
            box_predictions.append({'image_id':1,'category_id':1,'score':1-i*.001,'bbox':[x,y,4,4]})
            binary=np.zeros((100,160),dtype=np.uint8); binary[y:y+4,x:x+4]=1
            mask_predictions.append({'image_id':1,'category_id':1,'score':1-i*.001,
                                     'segmentation':encode_native_mask(binary,100,160)})
        for kind,predictions in [('bbox',box_predictions),('segm',mask_predictions)]:
            with self.subTest(kind=kind):
                result=coco_score(gt,predictions,kind)
                self.assertAlmostEqual(result['ap50_95_max100'],67/101,places=7)
                self.assertAlmostEqual(result['ap50_95_max300'],1.)

    def test_empty_predictions_are_zero_for_present_class(self):
        gt=ground_truth(); gt['annotations']=[annotation()]
        for kind in ('bbox','segm'):
            result=coco_score(gt,[],kind)
            self.assertEqual(result['ap50_95_max100'],0.)

    def test_empty_truth_has_undefined_ap_and_json_safe_values(self):
        result=coco_score(ground_truth(),[])
        self.assertIsNone(result['ap50_95_max100'])
        json.dumps(result,allow_nan=False)

    def test_native_mask_shape_dtype_and_binary_value_checks(self):
        mask=np.zeros((48,120),dtype=bool);mask[3:7,5:9]=True
        encoded=encode_native_mask(mask,48,120)
        self.assertEqual(mu.area(encoded),16)
        np.testing.assert_array_equal(mu.decode(encoded),mask)
        self.assertEqual(mu.area(encode_native_mask(np.zeros((48,120)),48,120)),0)
        for invalid in [np.empty((0,0)),np.zeros((120,48)),np.full((48,120),.3),np.full((48,120),np.nan)]:
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                encode_native_mask(invalid,48,120)


class OperatingPointTests(unittest.TestCase):
    @staticmethod
    def sample(truth,predictions,identifier='one'):
        return {'id':identifier,'group_id':'board1','source':'test','truth':truth,'predictions':predictions}

    def test_class_aware_one_to_one_matching_and_confidence(self):
        truth=[{'class_id':0,'bbox_xyxy':[0,0,10,10]}, {'class_id':1,'bbox_xyxy':[20,0,30,10]}]
        predictions=[{'class_id':0,'bbox_xyxy':[0,0,10,10],'score':.9},
                     {'class_id':0,'bbox_xyxy':[0,0,10,10],'score':.8},
                     {'class_id':0,'bbox_xyxy':[20,0,30,10],'score':.7},
                     {'class_id':1,'bbox_xyxy':[20,0,30,10],'score':.1}]
        result=operate([self.sample(truth,predictions)],.5,2)
        self.assertEqual(result['per_class_counts_tp_fp_fn'],[[1,2,0],[0,0,1]])
        self.assertAlmostEqual(result['precision'],1/3)
        self.assertEqual(result['recall'],.5)

    def test_negative_image_fp_rate_is_image_level(self):
        fp={'class_id':0,'bbox_xyxy':[0,0,10,10],'score':.8}
        samples=[self.sample([], [fp,fp]),self.sample([], [],'two')]
        result=operate(samples,.5,1)
        self.assertEqual(result['negative_false_positive_image_rate'],.5)
        self.assertEqual(result['negative_images_with_false_positive'],1)
        self.assertEqual(result['fp'],2)
        self.assertIsNone(result['recall'])

    def test_native_short_side_bin_boundaries(self):
        truth=[{'class_id':0,'bbox_xyxy':[i*50,0,i*50+s,s]} for i,s in enumerate([7,8,16,32])]
        predictions=[dict(t,score=.9) for t in truth]
        result=operate([self.sample(truth,predictions)],.5,1)
        for label in ['lt8','8to16','16to32','ge32']:
            self.assertEqual(result['size_recall'][label],{'detected':1,'total':1,'recall':1.})

    def test_half_open_box_iou(self):
        self.assertEqual(bbox_iou([0,0,10,10],[10,0,20,10]),0)
        self.assertAlmostEqual(bbox_iou([0,0,10,10],[5,0,15,10]),1/3)


if __name__=='__main__':
    unittest.main()
