"""Synthetic checks for diagnostic rules and COCO cached-subset correctness."""
import unittest
from supplementary_analysis import classify, geometry, CocoSubsets, fp_diagnostics, subtotal

class Tests(unittest.TestCase):
    def test_tp_not_reclassified_unknown(self):
        box=[0,0,10,10];s={'id':'x','truth':[{'class_id':0,'bbox_xyxy':box}], 'predictions':[{'class_id':0,'bbox_xyxy':box,'score':.9}]}
        m={'x':{'excluded_source_objects':[{'source_type':'unknown','source_object_index':1,'bbox_xyxy':box}]}}
        result=fp_diagnostics([s],m,{'operating':{'confidence':.5,'tp':1,'fp':0,'fn':0}})
        self.assertEqual(result['fp_records'],[])
    def test_sameclass_priority(self):
        p={'class_id':0,'bbox_xyxy':[0,0,10,10]};t=[dict(p)];ex=[{'source_type':'unknown','source_object_index':1,'bbox_xyxy':p['bbox_xyxy']}]
        r=classify(p,t,[True],ex)['strict_iou']
        self.assertEqual(r['category'],'other_same_class');self.assertEqual(r['detail']['kind'],'duplicate');self.assertTrue(r['conditions']['unknown_overlap_ge_050'])
    def test_iou_vs_ioa_and_rest(self):
        p={'class_id':0,'bbox_xyxy':[0,0,2,2]};ex=[{'source_type':'led','source_object_index':1,'bbox_xyxy':[0,0,10,10]}]
        r=classify(p,[],[],ex)
        self.assertEqual(r['strict_iou']['category'],'other_background');self.assertEqual(r['sensitivity_ioa_pred']['category'],'known_non_target');self.assertEqual(subtotal('led'),'rest')
    def test_unknown_tie_priority(self):
        p={'class_id':0,'bbox_xyxy':[0,0,10,10]};ex=[{'source_type':'pads','source_object_index':0,'bbox_xyxy':p['bbox_xyxy']},{'source_type':'unknown','source_object_index':2,'bbox_xyxy':p['bbox_xyxy']}]
        self.assertEqual(classify(p,[],[],ex)['strict_iou']['category'],'source_unknown')
    def test_coco_subset_reuse_missing_class(self):
        gt={'images':[{'id':1,'width':20,'height':20},{'id':2,'width':20,'height':20}], 'categories':[{'id':i+1,'name':n} for i,n in enumerate(['resistor','capacitor','ic','connector'])], 'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[0,0,10,10],'area':100,'iscrowd':0},{'id':2,'image_id':2,'category_id':2,'bbox':[0,0,10,10],'area':100,'iscrowd':0}]}
        pred=[{'image_id':1,'category_id':1,'bbox':[0,0,10,10],'score':.9}]
        c=CocoSubsets(gt,pred);c.verify_subset([1]);c.verify_subset([2]);c.verify_subset([1,2])
        self.assertAlmostEqual(c.subset([1])['ap50_95_max300'],1)
        self.assertEqual(c.subset([2])['ap50_95_max300'],0)
    def test_operating_iou_tie_uses_high_gt(self):
        b=[0,0,10,10];s={'id':'x','truth':[{'class_id':0,'bbox_xyxy':b},{'class_id':0,'bbox_xyxy':b}],'predictions':[{'class_id':0,'bbox_xyxy':b,'score':.9},{'class_id':0,'bbox_xyxy':b,'score':.8},{'class_id':0,'bbox_xyxy':b,'score':.7}]}
        r=fp_diagnostics([s],{'x':{'excluded_source_objects':[]}}, {'operating':{'confidence':.5,'tp':2,'fp':1,'fn':0}})
        self.assertEqual(r['fp_records'][0]['strict_iou']['detail']['gt_index'],0)

if __name__=='__main__':unittest.main()
