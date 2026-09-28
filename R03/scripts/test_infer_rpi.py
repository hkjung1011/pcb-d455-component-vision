"""CPU-only checks for original-coordinate inference and registry safety."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import infer_rpi as infer
from pcb_components.geometry import Rect, Tile, generate_tiles


class InferenceGeometryTests(unittest.TestCase):
    def test_restore_clips_to_crop_and_translates_original(self):
        tile=Tile(4,Rect(100,200,340,400),(800,1000))
        rows=infer.restore_box_predictions([[-10,20,50,250],[300,0,310,10]],[0,1],[.9,.8],tile,infer.PART_NAMES)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['bbox_xyxy'],[100.,220.,150.,400.])
        self.assertEqual(rows[0]['source_prediction_index'],0)

    def test_nonfinite_prediction_fails(self):
        tile=Tile(0,Rect(0,0,20,20),(20,20))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            infer.restore_box_predictions([[0,0,float('nan'),10]],[0],[.9],tile,infer.PART_NAMES)

    def test_class_aware_torchvision_nms(self):
        rows=[{'class_id':0,'confidence':.9,'bbox_xyxy':[0,0,20,20]},
              {'class_id':0,'confidence':.8,'bbox_xyxy':[1,1,21,21]},
              {'class_id':1,'confidence':.7,'bbox_xyxy':[0,0,20,20]}]
        self.assertEqual(infer.merge_boxes(rows),[rows[0],rows[2]])

    def test_roi_margin_clips_to_frame_and_no_board_gives_no_roi(self):
        rows=[{'bbox_xyxy':[10,20,90,80]}]
        self.assertEqual(infer.make_regions((90,100,3),rows,True,.1)[0].as_xyxy(),(2,14,98,86))
        self.assertEqual(infer.make_regions((90,100,3),[],True,.1),[])
        self.assertEqual(infer.make_regions((90,100,3),[],False)[0].as_xyxy(),(0,0,100,90))

    def test_native_tiles_are_cropped_before_model_and_restored(self):
        image=np.arange(100*140*3,dtype=np.uint16).reshape(100,140,3)
        class FakeModel:
            names=dict(enumerate(infer.PART_NAMES))
            def __init__(self):self.crops=[]
            def predict(self,crop,**kwargs):
                self.crops.append(crop.copy())
                return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[4.,5.,14.,15.]]),cls=torch.tensor([0.]),conf=torch.tensor([.9])))]
        model=FakeModel();cfg={'tile_size':64,'overlap':.25,'imgsz':64,'confidence':.5}
        args=SimpleNamespace(max_det=1000,device='cpu')
        output,meta=infer.predict_boxes(model,image,cfg,args,[Rect(0,0,140,100)],tiled=True)
        tiles=generate_tiles(image.shape,64,.25)
        self.assertEqual(len(tiles),6)
        for crop,tile in zip(model.crops,tiles):np.testing.assert_array_equal(crop,tile.extract(image))
        actual={tuple(row['bbox_xyxy']) for row in output}
        expected={(float(t.bounds.x1+4),float(t.bounds.y1+5),float(t.bounds.x1+14),float(t.bounds.y1+15)) for t in tiles}
        self.assertEqual(actual,expected)
        self.assertFalse(meta['pre_tiling_full_image_resized'])

    def test_empty_roi_never_calls_component_model(self):
        class FailModel:
            def predict(self,*a,**k):raise AssertionError('Must not invoke model')
        rows,meta=infer.predict_boxes(FailModel(),np.zeros((50,50,3)),{},SimpleNamespace(max_det=1000),[],True)
        self.assertEqual(rows,[])
        self.assertEqual(meta['crops'],[])

    def test_operating_threshold_is_after_prediction_and_nms(self):
        class FakeModel:
            names=dict(enumerate(infer.PART_NAMES))
            def predict(self,crop,**kwargs):
                self.kwargs=kwargs
                return [SimpleNamespace(boxes=SimpleNamespace(
                    xyxy=torch.tensor([[1.,1.,11.,11.],[20.,20.,30.,30.],[35.,35.,45.,45.]]),
                    cls=torch.tensor([0.,0.,0.]),conf=torch.tensor([.5,.25,.75])))]
        model=FakeModel();cfg={'imgsz':64,'confidence':.5}
        result,meta=infer.predict_boxes(model,np.zeros((50,50,3),dtype=np.uint8),cfg,
            SimpleNamespace(max_det=1000,device='cpu'),[Rect(0,0,50,50)])
        self.assertEqual(model.kwargs['conf'],.001)
        self.assertEqual(model.kwargs['iou'],.6)
        self.assertEqual(model.kwargs['max_det'],1000)
        self.assertEqual([r['confidence'] for r in result],[.5,.75])
        self.assertEqual(meta['predictions_after_cross_crop_nms'],3)
        self.assertEqual(meta['predictions_after_operating_threshold'],2)

    def test_registry_whole_accepts_zero_tile_and_validates_hash(self):
        with tempfile.TemporaryDirectory(dir=infer.ROOT/'reports') as directory:
            root=Path(directory);weight=root/'example.pt';weight.write_bytes(b'not-loaded-CPU-test-fixture')
            registry={'schema':'r03-inference-selection-v1',
                'board_detector':{'checkpoint':'example.pt','sha256':infer.sha256(weight),'confidence':.6,'imgsz':640},
                'parts':{'checkpoint':'example.pt','sha256':infer.sha256(weight),'confidence':.3,'imgsz':1024,'pipeline':'whole','tile_size':0}}
            path=root/'selection.json';path.write_text(json.dumps(registry))
            args=infer.parser().parse_args(['--input','sample.jpg','--output','new','--registry',str(path)])
            roles,selection=infer.resolve_settings(args)
            self.assertEqual(roles['parts']['tile_size'],0)
            self.assertEqual(roles['board_detector']['checkpoint'],str(weight.resolve()))
            self.assertEqual(selection['selection_status'],'REGISTRY_CONFIGURATION')
            registry['parts']['sha256']='0'*64;path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):infer.resolve_settings(args)

    def test_board_mask_png_restores_origin_hole_and_original_prediction_index(self):
        import cv2
        masks=np.zeros((2,20,30),dtype=np.float32)
        masks[1,4:10,6:14]=1
        masks[1,6,9]=0
        result=SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[1.,1.,10.,10.],[5.,3.,15.,11.]]),
            cls=torch.tensor([0.,0.]),conf=torch.tensor([.9,.8])),masks=SimpleNamespace(data=torch.from_numpy(masks)))
        class Boxes:
            xyxy=result.boxes.xyxy;cls=result.boxes.cls;conf=result.boxes.conf
            def __len__(self):return 2
        result.boxes=Boxes()
        model=SimpleNamespace(predict=lambda *a,**k:[result])
        with tempfile.TemporaryDirectory(dir=infer.ROOT/'reports') as folder:
            objects,meta=infer.predict_board_masks(model,np.zeros((20,30,3),dtype=np.uint8),
                {'imgsz':32,'confidence':.1},SimpleNamespace(max_det=1000,device='cpu'),Path(folder))
            self.assertEqual(meta['empty_masks_skipped'],1)
            self.assertEqual(len(objects),1)
            obj=objects[0]
            self.assertEqual(obj['source_prediction_index'],1)
            self.assertEqual(obj['mask']['bounds_xyxy'],[6,4,14,10])
            crop=cv2.imdecode(np.fromfile(Path(folder)/obj['mask']['file'],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            self.assertEqual(crop.shape,(6,8))
            self.assertEqual(crop[2,3],0)
            self.assertEqual(int(np.count_nonzero(crop)),47)
            self.assertEqual(obj['mask_scope'],'WHOLE_RASPBERRY_PI_BOARD_NOT_COMPONENT_MASK')


if __name__=='__main__':unittest.main()
