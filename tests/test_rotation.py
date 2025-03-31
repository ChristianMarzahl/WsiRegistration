import unittest
import sys
from pathlib import Path
import concurrent.futures
import qt_wsi_reg.registration_tree as registration
import openslide
from openslide import OpenSlide
import functools
import cv2
import numpy as np
from probreg import transformation as tf

from PIL import Image

class TestRegistrationMethods(unittest.TestCase):

    def setUp(self):
         self.parameters = {
                # feature extractor parameters
                "point_extractor": "sift",  #orb , sift
                "maxFeatures": 512, 
                "crossCheck": False, 
                "flann": False,
                "ratio": 0.7, 
                "use_gray": False,

                # QTree parameter 
                "homography": True,
                "filter_outliner": False,
                "debug": True,
                "target_depth": 0,
                "run_async": False,
                "num_workers": 2,
                "thumbnail_size": (1024, 1024)
            }


    def test_rotation(self):

        self.pic_results = Path(f"tests/results/rotation")
        self.pic_results.mkdir(exist_ok=True, parents=True)

        base_path = Path(r"examples/Rotation/")

        slide_paths = [(source, target) for source, target in 
                        zip([Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"),], 
                        [Path(r"0.png"), Path(r"45.png"), Path(r"90.png"), Path(r"135.png"), Path(r"180.png"), Path(r"225.png"), Path(r"270.png"),])]


        for soure_path, target_path in slide_paths[:]:

            qtree = registration.RegistrationQuadTree(source_slide_path=base_path/soure_path, target_slide_path=base_path/target_path, **self.parameters)
            
            fig = qtree.draw_feature_points(num_sub_pic=5, figsize=(10, 10), patch_size=64)[0]
            fig.savefig(self.pic_results / f'{soure_path.stem}-{target_path.stem}.png') 

            angle = qtree.get_rotation_angle
            gt_angle = int(target_path.stem)

            #self.assertTrue(int(angle) in range(gt_angle-5, gt_angle+5), f"Pred: {angle} GT: {gt_angle}")
            #print(str(qtree))

if __name__ == '__main__':

    unittest.main()