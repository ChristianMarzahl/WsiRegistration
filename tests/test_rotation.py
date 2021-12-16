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
                "target_depth": 1,
                "run_async": False,
                "num_workers": 2,
                "thumbnail_size": (1024, 1024)
            }

    def test_rotation_math(self):
        source_slide_path = base_path = Path(r"examples/Rotation/45.png")

        
        thumbnail_size = 512
        angle = -45
        source_thumbnail = openslide.open_slide(str(source_slide_path)).get_thumbnail((512,512))
        #source_thumbnail.save("source.png")

        target_thumbnail = source_thumbnail.rotate(angle, fillcolor=(255, 0, 0), expand=True)
        #target_thumbnail.save("target.png")

        #qtree = registration.RegistrationQuadTree(source_slide_path="source.png", target_slide_path="target.png", **self.parameters)

        # Source -> Target
        M = registration.RegistrationQuadTree.get_rotation_matrix_from_angle(-angle, width=source_thumbnail.width, height=source_thumbnail.height)
        H = registration.RegistrationQuadTree.get_inv_rotation_matrix_from_angle(-angle, width=source_thumbnail.width, height=source_thumbnail.height)

        tf_angle_target = tf.AffineTransformation(M[:2, :2], M[:2, 2:].reshape(-1))
        result = tf_angle_target.transform([247, 162])

        tf_angle_target_inv = tf.AffineTransformation(H[:2, :2], H[:2, 2:].reshape(-1))
        result_2 = tf_angle_target_inv.transform(result)

        # Target -> Source


        print("")

    def test_rotation(self):
        base_path = Path(r"examples/Rotation/")

        slide_paths = [(source, target) for source, target in 
                        zip([Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"),], 
                        [Path(r"0.png"), Path(r"45.png"), Path(r"90.png"), Path(r"135.png"), Path(r"180.png"), Path(r"225.png"), Path(r"270.png"),])]


        for soure_path, targert_path in slide_paths[1:]:

            qtree = registration.RegistrationQuadTree(source_slide_path=base_path/soure_path, target_slide_path=base_path/targert_path, **self.parameters)
            qtree.draw_feature_points(num_sub_pic=5, figsize=(10, 10), patch_size=64)[0].show()
            #print("")
            angle = qtree.get_rotation_angle
            gt_angle = int(targert_path.stem)

            #self.assertTrue(int(angle) in range(gt_angle-5, gt_angle+5), f"Pred: {angle} GT: {gt_angle}")
            print(str(qtree))

if __name__ == '__main__':

    unittest.main()