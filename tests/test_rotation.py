import unittest
import sys
from pathlib import Path
import concurrent.futures
import qt_wsi_reg.registration_tree as registration
from openslide import OpenSlide
import functools
import cv2
import numpy as np

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

                
        base_path = Path(r"examples/Rotation/")

        """
        img_path = base_path / Path(r"0.png")
        img = Image.open(img_path)

        for angle in [45, 90+45, 180+45]:
            rot_img =  img.rotate(angle, fillcolor=(255, 255, 255), expand=True)
            rot_img.save(f"{angle}.png")
        """

        slide_paths = [(source, target) for source, target in 
                        zip([Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"), Path(r"0.png"),], 
                        [Path(r"0.png"), Path(r"45.png"), Path(r"90.png"), Path(r"135.png"), Path(r"180.png"), Path(r"225.png"), Path(r"270.png"),])]


        for soure_path, targert_path in slide_paths[2:]:

            qtree = registration.RegistrationQuadTree(source_slide_path=base_path/soure_path, target_slide_path=base_path/targert_path, **self.parameters)
            #qtree.draw_feature_points(num_sub_pic=5, figsize=(10, 10))[0].show()
            #print("")
            angle = qtree.get_rotation_angle
            gt_angle = int(targert_path.stem)

            self.assertTrue(int(angle) in range(gt_angle-5, gt_angle+5), f"Pred: {angle} GT: {gt_angle}")
            print(str(qtree))

if __name__ == '__main__':

    unittest.main()