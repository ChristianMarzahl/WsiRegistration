import unittest
import sys
from pathlib import Path
import concurrent.futures
import qt_wsi_reg.registration_tree as registration
from openslide import OpenSlide
import functools


class TestRegistrationMethods(unittest.TestCase):

    def setUp(self):
        self.parameters = {
                # feature extractor parameters
                "point_extractor": "sift",  #orb , sift
                "maxFeatures": 2048, 
                "crossCheck": False, 
                "flann": False,
                "ratio": 0.7, 
                "use_gray": False,

                # QTree parameter 
                "homography": True,
                "filter_outliner": False,
                "debug": True,
                "target_depth": 0,
                "run_async": True,
                "thumbnail_size": (1024, 1024)
            }

        self.pickle_path = Path(f'IHC-HE.pickle')

        self.pic_results = Path(f"tests/results")
        self.pic_results.mkdir(exist_ok=True, parents=True)

    def test_SingleSlide(self):

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/IHC/SingleSlide/IHC").glob("*.tif"), 
                        Path(r"examples/IHC/SingleSlide/HE").glob("*.tif"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))

    def test_MultiSlide(self):

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/IHC/MultiSlide/IHC").glob("*.tif"), 
                        Path(r"examples/IHC/MultiSlide/HE").glob("*.tif"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))


if __name__ == '__main__':

    unittest.main()