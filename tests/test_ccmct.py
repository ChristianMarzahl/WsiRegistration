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
                "run_async": True,
                "num_workers": 2,
                "thumbnail_size": (1024, 1024)
            }


    def test_aperio(self):
        
        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/CCMCT").glob("*.svs"), 
                        Path(r"examples/4Scanner/Aperio/CCMCT").glob("*.svs"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))


    def test_axio(self):

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/CCMCT").glob("*.svs"), 
                        Path(r"examples/4Scanner/AxioScan/CCMCT").glob("*.tif"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))


    def test_20HT(self):

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/CCMCT").glob("*.svs"), 
                        Path(r"examples/4Scanner/NanoZoomer2.0HT/CCMCT").glob("*.ndpi"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))


    def test_S210(self):

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/CCMCT").glob("*.svs"), 
                        Path(r"examples/4Scanner/NanoZoomerS210/CCMCT").glob("*.ndpi"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))


if __name__ == '__main__':

    unittest.main()