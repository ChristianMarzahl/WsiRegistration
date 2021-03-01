import unittest
import sys
from pathlib import Path
import concurrent.futures
import qt_wsi_reg.registration_tree as registration
from openslide import OpenSlide
import functools
import pickle


class TestRegistrationMethods(unittest.TestCase):

    def setUp(self):
        self.parameters = {
                # feature extractor parameters
                "point_extractor": "sift",  #orb , sift
                "maxFeatures": 512, 
                "crossCheck": False, 
                "flann": False,
                "ratio": 0.6, 
                "use_gray": False,

                # QTree parameter 
                "homography": True,
                "filter_outliner": False,
                "debug": False,
                "target_depth": 1,
                "run_async": True,
                "thumbnail_size": (1024, 1024)
            }

        self.pickle_path = Path(f'Cyto.pickle')

        self.pic_results = Path(f"tests/results")
        self.pic_results.mkdir(exist_ok=True, parents=True)


    def tearDown(self):
        if self.pickle_path.exists():
            self.pickle_path.unlink()

    def test_draw_feature_points_aperio(self):

        parameters = self.parameters
        parameters["debug"] = True
        parameters["target_depth"] = 0

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/Cyto").glob("*.svs"), 
                        Path(r"examples/4Scanner/Aperio/Cyto").glob("*.svs"))}

        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            l0_fig, _ = qtree.draw_feature_points(num_sub_pic=5, figsize=(10, 10))
            l0_fig.savefig(self.pic_results / f'{soure_path.stem}-{targert_path.stem}.png')  


    def test_pickle(self):

        parameters = self.parameters
        qtree = registration.RegistrationQuadTree(source_slide_path=Path("examples/4Scanner/Aperio/Cyto/A_BB_563476_1.svs"), 
                                                    target_slide_path="examples/4Scanner/Aperio/Cyto/A_BB_563476_1.svs", 
                                                    **parameters)

        input_qt_str = str(qtree)

        with open(str(self.pickle_path), 'wb') as handle:
            pickle.dump(qtree, handle, protocol=pickle.DEFAULT_PROTOCOL)


        new_qt = pickle.load(open(str(self.pickle_path), "rb" ))
        result_qt_str = str(new_qt)

        self.assertEqual(input_qt_str, result_qt_str)

    def test_aperio_async(self):
        
        parameters = self.parameters
        parameters["run_async"] = True

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/Cyto").glob("*.svs"), 
                        Path(r"examples/4Scanner/Aperio/Cyto").glob("*.svs"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))

    def test_aperio(self):
        
        parameters = self.parameters
        parameters["run_async"] = False

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/Cyto").glob("*.svs"), 
                        Path(r"examples/4Scanner/Aperio/Cyto").glob("*.svs"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree)) 


    def test_axio(self):

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/Cyto").glob("*.svs"), 
                        Path(r"examples/4Scanner/AxioScan/Cyto").glob("*.tif"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))


    def test_20HT(self):

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/Cyto").glob("*.svs"), 
                        Path(r"examples/4Scanner/NanoZoomer2.0HT/Cyto").glob("*.ndpi"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))


    def test_S210(self):

        slide_paths = {source : target for source, target in 
                        zip(Path(r"examples/4Scanner/Aperio/Cyto").glob("*.svs"), 
                        Path(r"examples/4Scanner/NanoZoomerS210/Cyto").glob("*.ndpi"))}


        for soure_path, targert_path in slide_paths.items():

            qtree = registration.RegistrationQuadTree(source_slide_path=soure_path, target_slide_path=targert_path, **self.parameters)
            print(str(qtree))


if __name__ == '__main__':

    unittest.main()