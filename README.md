# Robust quad-tree based registration on whole slide images

[![PyPI version fury.io](https://badge.fury.io/py/qt-wsi-registration.svg)](https://pypi.python.org/pypi/qt-wsi-registration/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)



This is a library that implements a quad-tree based registration on whole slide images.


## Core features

* Whole Slide Image support
* Robust and fast
* Rigid and non-rigid transformation

## Additional Requirements

[Install OpennSlide](https://openslide.org/download/)


## Notebooks

Example notebooks are in the demo folder or  [![Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github//ChristianMarzahl/WsiRegistration).


## Ho-To:


Import package and create Quad-Tree.
```python
import qt_wsi_reg.registration_tree as registration

parameters = {
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
    "num_workers: 2,
    "thumbnail_size": (1024, 1024)
}

qtree = registration.RegistrationQuadTree(source_slide_path=Path("examples/4Scanner/Aperio/Cyto/A_BB_563476_1.svs"), target_slide_path="examples/4Scanner/Aperio/Cyto/A_BB_563476_1.svs", **parameters)

```

Show some registration debug information.

```python
qtree.draw_feature_points(num_sub_pic=5, figsize=(10, 10))
```

Show annotations on the source and target image in the format:

[["center_x", "center_y", "anno_width", "anno_height"]] 
```python
annos = np.array([["center_x", "center_y", "anno_width", "anno_height"]])
qtree.draw_annotations(annos, num_sub_pic=5, figsize=(10, 10))

```


Transform coordinates

```python
box = [source_anno.center_x, source_anno.center_y, source_anno.anno_width, source_anno.anno_height]

trans_box = qtree.transform_boxes(np.array([box]))[0]

```

