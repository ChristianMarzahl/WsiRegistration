import itertools
import pickle
import time
from enum import Enum
from functools import wraps
from inspect import currentframe
from pathlib import Path
import concurrent.futures
import functools

import cv2
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from openslide import OpenSlide
from PIL import Image
from probreg import cpd
from probreg import transformation as tf
from sklearn.neighbors import LocalOutlierFactor


class NodeOrientation(Enum):
    """[summary]
    Quad-Tree node orientation
    Args:
        Enum ([type]): [description]
    """
    TOP         = 0
    NORTH_WEST  = 1
    NORTH_EAST  = 2
    SOUTH_WEST  = 3
    SOUTH_EAST  = 4


class Point:
    """A point located at (x,y) in 2D space.

    Each Point object may be associated with a payload object.

    """

    def __init__(self, x, y, payload=None):
        self.x, self.y = x, y
        self.payload = payload

    def __repr__(self):
        return '{}: {}'.format(str((self.x, self.y)), repr(self.payload))
    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)

class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w/2, cx + w/2
        self.north_edge, self.south_edge = cy - h/2, cy + h/2

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)

    def create(cls, x, y, w, h):

        cx, cy = x + w // 2, y + h // 2

        return cls(cx, cy, w, h)

    def is_valid(self):
        return self.w > 0 and self.h > 0


    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""

        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (point_x >= self.west_edge and
                point_x <  self.east_edge and
                point_y >= self.north_edge and
                point_y < self.south_edge)

    def intersects(self, other):
        """Does Rect object other interesect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def draw(self, ax, c='k', lw=1, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)

class RegistrationQuadTree:
    """[summary]

    A class implementing the registration quadtree.

    """

    def __init__(self, source_slide_path:Path, target_slide_path:Path, source_boundary:Rect=None, target_boundary:Rect=None,
                        depth=0, target_depth=1, thumbnail_size=(2048, 2048),
                        run_async=False, node_orientation:NodeOrientation = NodeOrientation.TOP,
                        parent=None, homography:bool=True, filter_outliner:bool=False, **kwargs):
        """[summary]
        Init the current quadtree level

        Args:
            source_slide_path ([Path]): [source slide path as Path object]
            target_slide_path ([Path]): [target slide path as Path object]
            source_boundary ([Rect]): [source wsi region of interest]
            target_boundary ([Rect]): [target wsi region of interest]
            depth (int, optional): [current depth]. Defaults to 0.
            target_depth (int, optional): [maximal number of qt levels]. Defaults to 4.
            thumbnail_size (tuple, optional): [size of the thumbnail to extract keypoints from]. Defaults to (2048, 2048).
            run_async (bool, optional): [generate the quad-tree in a parallel maner]. Defaults to False.
            node_orientation (NodeOrientation, optional): [defines the four note orientations]. Defaults to NodeOrientation.TOP.
            parent ([RegistrationQuadTree], optional): [parent quad-tree]. Defaults to None.
            homography (bool, optional): [use cv.findHomography or a probReg]. Defaults to True.
            filter_outliner (bool, optional): [description]. Defaults to False.
        """


        kwargs.update({"run_async":run_async, "thumbnail_size":thumbnail_size, "homography":homography, 
                        "filter_outliner":filter_outliner, "target_depth":target_depth})
        
        self.kwargs = kwargs
        self.parent = parent
        self.node_orientation = node_orientation
        self.run_async = run_async
        self.thumbnail_size = thumbnail_size
        self.depth = depth
        self.target_depth = target_depth 
        self.source_slide_path = source_slide_path if isinstance(source_slide_path, Path) else Path(source_slide_path)
        self.target_slide_path = target_slide_path if isinstance(target_slide_path, Path) else Path(target_slide_path)

        self._target_slide_dimensions = None
        self._source_slide_dimensions = None 

        self.source_boundary = source_boundary
        self.target_boundary = target_boundary 
        
        if self.source_boundary == None: 
            source_slide = OpenSlide(str(source_slide_path))
            self._source_slide_dimensions = source_slide.dimensions
            self.source_boundary = Rect.create(Rect, 0, 0, source_slide.dimensions[0], source_slide.dimensions[1])

        if self.target_boundary == None: 
            target_slide = OpenSlide(str(target_slide_path))    
            self._target_slide_dimensions = target_slide.dimensions
            self.target_boundary = Rect.create(Rect, 0, 0, target_slide.dimensions[0], target_slide.dimensions[1])

        # start timer
        tic = time.perf_counter()

        if True: #self.run_async
            with concurrent.futures.ThreadPoolExecutor() as executor: # ProcessPoolExecutor  ThreadPoolExecutor
                source_func = functools.partial(RegistrationQuadTree.get_region_thumbnail, slide_path=self.source_slide_path, boundary=self.source_boundary, depth=self.depth + 1, size=self.thumbnail_size)
                target_func = functools.partial(RegistrationQuadTree.get_region_thumbnail, slide_path=self.target_slide_path, boundary=self.target_boundary, depth=self.depth + 1, size=self.thumbnail_size)

                submitted = {executor.submit(func) : area  for func, area in zip([source_func, target_func], ["source", "target"])}
                for future in concurrent.futures.as_completed(submitted):
                    try:
                        if submitted[future] == "source":
                            self.source_thumbnail, self.source_scale = future.result()
                        else:
                            self.target_thumbnail, self.target_scale = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (submitted[future], exc))
        else:
            self.source_thumbnail, self.source_scale = RegistrationQuadTree.get_region_thumbnail(slide_path=self.source_slide_path, boundary=self.source_boundary, depth=self.depth + 1, size=self.thumbnail_size)
            self.target_thumbnail, self.target_scale = RegistrationQuadTree.get_region_thumbnail(slide_path=self.target_slide_path, boundary=self.target_boundary, depth=self.depth + 1, size=self.thumbnail_size)
        
        self.ptsA, self.ptsB, self.matchedVis = self.extract_matching_points(self.source_thumbnail, self.target_thumbnail, source_scale=self.source_scale, target_scale=self.target_scale, **kwargs)

        if filter_outliner:
            self.ptsA, self.ptsB, self.scale_factors = self.filter_outliner(self.ptsA, self.ptsB)
        
        if homography:
            self.mean_reg_error, self.sigma2, self.q, self.tf_param = self._get_min_reg_error_hom(self.ptsA, self.ptsB)
        else:
            self.tf_param, self.sigma2, self.q = cpd.registration_cpd(self.ptsA, self.ptsB, 'affine')
            self.mean_reg_error = np.linalg.norm(self.tf_param.transform(self.ptsA)-self.ptsB, axis=1).mean()

            mean_reg_error, _, _, tf_param = self._get_min_reg_error_hom(self.ptsA, self.ptsB)

            if self.mean_reg_error > mean_reg_error:
                self.mean_reg_error = self.mean_reg_error
                self.tf_param = tf_param


        self.b = self.tf_param.b
        self.t = self.tf_param.t
        self.mpp_x_scale = self.tf_param.b[0][0]
        self.mpp_y_scale = self.tf_param.b[1][1]

        self.max_points = len(self.ptsA)
        self.points = self.ptsA
        # A flag to indicate whether this node has divided (branched) or not.
        self.divided = False

        self.nw, self.ne, self.se, self.sw = None, None, None, None
        if depth < target_depth:
            self.divide()

        # done stop timer
        toc = time.perf_counter()
        self.run_time = toc - tic


    def _get_min_reg_error_hom(self, ptsA, ptsB):
        mean_reg_error = 99999999

        tf_param = None
        for outline_filter in [0, cv2.RANSAC, cv2.RHO, cv2.LMEDS]:
            homography, mask = cv2.findHomography(self.ptsA, self.ptsB, outline_filter) 
            temp_tf_param = tf.AffineTransformation(homography[:2, :2], homography[:2, 2:].reshape(-1))

            temp_mean_reg_error = np.linalg.norm(temp_tf_param.transform(ptsA)-ptsB, axis=1).mean()
            if temp_mean_reg_error < mean_reg_error:
                mean_reg_error = temp_mean_reg_error
                sigma2, q, tf_param = -1, -1, temp_tf_param

        return mean_reg_error, sigma2, q, tf_param

    @property
    def source_thumbnail(self):

        if self._source_thumbnail is None:
            self.source_thumbnail, self.source_scale = self.get_region_thumbnail(self.source_slide, self.source_boundary, self.thumbnail_size)

        return self._source_thumbnail

    @source_thumbnail.setter 
    def source_thumbnail(self, thumbnail):
        self._source_thumbnail = thumbnail

    @property
    def target_thumbnail(self):

        if self._target_thumbnail is None:
            self._target_thumbnail, self.target_scale = self.get_region_thumbnail(self.target_slide, self.target_boundary, self.thumbnail_size)

        return self._target_thumbnail

    @target_thumbnail.setter 
    def target_thumbnail(self, thumbnail):
        self._target_thumbnail = thumbnail


    @property
    def source_slide_dimensions(self):

        if self._source_slide_dimensions is None:
            self.source_slide_dimensions = OpenSlide(str(self.source_slide_path)).dimensions
        
        return self._source_slide_dimensions

    @source_slide_dimensions.setter 
    def source_slide_dimensions(self, dimensions):

        self._source_slide_dimensions = dimensions

        if self.nw is not None: self.nw.source_slide_dimensions = dimensions
        if self.ne is not None: self.ne.source_slide_dimensions = dimensions
        if self.se is not None: self.se.source_slide_dimensions = dimensions
        if self.sw is not None: self.sw.source_slide_dimensions = dimensions


    @property
    def target_slide_dimensions(self):

        if self._target_slide_dimensions is None:
            self._target_slide_dimensions = OpenSlide(str(self.target_slide_path)).dimensions
        
        return self._target_slide_dimensions

    @target_slide_dimensions.setter 
    def target_slide_dimensions(self, dimensions):

        self._target_slide_dimensions = dimensions

        if self.nw is not None: self.nw.target_slide_dimensions = dimensions
        if self.ne is not None: self.ne.target_slide_dimensions = dimensions
        if self.se is not None: self.se.target_slide_dimensions = dimensions
        if self.sw is not None: self.sw.target_slide_dimensions = dimensions

    """
    @property
    def source_slide(self):

        if self._source_slide is None:
            raise Exception('Please set source_slide and target_slide after pickle load')

        return self._source_slide

    @source_slide.setter 
    def source_slide(self, slide):

        self._source_slide = slide

        if self.nw is not None: self.nw.source_slide = slide
        if self.ne is not None: self.ne.source_slide = slide
        if self.se is not None: self.se.source_slide = slide
        if self.sw is not None: self.sw.source_slide = slide

    @property
    def target_slide(self):

        if self._target_slide is None:
            raise Exception('Please set source_slide and target_slide after pickle load')

        return self._target_slide

    @target_slide.setter 
    def target_slide(self, slide):

        self._target_slide = slide

        if self.nw is not None: self.nw.target_slide = slide
        if self.ne is not None: self.ne.target_slide = slide
        if self.se is not None: self.se.target_slide = slide
        if self.sw is not None: self.sw.target_slide = slide
    """

    @property
    def source_path(self):
        return self.source_slide_path
            
    @property
    def target_path(self):
        return self.target_slide_path

    @property
    def source_name(self):
        return self.source_slide_path.stem
            
    @property
    def target_name(self):
        return  self.target_slide_path.stem
        
    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        
        s = ""
        if self.depth == 0:
            s +=  f"Source: {self.source_name} \n"
            s +=  f"Target: {self.target_name} \n"
        
        s += f"Source: {self.source_boundary} Target: {self.target_boundary}" + '\n' 
        s += sp + f'x: [{self.tf_param.b[0][0]:4.3f}, {self.tf_param.b[0][1]:4.3f}, {self.tf_param.t[0]:4.3f}], y: [{self.tf_param.b[1][0]:4.3f}, {self.tf_param.b[1][1]:4.3f}, {self.tf_param.t[1]:4.3f}]] error: {self.mean_reg_error:4.3f}'
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
                sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        source_cx, source_cy = self.source_boundary.cx, self.source_boundary.cy
        source_w, source_h = self.source_boundary.w / 2, self.source_boundary.h / 2

        # transform target bounding box

        new_xmin, new_ymin = self.transform_boxes([(self.source_boundary.west_edge, self.source_boundary.north_edge, 50, 50)])[0][:2]
        new_xmax, new_ymax = self.transform_boxes([(self.source_boundary.east_edge, self.source_boundary.south_edge, 50, 50)])[0][:2]

        # set new box coordinates withhin old limits
        new_ymin, new_xmin = max(new_ymin, 0), max(new_xmin, 0)
        new_ymax, new_xmax = min(new_ymax, self.target_slide_dimensions[1]), min(new_xmax, self.target_slide_dimensions[0])

        # transform target center
        target_cx, target_cy = self.transform_boxes([(self.source_boundary.cx, self.source_boundary.cy, 50, 50)])[0][:2]

        # create target boxes
        target_nw = Rect.create(Rect, new_xmin, new_ymin, target_cx - new_xmin, target_cy - new_ymin)
        target_sw = Rect.create(Rect, new_xmin, target_cy, target_cx - new_xmin, new_ymax - target_cy)

        target_ne = Rect.create(Rect, target_cx, new_ymin,  new_xmax - target_cx, target_cy - new_ymin)
        target_se = Rect.create(Rect, target_cx, target_cy, new_xmax - target_cx, new_ymax - target_cy)


        # create target boxes
        source_nw = Rect(source_cx - source_w/2, source_cy - source_h/2, source_w, source_h)
        source_sw = Rect(source_cx - source_w/2, source_cy + source_h/2, source_w, source_h)

        source_ne = Rect(source_cx + source_w/2, source_cy - source_h/2, source_w, source_h)
        source_se = Rect(source_cx + source_w/2, source_cy + source_h/2, source_w, source_h)


        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.

        qt_functions = {}
        
        if source_nw.is_valid() and target_nw.is_valid():
            qt_functions[NodeOrientation.NORTH_WEST] = functools.partial(RegistrationQuadTree, source_slide_path=self.source_slide_path, target_slide_path=self.target_slide_path, source_boundary=source_nw, target_boundary=target_nw,  depth=self.depth + 1, node_orientation=NodeOrientation.NORTH_WEST, parent=self, **self.kwargs)

        if source_ne.is_valid() and target_ne.is_valid():
            qt_functions[NodeOrientation.NORTH_EAST] = functools.partial(RegistrationQuadTree, source_slide_path=self.source_slide_path, target_slide_path=self.target_slide_path, source_boundary=source_ne, target_boundary=target_ne,  depth=self.depth + 1, node_orientation=NodeOrientation.NORTH_EAST, parent=self, **self.kwargs)

        if source_se.is_valid() and target_se.is_valid():
            qt_functions[NodeOrientation.SOUTH_EAST] = functools.partial(RegistrationQuadTree, source_slide_path=self.source_slide_path, target_slide_path=self.target_slide_path, source_boundary=source_se, target_boundary=target_se,  depth=self.depth + 1, node_orientation=NodeOrientation.SOUTH_EAST, parent=self, **self.kwargs)

        if source_sw.is_valid() and target_sw.is_valid():
            qt_functions[NodeOrientation.SOUTH_WEST] = functools.partial(RegistrationQuadTree, source_slide_path=self.source_slide_path, target_slide_path=self.target_slide_path, source_boundary=source_sw, target_boundary=target_sw,  depth=self.depth + 1, node_orientation=NodeOrientation.SOUTH_WEST, parent=self, **self.kwargs)

        if self.run_async == True:
            with concurrent.futures.ThreadPoolExecutor() as executor: # ProcessPoolExecutor  ThreadPoolExecutor

                submitted = {executor.submit(func) : area  for area, func in qt_functions.items()}
                for future in concurrent.futures.as_completed(submitted):
                    try:
                        if submitted[future] == NodeOrientation.NORTH_WEST:
                            self.nw = future.result()
                        elif submitted[future] == NodeOrientation.NORTH_EAST:
                            self.ne = future.result()
                        elif submitted[future] == NodeOrientation.SOUTH_EAST:
                            self.se = future.result()
                        elif submitted[future] == NodeOrientation.SOUTH_WEST:
                            self.sw = future.result()

                        self.divided = True
                    except Exception as exc:
                        print('%r generated an exception: %s' % (submitted[future], exc))
        else:
            for area, func in qt_functions.items():
                try:
                    if area == NodeOrientation.NORTH_WEST:
                        self.nw = func()
                    elif area == NodeOrientation.NORTH_EAST:
                        self.ne = func()
                    elif area == NodeOrientation.SOUTH_EAST:
                        self.se = func()
                    elif area == NodeOrientation.SOUTH_WEST:
                        self.sw = func()

                    self.divided = True

                except Exception as exc:
                    print('%r generated an exception: %s' % (submitted[future], exc))

    def draw_feature_points(self, num_sub_pic:int=5, figsize=(16, 16)):
        
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        fig.suptitle(f'{self.source_name} --> {self.target_name}')
        gs = fig.add_gridspec(5, num_sub_pic)

        f_ax_match = fig.add_subplot(gs[:2, :])
        f_ax_match.imshow(self.matchedVis)

        source_slide = OpenSlide(str(self.source_slide_path))
        target_slide = OpenSlide(str(self.target_slide_path))

        tf_temp = tf.AffineTransformation(self.tf_param.b, self.tf_param.t)
        
        for idx, (pA, pB) in enumerate(zip(self.ptsA[:num_sub_pic].copy(), self.ptsB[:num_sub_pic].copy())):
            size = 512

            transformed = pA.copy()
            pA = (pA + (self.source_boundary.west_edge, self.source_boundary.north_edge)).astype(int)
            pB = (pB + (self.target_boundary.west_edge, self.target_boundary.north_edge)).astype(int)

            transformed = (tf_temp.transform(transformed) + (self.target_boundary.west_edge, self.target_boundary.north_edge)).astype(int)
            pA = pA.astype(int)

            size_target_x, size_target_y = int(size * self.mpp_x_scale), int(size * self.mpp_y_scale)

            image_source, image_target, image_target_trans = np.zeros(shape=(1,1)), np.zeros(shape=(1,1)), np.zeros(shape=(1,1))

            pA_location = pA.astype(int) - (size // 2, size // 2)
            if size > 0:
                image_source = source_slide.read_region(location=pA_location, level=0, size=(size, size))

            pB_location = pB.astype(int) - (size_target_x // 2, size_target_y // 2)
            if size_target_x > 0 and size_target_y > 0:
                image_target = target_slide.read_region(location=pB_location, level=0, size=(size_target_x, size_target_y))

            trans_location = transformed - (size_target_x // 2, size_target_y // 2)
            if size_target_x > 0 and size_target_y > 0:
                image_target_trans = target_slide.read_region(location=trans_location, level=0, size=(size_target_x, size_target_y))

            ax = fig.add_subplot(gs[2, idx])
            ax.set_title(f'Source:  {pA}')
            ax.imshow(image_source)

            ax = fig.add_subplot(gs[3, idx])
            ax.set_title(f'Trans:  {transformed}')
            ax.imshow(image_target_trans)

            ax = fig.add_subplot(gs[4, idx])
            ax.set_title(f'GT:  {pB}')
            ax.imshow(image_target)
        
        if self.divided:
            sub_draws = []
            if self.nw is not None: sub_draws.append(self.nw.draw_feature_points(num_sub_pic, figsize))
            if self.ne is not None: sub_draws.append(self.ne.draw_feature_points(num_sub_pic, figsize))
            if self.se is not None: sub_draws.append(self.se.draw_feature_points(num_sub_pic, figsize))
            if self.sw is not None: sub_draws.append(self.sw.draw_feature_points(num_sub_pic, figsize))

            return fig, sub_draws
        else:
            return fig, None


    def filter_boxes(self, boxes):
        """[summary]
            Filter boxes that are not visibile in the quadtree level 
        Args:
            boxes ([type]): [Array of boxes: [xc, cy, w, h]]

        Returns:
            [type]: [description]
        """

        boxes = boxes[((boxes[:, 0] > self.source_boundary.west_edge) & (boxes[:, 0] < self.source_boundary.east_edge))]
        boxes = boxes[((boxes[:, 1] > self.source_boundary.north_edge) & (boxes[:, 1] < self.source_boundary.south_edge))]

        return boxes

    def transform_boxes(self, boxes, max_depth=100):
        """[summary]
            Transform box coordinages from the soure to the target domain coordinate system
        Args:
            boxes ([type]): [Array of boxes: [xc, cy, w, h]]

        Returns:
            [type]: [description]
        """

        tf_temp = tf.AffineTransformation(self.tf_param.b, self.tf_param.t)

        result_boxes = []
        for box in boxes:
            box = np.array(box)
            point = Point(box[0], box[1])
            
            #if self.nw is not None and self.nw.mean_reg_error <= self.mean_reg_error and self.nw.source_boundary.contains(point) and self.nw.depth <= max_depth: #q
            if self.nw is not None and self.nw.source_boundary.contains(point) and self.nw.depth <= max_depth:
                box = self.nw.transform_boxes([box], max_depth)[0]   
            #elif self.ne is not None and self.ne.mean_reg_error <= self.mean_reg_error and self.ne.source_boundary.contains(point) and self.ne.depth <= max_depth:
            elif self.ne is not None and self.ne.source_boundary.contains(point) and self.ne.depth <= max_depth:
                box = self.ne.transform_boxes([box], max_depth)[0] 
            #elif self.se is not None and self.se.mean_reg_error <= self.mean_reg_error and self.se.source_boundary.contains(point) and self.se.depth <= max_depth:
            elif self.se is not None and self.se.source_boundary.contains(point) and self.se.depth <= max_depth:
                box = self.se.transform_boxes([box], max_depth)[0] 
            #elif self.sw is not None and self.sw.mean_reg_error <= self.mean_reg_error and self.sw.source_boundary.contains(point) and self.sw.depth <= max_depth:
            elif self.sw is not None and self.sw.source_boundary.contains(point) and self.sw.depth <= max_depth:
                box = self.sw.transform_boxes([box], max_depth)[0] 
            else:

                source_boxes = box[:2] - (self.source_boundary.west_edge, self.source_boundary.north_edge) 
                transformed_xy = tf_temp.transform(source_boxes) + (self.target_boundary.west_edge, self.target_boundary.north_edge)
                transformed_wh = box[2:] * np.array([self.mpp_x_scale, self.mpp_y_scale])

                box = np.hstack([transformed_xy, transformed_wh])

            result_boxes.append(box)

        #if self.depth == 0:
            #result_boxes = np.array(list(itertools.chain(*result_boxes)))

        return result_boxes
        
    def draw_annotations(self, boxes, figsize=(16, 16), num_sub_pic:int=5):
        """[summary]
        Draw annotations on patches from the source and target slide
        Args:
            boxes ([type]): Array of boxes: [xc, cy, w, h]]
            figsize (tuple, optional): description. Defaults to (16, 16).
            num_sub_pic (int, optional): description. Defaults to 5.

        Returns:
            [type]: Array of figures for each level
        """

        source_boxes = boxes.copy()
        source_boxes = self.filter_boxes(source_boxes)
        target_boxes = self.transform_boxes(source_boxes)
        
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        fig.suptitle(f'{self.source_name} --> {self.target_name}')
        gs = fig.add_gridspec(4, num_sub_pic)

        f_ax_match = fig.add_subplot(gs[:2, :])
        f_ax_match.imshow(self.matchedVis)

        source_slide = OpenSlide(str(self.source_slide_path))
        target_slide = OpenSlide(str(self.target_slide_path))
        
        for idx, (source_box, target_box)  in enumerate(zip(source_boxes[:num_sub_pic], target_boxes[:num_sub_pic])):
            size = 512

            pA = np.array(source_box[:2]).astype(int)
            source_anno_width, source_anno_height = source_box[2:4]
            source_x1, source_y1 = (size / 2) - source_anno_width / 2, (size / 2)  - source_anno_height / 2
            
            
            transformed = target_box[:2].astype(int)
            size_target_x, size_target_y = abs(int(size * self.mpp_x_scale)), abs(int(size * self.mpp_y_scale))
            
            target_anno_width, target_anno_height = int(source_anno_width * self.mpp_x_scale), int(source_anno_height * self.mpp_y_scale)
            target_x1, target_y1 = (size_target_x / 2) - target_anno_width / 2, (size_target_y / 2)  - target_anno_height / 2
            

            image_source, image_target_trans = np.zeros(shape=(1,1)), np.zeros(shape=(1,1))

            pA_location = pA.astype(int) - (size // 2, size // 2)
            if size > 0:
                image_source = source_slide.read_region(location=pA_location, level=0, size=(size, size))

            trans_location = transformed - (size_target_x // 2, size_target_y // 2)
            if size_target_x > 0 and size_target_y > 0:
                image_target_trans = target_slide.read_region(trans_location, level=0, size=(size_target_x, size_target_y))


            ax = fig.add_subplot(gs[2, idx])
            ax.set_title(f'Source:  {pA}')
            ax.imshow(image_source)            
            rect = patches.Rectangle((source_x1, source_y1), source_anno_width, source_anno_height, 
                                linewidth=3, edgecolor='m', facecolor='none')
            ax.add_patch(rect)
            

            ax = fig.add_subplot(gs[3, idx])
            ax.set_title(f'Trans:  {transformed}')
            ax.imshow(image_target_trans)
            rect = patches.Rectangle((target_x1, target_y1), target_anno_width, target_anno_height, 
                                 linewidth=3, edgecolor='m', facecolor='none')
            ax.add_patch(rect)

        
        if self.divided:
            sub_draws = []
            if self.nw is not None: sub_draws.append(self.nw.draw_annotations(boxes, figsize, num_sub_pic))
            if self.ne is not None: sub_draws.append(self.ne.draw_annotations(boxes, figsize, num_sub_pic))
            if self.se is not None: sub_draws.append(self.se.draw_annotations(boxes, figsize, num_sub_pic))
            if self.sw is not None: sub_draws.append(self.sw.draw_annotations(boxes, figsize, num_sub_pic))

            return fig, sub_draws
        else:
            return fig, None

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.source_boundary.draw(ax)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)
          
    def filter_outliner(self, ptsA, ptsB):
        
        scales =  ptsA / ptsB
        
        #if self.parent is None:
        inliners = LocalOutlierFactor(n_neighbors=int(len(scales) * 0.25)).fit_predict(scales) == 1
        #else:
        #    inliners = scales[:, 0] > min(self.parent.scale_factors[:, 0]) & scales[:, 0] < max(self.parent.scale_factors[:, 0]) & scales[:, 1] > min(self.parent.scale_factors[:, 1]) & scales[:, 1] < max(self.parent.scale_factors[:, 1])

        return ptsA[inliners], ptsB[inliners], ptsA[inliners] / ptsB[inliners]

    def _get_detector_matcher(self, point_extractor="orb", maxFeatures:int=500, crossCheck:bool=False, flann:bool=False, **kwargs):

        kwargs.update({"point_extractor":point_extractor, "maxFeatures":maxFeatures, "crossCheck":crossCheck, "flann":flann})
    
        if point_extractor == "orb":
            detector = cv2.ORB_create(maxFeatures)
            norm = cv2.NORM_HAMMING
        elif point_extractor == "sift":
            detector = cv2.SIFT_create() # maxFeatures
            norm = cv2.NORM_L2
        else:
            return None, None
        
        if flann:
            if norm == cv2.NORM_L2:
                flann_params = dict(algorithm = 1, trees = 5)
            else:
                flann_params= dict(algorithm = 6,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
            matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        else:
            matcher = cv2.BFMatcher(norm, crossCheck)
        return detector, matcher

    def _filter_matches(self, kp1, kp2, matches, ratio = 0.75, **kwargs):
        kwargs.update({"ratio":ratio})

        mkp1, mkp2, good = [], [], []
        for match in matches:
            if len(match) < 2:
                break
            
            m, n = match
            if m.distance < n.distance * ratio:
                good.append([m])
                mkp1.append(np.array(kp1[m.queryIdx].pt))
                mkp2.append(np.array(kp2[m.trainIdx].pt))

        return mkp1, mkp2, good 

    def extract_matching_points(self, source_image, target_image,  
                    debug=False, 
                    source_scale:[tuple]=[(1,1)], 
                    target_scale:[tuple]=[(1,1)],
                    point_extractor:callable="orb",
                    use_gray:bool=False, 
                    **kwargs
                    ):
        kwargs.update({"debug":debug, "point_extractor":point_extractor, "use_gray":use_gray})

        source_scale = np.array(source_scale)
        target_scale = np.array(target_scale)

        source_image = np.array(source_image) if type(source_image) == Image.Image else source_image
        target_image = np.array(target_image) if type(target_image) == Image.Image else target_image
        
        if callable(point_extractor):
            kpsA_ori, descsA, kpsB, descsB, matches = point_extractor(source_image, target_image)
        else:
            detector, matcher = self._get_detector_matcher(**kwargs)

            kpsA_ori, descsA = detector.detectAndCompute(source_image, None) if use_gray == False else detector.detectAndCompute(cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY), None)
            kpsB_ori, descsB = detector.detectAndCompute(target_image, None) if use_gray == False else detector.detectAndCompute(cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY), None)

            matches = matcher.knnMatch(descsA, descsB, k=2)
            kpsA, kpsB, matches = self._filter_matches(kpsA_ori, kpsB_ori, matches, **kwargs)

        # check to see if we should visualize the matched keypoints
        matchedVis = None
        if debug:
            #matchedVis = cv2.drawMatches(source_image, kpsA, target_image, kpsB, matches, None)
            matchedVis = cv2.drawMatchesKnn(source_image, kpsA_ori, target_image, kpsB_ori, matches, 
                                    None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
        ptsA, ptsB = [], []

        for ptA, ptB in zip(kpsA, kpsB):
            # scale points 
            for s_scale, t_scale in zip(source_scale, target_scale):
                ptA *= s_scale 
                ptB *= t_scale 

            ptsA.append(ptA)
            ptsB.append(ptB)

        return np.array(ptsA), np.array(ptsB), matchedVis


    @staticmethod
    def get_region_thumbnail(slide_path:Path, boundary:Rect, depth:int=0, size=(2048, 2048)):

        scale = []

        slide = OpenSlide(str(slide_path))
        downsample = max(*[dim / thumb for dim, thumb in zip((boundary.w, boundary.h), (size[0] * depth, size[1] * depth))])        
        level = slide.get_best_level_for_downsample(downsample)

        downsample = slide.level_downsamples[level]

        x, y, w, h = int(boundary.west_edge), int(boundary.north_edge), int(boundary.w / downsample), int(boundary.h / downsample)
        scale.append(np.array((boundary.w, boundary.h)) / (w, h))

        tile = slide.read_region((x, y), level, (w, h))

        thumb = Image.new('RGB', tile.size, '#ffffff')
        thumb.paste(tile, None, tile)
        thumb.thumbnail(size, Image.ANTIALIAS)
        scale.append(np.array([w, h]) / thumb.size)

        return thumb, scale

    def __getstate__(self):

        attributes = self.__dict__.copy()

        del attributes['matchedVis']
        del attributes['_source_thumbnail']
        del attributes['_target_thumbnail']
        del attributes['tf_param']
        return attributes

    def __setstate__(self, state):

        self.__dict__ = state

        self._source_thumbnail = None
        self._target_thumbnail = None
        self.matchedVis = None

        self.tf_param = tf.AffineTransformation(self.__dict__["b"], self.__dict__["t"])


