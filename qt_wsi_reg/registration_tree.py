import itertools
import pathlib
import pickle
import math
import random
import time
from enum import Enum
from functools import wraps
from inspect import currentframe
from pathlib import Path
import concurrent.futures
import functools
import operator

from typing import Dict, Tuple, Sequence, List

from numpy.linalg import inv
import cv2
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import openslide 
from PIL import Image
from probreg import cpd
from probreg import transformation as tf
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.patches import Polygon
from scipy import interpolate

class OpenSlideWrapper(openslide.OpenSlide):
    def __init__(self, slide_path):
        super().__init__(slide_path)
        self.slide_path = slide_path  # Store path for later reconstruction
    
    def __reduce__(self):
        # Define how to pickle the object
        return (self.__class__, (self.slide_path,))
    
class ImageSlideWrapper(openslide.ImageSlide):
    def __init__(self, slide_path):
        super().__init__(slide_path)
        self.slide_path = slide_path  # Store path for later reconstruction
    
    def __reduce__(self):
        # Define how to pickle the object
        return (self.__class__, (self.slide_path,))


def open_slide(filename):
    """Open a whole-slide or regular image.

    Return an OpenSlide object for whole-slide images and an ImageSlide
    object for other types of images."""
    try:
        return OpenSlideWrapper(filename)
    except openslide.OpenSlideUnsupportedFormatError:
        return ImageSlideWrapper(filename)

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

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

    @property
    def to_array(self):
        return [self.x, self.y]

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)


class Triangle():

    def __init__(self, p1:Point, p2:Point, p3:Point) -> None:

        self.pt1 = p1
        self.pt2 = p2
        self.pt3 = p3
        
    def draw(self, ax, c='k', lw=1, **kwargs):

        pts = np.array([[self.pt1.x, self.pt1.y], [self.pt2.x, self.pt2.y], [self.pt3.x, self.pt3.y]])
        p = Polygon(pts, closed=False, fc=[1,0,0, 0.05], ec=(0,0,0,1), lw=lw)
        #ax = plt.gca()
        ax.add_patch(p)        

    @property
    def area(self) -> float:

        # calc the lenght of each side
        a = self.pt1.distance_to(self.pt2)
        b = self.pt1.distance_to(self.pt3)
        c = self.pt2.distance_to(self.pt3)

        s = (a + b + c) / 2
        return (s*(s-a)*(s-b)*(s-c)) ** 0.5

    @property
    def points(self):
        return np.array([[self.pt1.x, self.pt1.y], [self.pt2.x, self.pt2.y], [self.pt3.x, self.pt3.y]])

    def _sign(self, p1:Point, p2:Point, p3:Point) -> bool:
    
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
    

    def contains(self, pt:Point) -> bool:
        # https://stackoverflow.com/a/2049593
        if type(pt) is not Point:
            pt = Point(*pt)

        d1 = self._sign(pt, self.pt1, self.pt2)
        d2 = self._sign(pt, self.pt2, self.pt3)
        d3 = self._sign(pt, self.pt3, self.pt1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not(has_neg and has_pos)

class TriangleRegistrationPair:

    def __init__(self, source:Triangle, target:Triangle, homography:bool=True) -> None:

        self.source = source
        self.target = target
        self.error = -1
        self.error_inv = -1

        _, _, _, self.tf_param = RegistrationQuadTree.estimate_homography(source.points, target.points, homography)
        self.tf_param_inv = tf.AffineTransformation(self.get_homography_inv[:2, :2], self.get_homography_inv[:2, 2:].reshape(-1))


    def contains(self, pt:Point):

        return self.source.contains(pt)

    def target_contains(self, pt:Point):

        return self.target.contains(pt)

    def get_reg_error(self, ptsA, ptsB):

        self.error = np.linalg.norm(self.tf_param.transform(ptsA)-ptsB, axis=1).mean() 
        return self.error

    def get_reg_error_inv(self, ptsA, ptsB):

        self.error_inv = np.linalg.norm(self.tf_param_inv.transform(ptsA)-ptsB, axis=1).mean()
        return self.error_inv

    @property
    def get_homography(self):

        H = np.identity(3)
        H[:2, :2] = self.tf_param.b
        H[:2, 2:] = self.tf_param.t.reshape(2,1)
        return H

    @property
    def get_homography_inv(self):

        H = self.get_homography
        if (cv2.determinant(H) != 0.0):
            return inv(H)
        else:
            return np.array([[1., 0., -H[0,-1]],
                          [0., 1., -H[1,-1]],
                          [0., 0., 0.]])


    @property
    def get_dict_representation(self):

        H = self.get_homography
        H_inv = self.get_homography_inv
        
        representation = {
            "t_00":H[0,0], "t_01":H[0,1], "t_02":H[0,2],
            "t_10":H[1,0], "t_11":H[1,1], "t_12":H[1,2],
            "t_20":H[2,0], "t_21":H[2,1], "t_22":H[2,2],

            "t_00_inv":H_inv[0,0], "t_01_inv":H_inv[0,1], "t_02_inv":H_inv[0,2],
            "t_10_inv":H_inv[1,0], "t_11_inv":H_inv[1,1], "t_12_inv":H_inv[1,2],
            "t_20_inv":H_inv[2,0], "t_21_inv":H_inv[2,1], "t_22_inv":H_inv[2,2],

            "source": {
                "p1": {"x": int(self.source.pt1.x), "y": int(self.source.pt1.y)},
                "p2": {"x": int(self.source.pt2.x), "y": int(self.source.pt2.y)},
                "p3": {"x": int(self.source.pt3.x), "y": int(self.source.pt3.y)},
            },

            "target": {
                "p1": {"x": int(self.target.pt1.x), "y": int(self.target.pt1.y)},
                "p2": {"x": int(self.target.pt2.x), "y": int(self.target.pt2.y)},
                "p3": {"x": int(self.target.pt3.x), "y": int(self.target.pt3.y)},
            },
        }


        return representation


    def __getstate__(self):

        attributes = self.__dict__.copy()

        attributes["homography"] = self.get_homography
        attributes["homography_inv"] = self.get_homography_inv

        del attributes['tf_param']
        del attributes['tf_param_inv']
        return attributes

    def __setstate__(self, state):

        self.__dict__ = state

        self.tf_param = tf.AffineTransformation(self.__dict__["homography"][:2, :2], self.__dict__["homography"][:2, 2:].reshape(-1))
        self.tf_param_inv = tf.AffineTransformation(self.__dict__["homography_inv"][:2, :2], self.__dict__["homography_inv"][:2, 2:].reshape(-1))



class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx:float, cy:float, w:float, h:float):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w/2, cx + w/2
        self.north_edge, self.south_edge = cy - h/2, cy + h/2


    @property
    def rect_repr(self):
        return (int(self.west_edge), int(self.north_edge), int(self.w), int(self.h))
        #return (self.p1[0], self.p1[1], self.w, self.h)

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)

    def create(cls, x:float, y:float, w:float, h:float):

        cx, cy = x + w // 2, y + h // 2

        return cls(cx, cy, w, h)

    def is_valid(self):
        return self.w > 0 and self.h > 0


    def contains(self, point:Point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""

        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (point_x >= self.west_edge and
                point_x <=  self.east_edge and
                point_y >= self.north_edge and
                point_y <= self.south_edge)


    def contains_traingle(self, traingle:Triangle):

        return self.contains(traingle.pt1) and self.contains(traingle.pt2) and self.contains(traingle.pt3)


    def intersects(self, other):
        """Does Rect object other interesect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def get_line_intersection(self, p1:Point, p2:Point):
        """ 
        Returns the point of intersection of the lines passing through a2,a1 and the rectangle.
        p1: [x, y] a point on the first line
        p2: [x, y] another point on the first line
        """

        wn_ws = (Point(x=self.west_edge, y=self.north_edge), Point(x=self.west_edge, y=self.south_edge))
        es_ws = (Point(x=self.east_edge, y=self.south_edge), Point(x=self.west_edge, y=self.south_edge))

        en_wn = (Point(x=self.east_edge, y=self.north_edge), Point(x=self.west_edge, y=self.north_edge))
        en_ws = (Point(x=self.east_edge, y=self.north_edge), Point(x=self.east_edge, y=self.south_edge))


        intersection_points = []
        for rp1, rp2 in [wn_ws, es_ws, en_wn, en_ws]:
            s = np.vstack([p1.to_array, p2.to_array, rp1.to_array,rp2.to_array])        # s for stacked
            h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
            l1 = np.cross(h[0], h[1])           # get first line
            l2 = np.cross(h[2], h[3])           # get second line
            x, y, z = np.cross(l1, l2)          # point of intersection
            if z != 0: # lines are not parallel
                p = Point(x=x/z, y=y/z)
                if self.contains(p):# check that the point is a the border of the rectangle
                    # check if the point is between the two points p1 and p2
                    if Rect.create(Rect, x=min(p1.x, p1.x), y=min(p1.y, p1.y), w=abs(p1.x-p2.x), h=abs(p1.y-p2.y)).contains(p):
                        intersection_points.append((p, p1, p2))

        return intersection_points


    def draw(self, ax, c='k', lw=1, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)



class RotatedRect:
    """A rectangle based on the four corner points."""

    def __init__(self, p1:tuple, p2:tuple, p3:tuple, p4:tuple):


        self.p1, self.p2, self.p3, self.p4 = p1, p2, p3, p4
        self.point_array = np.array([p1, p2, p3, p4])
        self.rect = cv2.minAreaRect(self.point_array.astype(int))
        self.box = cv2.boxPoints(self.rect)

        self.cx, self.cy = self.rect[0]
        self.h_min, self.w_min = self.rect[1]
        self.angle = self.rect[2]
        
        self.west_edge = min(self.point_array[:, 0]).astype(int)
        self.east_edge = max(self.point_array[:, 0]).astype(int)
        self.north_edge= min(self.point_array[:, 1]).astype(int)
        self.south_edge= max(self.point_array[:, 1]).astype(int)

        self.h, self.w = self.south_edge - self.north_edge, self.east_edge - self.west_edge, 


    def is_valid(self):
        return self.w > 0 and self.h > 0

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)


class RegistrationQuadTree:
    """[summary]

    A class implementing the registration quadtree.

    """

    def __init__(self, source_slide_path:Path=None, target_slide_path:Path=None, source_boundary:Rect=None, target_boundary:Rect=None,
                        depth=0, target_depth=1, thumbnail_size=(2048, 2048),
                        run_async=False, node_orientation:NodeOrientation = NodeOrientation.TOP,
                        parent=None, homography:bool=True, filter_outliner:bool=False, num_workers:int=2, initial_rotation=None, source_slide:openslide.OpenSlide=None,target_slide:openslide.OpenSlide=None, **kwargs):
        """[summary]
        Init the current quadtree level

        Args:
            source_slide_path ([Path]): [source slide path as Path object]
            target_slide_path ([Path]): [target slide path as Path object]
            source_slide ([openslide.OpenSlide]): [source slide path as OpenSlide object]
            target_slide ([openslide.OpenSlide]): [target slide path as OpenSlide object]
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
            num_workers (int, optional): [current depth]. Defaults to 2.
        """


        kwargs.update({"run_async":run_async, "thumbnail_size":thumbnail_size, "homography":homography, 
                        "filter_outliner":filter_outliner, "target_depth":target_depth, "num_workers": num_workers, 
                        "initial_rotation": initial_rotation})
        
        self.kwargs = kwargs
        self.parent = parent
        self.node_orientation = node_orientation
        self.run_async = run_async
        self.thumbnail_size = thumbnail_size
        self.depth = depth
        self.target_depth = target_depth 
        self.num_workers = num_workers
        self.initial_rotation = initial_rotation

        if (source_slide is None) and (source_slide_path is None):
            raise ValueError('Either source_slide or source_slide_path need to be set.')
        if not source_slide_path is None and not isinstance(source_slide_path,str) and not isinstance(source_slide_path, pathlib.PurePath):
            raise ValueError('source_slide_path needs to be either str or Path type.')
        if (source_slide is None):
            self.source_slide_path = Path(source_slide_path) if isinstance(source_slide_path, str) else source_slide_path

        if (target_slide is None) and (target_slide_path is None):
            raise ValueError('Either source_slide or source_slide_path need to be set.')
        if not target_slide_path is None and not isinstance(target_slide_path,str) and not isinstance(target_slide_path, pathlib.PurePath):
            raise ValueError('target_slide_path needs to be either str or Path type.')
        if (target_slide is None):            
            self.target_slide_path = Path(target_slide_path) if isinstance(target_slide_path, str) else target_slide_path

        self._target_slide_dimensions = None
        self._source_slide_dimensions = None 

        self._mpp_x_scale = 1
        self._mpp_y_scale = 1

        self.source_boundary = source_boundary
        self.target_boundary = target_boundary 
        
        if source_slide is None:
            source_slide = open_slide(str(source_slide_path))
        self.source_slide = source_slide

        if self.source_boundary == None: 
            self._source_slide_dimensions = source_slide.dimensions
            self.source_boundary = Rect.create(Rect, 0, 0, source_slide.dimensions[0], source_slide.dimensions[1])

        if target_slide is None:
            target_slide = open_slide(str(target_slide_path))    
        self.target_slide = target_slide
        if self.target_boundary == None: 
            self._target_slide_dimensions = target_slide.dimensions
            self.target_boundary = Rect.create(Rect, 0, 0, target_slide.dimensions[0], target_slide.dimensions[1])

        # start timer
        tic = time.perf_counter()

        if self.run_async: #self.run_async
            with concurrent.futures.ThreadPoolExecutor() as executor: # ProcessPoolExecutor  ThreadPoolExecutor
                source_func = functools.partial(RegistrationQuadTree.get_region_thumbnail, slide=self.source_slide, boundary=self.source_boundary, depth=self.depth + 1, size=self.thumbnail_size)
                target_func = functools.partial(RegistrationQuadTree.get_region_thumbnail, slide=self.target_slide, boundary=self.target_boundary, depth=self.depth + 1, size=self.thumbnail_size)

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
            self.source_thumbnail, self.source_scale = RegistrationQuadTree.get_region_thumbnail(slide=self.source_slide, boundary=self.source_boundary, depth=self.depth + 1, size=self.thumbnail_size)
            self.target_thumbnail, self.target_scale = RegistrationQuadTree.get_region_thumbnail(slide=self.target_slide, boundary=self.target_boundary, depth=self.depth + 1, size=self.thumbnail_size)

        # if the initial rotation angle is to big rotate the image to make the matching easier 
        inv_rotation_matrix = None
        if self.initial_rotation is not None and int(self.initial_rotation) not in range(-12, 12): #tolerance of 5°            
            inv_rotation_matrix = self.get_inv_rotation_matrix_from_angle(-self.initial_rotation, width=self.target_thumbnail.width, height=self.target_thumbnail.height)
            # rotate image 
            self.target_thumbnail = self.target_thumbnail.rotate(self.initial_rotation, fillcolor=(255, 0, 0), expand=True)
            
        self.ptsA, self.ptsB, self.matchedVis, self.mean_reg_error, self.sigma2, self.q, self.tf_param, self.mean_reg_error = self.perform_registration(self.source_thumbnail, self.target_thumbnail, source_scale=self.source_scale, target_scale=self.target_scale, inv_rotation_matrix=inv_rotation_matrix, **kwargs)
        
        # invert rotation to get scale
        M = self.get_homography@self.get_inv_rotation_matrix
        self.mpp_x_scale, self.mpp_y_scale = M[0][0], M[1][1]

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

    def perform_registration(self, source_thumbnail, target_thumbnail, source_scale, target_scale, filter_outliner, homography, inv_rotation_matrix=None, **kwargs):

        ptsA, ptsB, matchedVis = self.extract_matching_points(source_thumbnail, target_thumbnail, source_scale=source_scale, target_scale=target_scale, inv_rotation_matrix=inv_rotation_matrix, **kwargs)

        if filter_outliner:
            ptsA, ptsB, scale_factors = self.filter_outliner(ptsA, ptsB, homography)
        
        mean_reg_error, sigma2, q, tf_param = self.estimate_homography(ptsA, ptsB)

        return ptsA, ptsB, matchedVis, mean_reg_error, sigma2, q, tf_param, mean_reg_error

    @staticmethod
    def estimate_homography(ptsA, ptsB, homography:bool=True):

        if homography:
            mean_reg_error, sigma2, q, tf_param = RegistrationQuadTree._get_min_reg_error_hom(ptsA, ptsB)
        else:
            tf_param, sigma2, q = cpd.registration_cpd(ptsA, ptsB, 'affine')
            mean_reg_error = np.linalg.norm(tf_param.transform(ptsA)-ptsB, axis=1).mean()

            mean_reg_error_temp, _, _, tf_param_temp = RegistrationQuadTree._get_min_reg_error_hom(ptsA, ptsB)

            if mean_reg_error > mean_reg_error_temp:
                mean_reg_error = mean_reg_error_temp
                tf_param = tf_param_temp

        return mean_reg_error, sigma2, q, tf_param

    @staticmethod
    def _get_min_reg_error_hom(ptsA, ptsB):
        mean_reg_error = 99999999

        affine = np.array([[1,0,0], [0,1,0], [0,0,1]])
        tf_param = tf.AffineTransformation(affine[:2, :2], affine[:2, 2:].reshape(-1))
        sigma2, q = -1, -1
        if len(ptsA) >= 3:
            for outline_filter in [0, cv2.RANSAC, cv2.RHO, cv2.LMEDS]:
                affine, mask = cv2.estimateAffine2D(ptsA, ptsB, outline_filter)

                if affine is not None:
                    temp_tf_param = tf.AffineTransformation(affine[:2, :2], affine[:2, 2:].reshape(-1))

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
            self.source_slide_dimensions = self.source_slide.dimensions
        
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
            self._target_slide_dimensions = self.target_slide.dimensions
        
        return self._target_slide_dimensions

    @target_slide_dimensions.setter 
    def target_slide_dimensions(self, dimensions):

        self._target_slide_dimensions = dimensions

        if self.nw is not None: self.nw.target_slide_dimensions = dimensions
        if self.ne is not None: self.ne.target_slide_dimensions = dimensions
        if self.se is not None: self.se.target_slide_dimensions = dimensions
        if self.sw is not None: self.sw.target_slide_dimensions = dimensions

    @property
    def source_path(self):
        if hasattr(self.source_slide, '_filename'):
            return Path(self.source_slide._filename)
        elif hasattr(self.source_slide, '_image'):
            return Path(self.source_slide._image.filename)
        else:
            raise NotImplementedError('Missing _filename property in OpenSlide object source.')
            
    @property
    def target_path(self):
        if hasattr(self.target_slide, '_filename'):
            return Path(self.target_slide._filename)
        elif hasattr(self.target_slide, '_image'):
            return Path(self.target_slide._image.filename)
        else:
            raise NotImplementedError('Missing _filename property in OpenSlide object target.')

    @property
    def source_name(self):
        return self.source_path.stem
            
    @property
    def target_name(self):
        return  self.target_path.stem
        
    @property
    def mpp_x_scale(self): #get from rotated image
        return self._mpp_x_scale

    @mpp_x_scale.setter 
    def mpp_x_scale(self, scale):
        self._mpp_x_scale = min(scale, 5)

    @property
    def mpp_y_scale(self): #get from rotated image
        return self._mpp_y_scale

    @mpp_y_scale.setter 
    def mpp_y_scale(self, scale):
        self._mpp_y_scale = min(scale, 5)

    @property
    def get_homography(self):

        H = np.identity(3)
        H[:2, :2] = self.tf_param.b
        H[:2, 2:] = self.tf_param.t.reshape(2,1)
        return H

    @property
    def get_homography_inv(self):

        H = self.get_homography
        if (cv2.determinant(H) != 0.0):
            return inv(H)
        else:
            return np.array([[1., 0., -H[0,-1]],
                          [0., 1., -H[1,-1]],
                          [0., 0., 0.]])

    @property
    def get_rotation_angle(self):

        return - math.atan2(self.tf_param.b[0,1], self.tf_param.b[0,0]) * 180 / math.pi

    @property
    def get_rotation_matrix(self):

        phi = self.get_rotation_angle * math.pi / 180
        return np.array([[np.cos(phi), - np.sin(phi), 0],
                                        [np.sin(phi), np.cos(phi), 0], 
                                        [0., 0, 1]])

    @property
    def get_inv_rotation_matrix(self):

        M = self.get_rotation_matrix
        return inv(M)

    @staticmethod
    def get_rotation_matrix_from_angle(angle:float, width:int, height:int):

        (cX, cY) = (width // 2, height // 2)
        M = np.identity(3)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((height * sin) + (width * cos))
        nH = int((height * cos) + (width * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        return M

    @staticmethod
    def get_inv_rotation_matrix_from_angle(angle:float, width:int, height:int):

        M = RegistrationQuadTree.get_rotation_matrix_from_angle(angle, width=width, height=height)

        H = np.identity(3)
        H[:2, :2] = M[:2, :2]
        H[:2, 2:] = M[:2, 2:]
        H = inv(H)

        return H[:2]

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
        # p1    p1h     p2 
        #    nw     ne
        # p4h   pc     p2h
        #    sw     se
        # p4    p3h    p3
        west_edge, east_edge   = self.source_boundary.west_edge, self.source_boundary.east_edge
        north_edge, south_edge = self.source_boundary.north_edge, self.source_boundary.south_edge

        p1  = self.transform_boxes([(west_edge, north_edge, 50, 50)])[0][:2] # top left
        p1h = self.transform_boxes([(west_edge + (east_edge-west_edge) / 2, north_edge, 50, 50)])[0][:2] # center top

        p2 =  self.transform_boxes([(east_edge, north_edge, 50, 50)])[0][:2] # top right
        p2h = self.transform_boxes([(east_edge, north_edge + (south_edge-north_edge) / 2, 50, 50)])[0][:2] # center right

        p3 =  self.transform_boxes([(east_edge, south_edge, 50, 50)])[0][:2] # bottom right
        p3h = self.transform_boxes([(west_edge + (east_edge-west_edge) / 2, south_edge, 50, 50)])[0][:2] # center bottom

        p4 =  self.transform_boxes([(west_edge, south_edge, 50, 50)])[0][:2] # bottom left
        p4h = self.transform_boxes([(west_edge, north_edge + (south_edge-north_edge) / 2, 50, 50)])[0][:2] # center left

        pc = self.transform_boxes([(west_edge + (east_edge-west_edge) / 2, north_edge + (south_edge-north_edge) / 2, 50, 50)])[0][:2] # center
        
        # set new box coordinates withhin old limits
        p1, p2, p3, p4 = [(max(p[0], 0), max(p[1], 0)) for p in [p1, p2, p3, p4]]
        p1, p2, p3, p4 = [(min(p[0], self.target_slide_dimensions[0]), min(p[1], self.target_slide_dimensions[1])) for p in [p1, p2, p3, p4]]

        targetRotatedRect = RotatedRect(p1, p2, p3, p4)

        # create target boxes
        target_nw = RotatedRect(p1=p1,  p2=p1h, p3=pc,  p4=p4h)
        target_sw = RotatedRect(p1=p4h, p2=pc,  p3=p3h, p4=p4)
        target_ne = RotatedRect(p1=p1h, p2=p2,  p3=p2h, p4=pc)
        target_se = RotatedRect(p1=pc,  p2=p2h, p3=p3,  p4=p3h)

        # create source boxes
        source_nw = Rect(source_cx - source_w/2, source_cy - source_h/2, source_w, source_h)
        source_sw = Rect(source_cx - source_w/2, source_cy + source_h/2, source_w, source_h)

        source_ne = Rect(source_cx + source_w/2, source_cy - source_h/2, source_w, source_h)
        source_se = Rect(source_cx + source_w/2, source_cy + source_h/2, source_w, source_h)


        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.

        qt_functions = {}
        
        if source_nw.is_valid() and target_nw.is_valid():
            qt_functions[NodeOrientation.NORTH_WEST] = functools.partial(self.__class__, source_slide=self.source_slide, target_slide=self.target_slide, source_boundary=source_nw, target_boundary=target_nw,  depth=self.depth + 1, node_orientation=NodeOrientation.NORTH_WEST, parent=self, **self.kwargs)

        if source_ne.is_valid() and target_ne.is_valid():
            qt_functions[NodeOrientation.NORTH_EAST] = functools.partial(self.__class__, source_slide=self.source_slide, target_slide=self.target_slide, source_boundary=source_ne, target_boundary=target_ne,  depth=self.depth + 1, node_orientation=NodeOrientation.NORTH_EAST, parent=self, **self.kwargs)

        if source_se.is_valid() and target_se.is_valid():
            qt_functions[NodeOrientation.SOUTH_EAST] = functools.partial(self.__class__, source_slide=self.source_slide, target_slide=self.target_slide, source_boundary=source_se, target_boundary=target_se,  depth=self.depth + 1, node_orientation=NodeOrientation.SOUTH_EAST, parent=self, **self.kwargs)

        if source_sw.is_valid() and target_sw.is_valid():
            qt_functions[NodeOrientation.SOUTH_WEST] = functools.partial(self.__class__, source_slide=self.source_slide, target_slide=self.target_slide, source_boundary=source_sw, target_boundary=target_sw,  depth=self.depth + 1, node_orientation=NodeOrientation.SOUTH_WEST, parent=self, **self.kwargs)

        if self.run_async == True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor: # ProcessPoolExecutor  ThreadPoolExecutor

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
                    print( f'generated an exception: {exc}')

    def draw_feature_points(self, num_sub_pic:int=5, figsize=(16, 16), patch_size:int=512):
        
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        fig.suptitle(f'{self.source_name} --> {self.target_name}')
        gs = fig.add_gridspec(5, num_sub_pic)

        f_ax_match = fig.add_subplot(gs[:2, :3])
        f_ax_match.imshow(self.matchedVis)

        source_slide = self.source_slide
        target_slide = self.target_slide

        tf_temp = tf.AffineTransformation(self.tf_param.b, self.tf_param.t)

        # quiver plot
        try:
            ptA_quiver = self.scale_points([(pt + (self.source_boundary.west_edge, self.source_boundary.north_edge)).astype(int) for pt in np.copy(self.ptsA)], self.source_scale, operator.truediv)
            ptB_quiver = self.scale_points([(pt + (self.target_boundary.west_edge, self.target_boundary.north_edge)).astype(int) for pt in np.copy(self.ptsB)], self.target_scale, operator.truediv)

            u_max = max(abs(ptA_quiver[:, 0] - ptB_quiver[:, 0] + 0.0001))
            v_max = max(abs(ptA_quiver[:, 1] - ptB_quiver[:, 1] + 0.0001))

            x, y = ptA_quiver[:, 0], ptA_quiver[:, 1]
            u, v = (ptA_quiver[:, 0] - ptB_quiver[:, 0] + 0.0001) / u_max, (ptA_quiver[:, 1] - ptB_quiver[:, 1] + 0.0001) / v_max

            quiver_plot = fig.add_subplot(gs[:2, 3])
            quiver_plot.imshow(self.source_thumbnail)
            quiver_plot.quiver(x, y, u, v) # , color=distance


            xx = np.linspace(0, self.source_thumbnail.width, int(self.source_thumbnail.width / 20)) 
            yy = np.linspace(0, self.source_thumbnail.height, int(self.source_thumbnail.height / 20)) 
            xx, yy = np.meshgrid(xx, yy)


            points = np.transpose(np.vstack((x, y)))
            u_interp = interpolate.griddata(points, u, (xx, yy), method='cubic')
            v_interp = interpolate.griddata(points, v, (xx, yy), method='cubic')

            interpolated_plot = fig.add_subplot(gs[:2, 4])
            interpolated_plot.imshow(self.source_thumbnail)
            interpolated_plot.quiver(xx, yy, u_interp, v_interp) # , color=distance
        except Exception as inst:
            print(inst)

        for idx, (pA, pB) in enumerate(zip(self.ptsA[:num_sub_pic].copy(), self.ptsB[:num_sub_pic].copy())):
            size = patch_size

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

            # rotate points
            pB_location = pB.astype(int) - (size_target_x // 2, size_target_y // 2)
            if size_target_x > 0 and size_target_y > 0:                
                image_target = target_slide.read_region(location=pB_location, level=0, size=(size, size))
                image_target = image_target.rotate(self.get_rotation_angle, fillcolor=(255, 255, 255), expand=True)

            trans_location = transformed - (size_target_x // 2, size_target_y // 2)
            if size_target_x > 0 and size_target_y > 0:
                image_target_trans = target_slide.read_region(location=trans_location, level=0, size=(size_target_x, size_target_y))
                image_target_trans = image_target_trans.rotate(self.get_rotation_angle, fillcolor=(255, 255, 255), expand=True)

            ax_0 = fig.add_subplot(gs[2, idx])
            ax_0.set_title(f'Source:  {pA}')
            ax_0.imshow(image_source)

            ax_1 = fig.add_subplot(gs[3, idx])
            ax_1.set_title(f'Trans:  {transformed}')
            ax_1.imshow(image_target_trans)

            ax_2 = fig.add_subplot(gs[4, idx])
            ax_2.set_title(f'GT:  {pB}')
            ax_2.imshow(image_target)
        

        if self.divided:
            figures = [fig]
            if self.nw is not None: figures.append(self.nw.draw_feature_points(num_sub_pic, figsize)[0])
            if self.ne is not None: figures.append(self.ne.draw_feature_points(num_sub_pic, figsize)[0])
            if self.se is not None: figures.append(self.se.draw_feature_points(num_sub_pic, figsize)[0])
            if self.sw is not None: figures.append(self.sw.draw_feature_points(num_sub_pic, figsize)[0])

            return figures
        else:
            return [fig]


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

        source_slide = self.source_slide
        target_slide = self.target_slide
        
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
                    inv_rotation_matrix=None,
                    **kwargs
                    ):
        kwargs.update({"debug":debug, "point_extractor":point_extractor, "use_gray":use_gray, "inv_rotation_matrix":inv_rotation_matrix})

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
            
        # rotate points back to the original position
        if inv_rotation_matrix is not None:
            tf_angle_target_inv = tf.AffineTransformation(inv_rotation_matrix[:2, :2], inv_rotation_matrix[:2, 2:].reshape(-1))
            kpsB = tf_angle_target_inv.transform(kpsB)

        ptsA, ptsB = self.scale_points(kpsA, source_scale, operator.mul), self.scale_points(kpsB, target_scale, operator.mul)
        return ptsA, ptsB, matchedVis

    def scale_points(self, kps, scale, op=operator.mul):

        pts = []

        for pt in np.copy(kps):
            # scale points 
            for s_scale in scale:
                pt = op(pt, s_scale) 

            pts.append(pt)

        return np.array(pts)

    @staticmethod
    def get_region_thumbnail(slide:openslide.OpenSlide, boundary:Rect, depth:int=0, size=(2048, 2048)):

        scale = []

        downsample = max(*[dim / thumb for dim, thumb in zip((boundary.w, boundary.h), (size[0] * depth, size[1] * depth))])        
        level = slide.get_best_level_for_downsample(downsample)

        downsample = slide.level_downsamples[level]

        x, y, w, h = int(boundary.west_edge), int(boundary.north_edge), int(boundary.w / downsample), int(boundary.h / downsample)
        scale.append(np.array((boundary.w, boundary.h)) / (w, h))

        tile = slide.read_region((x, y), level, (w, h))

        thumb = Image.new('RGB', tile.size, '#ffffff')
        thumb.paste(tile, None, tile)
        thumb.thumbnail(size, Image.LANCZOS)
        scale.append(np.array([w, h]) / thumb.size)


        return thumb, scale

    @property
    def get_dict_representation(self):

        H = self.get_homography
        H_inv = self.get_homography_inv

        representation = {
            "t_00":H[0,0], "t_01":H[0,1], "t_02":H[0,2],
            "t_10":H[1,0], "t_11":H[1,1], "t_12":H[1,2],
            "t_20":H[2,0], "t_21":H[2,1], "t_22":H[2,2],

            "t_00_inv":H_inv[0,0], "t_01_inv":H_inv[0,1], "t_02_inv":H_inv[0,2],
            "t_10_inv":H_inv[1,0], "t_11_inv":H_inv[1,1], "t_12_inv":H_inv[1,2],
            "t_20_inv":H_inv[2,0], "t_21_inv":H_inv[2,1], "t_22_inv":H_inv[2,2],

            "x_min": self.source_boundary.west_edge,
            "y_min": self.source_boundary.north_edge,
            "x_max": self.source_boundary.east_edge,
            "y_max": self.source_boundary.south_edge,

            "ne": self.ne.get_dict_representation if self.ne is not None else None ,
            "se": self.se.get_dict_representation if self.se is not None else None ,
            "nw": self.nw.get_dict_representation if self.nw is not None else None ,
            "sw": self.sw.get_dict_representation if self.sw is not None else None ,
        }


        return representation


    def __getstate__(self):

        attributes = self.__dict__.copy()

        attributes["homography"] = self.get_homography
        attributes["source_slide"] = self.source_slide
        attributes["target_slide"] = self.target_slide

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

        self.source_slide = self.__dict__["source_slide"]
        self.target_slide = self.__dict__["target_slide"]

        self.tf_param = tf.AffineTransformation(self.__dict__["homography"][:2, :2], self.__dict__["homography"][:2, 2:].reshape(-1))


