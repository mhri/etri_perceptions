# -*- coding: utf-8 -*-

"""This module defines a set of utility functions.

Author: Minsu Jang
"""
import os
from abc import ABCMeta, abstractmethod
from threading import Thread
from Queue import Queue
import numpy as np
import dlib
import cv2

class ImageWriter(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.img_queue = Queue()
        self.stop_flag = False

    def put(self, img, file_name):
        self.img_queue.put((img, file_name))

    def run(self):
        while True:
            img, file_name = self.img_queue.get()
            self.img_queue.task_done()
            cv2.imwrite(file_name, img)

            if self.stop_flag == True:
                break

    def stop(self):
        self.stop_flag = True


def list_files(root_dir, suffix=None):
    """
    Walks a directory recursively and returns a list of files.
    """
    files = []
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        for fname in file_list:
            if fname[0] == '.':
                continue
            if suffix != None and not fname.endswith(suffix):
                continue
            file_path = dir_name + "/" + fname
            files.append(file_path)
    return files


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def measure_blur(img):
    '''
    Measure the amount of blur in the input image. 
    '''
    return cv2.Laplacian(img, cv2.CV_64F).var()


def get_key_with_maxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

def get_key_with_minval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the min value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(min(v))]

class FaceDetectionListener(object):
    '''
    A listener interface for face detection.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def on_face_detection(self, img, dets):
        '''
        Notify an event of face detection.
        '''
        pass

class RegionOfInterest(object):
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

class FaceDetection(object):
    __metaclass__ = ABCMeta

    def __init__(self, listener=None, image_resize_ratio=1.0):
        self.listeners = []
        if listener != None:
            self.listeners.append(listener)
        self.resize_factor_ = 0.5
        self.inverse_resize_factor_ = 1.0 / self.resize_factor_

    @abstractmethod 
    def detect(self, img):
        pass

    def resize(self, img):
        return cv2.resize(img, (0, 0), fx=self.resize_factor_, fy=self.resize_factor_)

    def add_listener(self, listener):
        self.listeners.append(listener)
        
    def process(self, img):
        '''
        Process image.
        '''
        resized_img = self.resize(img)
        rois = self.detect(resized_img)
        if len(rois) > 0:
            if self.resize_factor_ != 1.0:
                for roi in rois:
                    roi.left = roi.left * self.inverse_resize_factor_
                    roi.top = roi.top * self.inverse_resize_factor_
                    roi.right = roi.right * self.inverse_resize_factor_
                    roi.bottom = roi.bottom * self.inverse_resize_factor_

            for listener in self.listeners:
                listener.on_face_detection(img, rois)


class FaceDetectionDLIB(FaceDetection):
    '''
    Face Detector using DLIB
    '''
    def __init__(self, listener=None, image_resize_ratio=0.5):
        FaceDetection.__init__(self, listener, image_resize_ratio)
        self.face_detector = dlib.get_frontal_face_detector()

    def detect(self, img):
        dets = self.face_detector(img, 1)
        rois = []
        for k, d in enumerate(dets):
            rois.append(RegionOfInterest(d.left(), d.top(), d.right(), d.bottom()))
        return rois


class FaceAlignment(object):
    '''
    Perform face alignment to get 68 facial feature points.
    '''
    def __init__(self, predictor_path="/Volumes/KobaiaData/Developments/shri_social_perception/shape_predictor_68_face_landmarks.dat"):
        self.predictor_path = predictor_path
        self.predictor = dlib.shape_predictor(self.predictor_path)


    '''
    Detect face landmarks and return a list of 2D coordinates.
    '''
    def detect_landmarks(self, img, rect):
        '''
        Parameters
        ----------
        img: an input image
        '''
        rect = dlib.dlib.rectangle(int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        landmarks = self.process(img, rect)
        return list(map(lambda p: (p.x,p.y), landmarks.parts()))


    def process(self, img, face_rect):
        '''
        Extract facial feature points.

        Parameters
        ----------
        img: an input image with faces
        face_rect: a rectangular roi for a face
        '''
        shape = self.predictor(img, face_rect)
        return shape

    def render1(self, img, shape):
        '''
        Render all the facial feature points.
        '''
        self.draw_lines(img, shape, 0, 67)
        return img

    def draw_lines1(self, img, shape, start_index, end_index):
        '''
        Draw a line along the feature points denoted by shape.
        '''
        for i in range(start_index+1, end_index+1):
            cv2.line(img, (shape.part(i).x, shape.part(i).y), (shape.part(i-1).x, shape.part(i-1).y), (255, 255, 0))

    def render(self, img, landmarks):
        '''
        Render all the facial feature points.
        '''
        self.draw_lines(img, landmarks, 0, 67)
        return img

    def draw_lines(self, img, landmarks, start_index, end_index):
        '''
        Draw a line along the feature points denoted by shape.
        '''
        for i in range(start_index+1, end_index+1):
            cv2.line(img, (landmarks[i][0], landmarks[i][1]), (landmarks[i-1][0], landmarks[i-1][1]), (255, 255, 0))

