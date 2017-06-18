#!/usr/bin/env python2
#-*- encoding: utf8 -*-

'''
Face Recognizer

Author: Minsu Jang (minsu@etri.re.kr)
'''

import uuid
import time
import logging
import openface
import cv2
import numpy as np
import threading
from magma_util import list_files
from magma_util import softmax
from magma_util import measure_blur

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

#: Landmark indices.
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]


class FaceAligner(object):
    '''
    Align face images
    '''

    def __init__(self, img_dim=96, landmark_indices=INNER_EYES_AND_BOTTOM_LIP):
        self.landmark_indices = landmark_indices
        self.img_dim = img_dim


    def align(self, img, landmarks):
        '''
        Arguments
        ---------
        img: an RGB image
        landmarks: a list of (x,y) points aligned to a face in the input image
        '''
        assert img is not None
        assert landmarks is not None

        np_landmarks = np.float32(landmarks)
        np_landmark_indices = np.array(self.landmark_indices)

        transform_h = cv2.getAffineTransform(np_landmarks[np_landmark_indices],
                                             self.img_dim * MINMAX_TEMPLATE[np_landmark_indices])
        thumbnail = cv2.warpAffine(img, transform_h, (self.img_dim, self.img_dim))

        return thumbnail

class PeriodicExecutor(threading.Thread):
    def __init__(self, sleep, func, params):
        """ execute func(params) every 'sleep' seconds """
        self.func = func
        self.params = params
        self.sleep = sleep
        threading.Thread.__init__(self, name="PeriodicExecutor")
        self.setDaemon(1)

    def run(self):
        while 1:
            time.sleep(self.sleep)
            apply(self.func, self.params)


class FaceRecognizer(object):
    '''
    Recognize a face
    '''

    def __init__(self, model_path, registry_path, img_dim=96, min_blur_measure=120):
        '''
        Arguments
        ---------
        model_path: a neural network weights for retrieving face representations
        registry_path: a path to the folder where registered face images are saved
        img_dim: dimention of the aligned face image
        '''
        self.model_path = model_path
        self.registry_path = registry_path
        self.img_dim = img_dim
        self.face_aligner = FaceAligner()
        self.registered_face_imgs = {}
        self.representations = {}

        self.min_blur_measure = min_blur_measure

        np.set_printoptions(precision=2)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.net = openface.TorchNeuralNet(self.model_path, self.img_dim)
        self.load_registered_faces(self.registry_path)

        #threading.Timer(5, self.load_registered_faces).start()

    def register_face(self, img, landmarks, name):
        '''
        Register a face.

        Arguments
        ---------
        img: an input image
        landmarks: a list of (x,y) points representing facial landmarks
        '''
        aligned_img = self.face_aligner.align(img, landmarks)
        cv2.imwrite(self.registry_path + '/' + str(uuid.uuid1()) + '_' + name + '.jpg', aligned_img)


    def load_registered_faces(self, img_path=None):
        '''
        Load registered images.
        '''
        self.registered_face_imgs = {}
        self.representations = {}
        if img_path is None:
            img_path = self.registry_path
        img_file_paths = list_files(img_path)
        for path in img_file_paths:
            self.logger.info("Loading a face image: {}".format(path))
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            name = path.split('/')[-1].split('_')[1].split('.')[0]
            print("FACE_RECOGNIZER: Loading a face image for %s - %s"%(name, path))
            self.logger.info("FACE_RECOGNIZER: Loading a face image for %s - %s", name, path)
            if name not in self.registered_face_imgs:
                self.registered_face_imgs[name] = []
                self.representations[name] = []
            self.registered_face_imgs[name].append(img)
            self.representations[name].append(self.get_face_rep(img))


    def recognize(self, img, landmarks):
        '''
        Recognize a face image.
        '''
        matching_scores = {}
        image_to_save = None
        aligned_img = self.face_aligner.align(img, landmarks)
        #t = time.time()
        rep = self.get_face_rep(aligned_img)
        for name in self.representations:
            representations = self.representations[name]
            for representation in representations:
                if name not in matching_scores:
                    matching_scores[name] = []
                distance = rep - representation
                matching_score = np.dot(distance, distance)
                matching_scores[name].append(matching_score)
                self.logger.debug("Comparing with %s: distance=%3.2f", name, matching_score)
        self.logger.debug("Face Matching Scores: %s", matching_scores)
        for key in matching_scores:
            matching_scores[key] = np.average(np.array(matching_scores[key]))
        #print "DISTANCE: ", matching_scores
        if min(matching_scores.values()) >= 0.6:
            matching_scores = None
        else:
            values = matching_scores.values()
            values1 = np.sum(values) / np.array(values)
            softmax_values = softmax(values1)
            for key in matching_scores:
                matching_scores[key] = softmax_values[values.index(matching_scores[key])]
        if measure_blur(aligned_img) > self.min_blur_measure:
            image_to_save = aligned_img
        return matching_scores, image_to_save


    def recognize1(self, img, landmarks):
        '''
        Recognize a face image.
        '''
        matching_scores = {}
        aligned_img = self.face_aligner.align(img, landmarks)
        rep = self.get_face_rep(aligned_img)
        for name_id in self.representations:
            name = name_id.split('_')[-1]
            if name not in matching_scores:
                matching_scores[name] = 100000
            distance = rep - self.representations[name_id]
            matching_score = np.dot(distance, distance)
            if matching_score < matching_scores[name]:
                matching_scores[name] = matching_score
            self.logger.debug("Comparing with {}: distance={:0.3f}".format(name_id, matching_score))
        print "Face Recognized: ", matching_scores
        values = matching_scores.values()
        values1 = np.sum(values) / np.array(values)
        softmax_values = softmax(values1)
        for key in matching_scores:
            matching_scores[key] = softmax_values[values.index(matching_scores[key])]
        return matching_scores


    def get_face_rep(self, img):
        '''
        Get a metric representation of a face image.
        '''
        start = time.time()
        rep = self.net.forward(img)
        self.logger.debug("OpenFace forward pass took {} seconds.".format(time.time() - start))
        return rep
