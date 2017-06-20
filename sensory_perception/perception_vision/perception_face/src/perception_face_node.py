#!/usr/bin/env python
#-*- encoding: utf8 -*-
'''
Face-based Perception Processing

Author: Minsu Jang (minsu@etri.re.kr)
'''

import threading, time
import json
import rospy
import cv2
import message_filters
import rospkg
import numpy as np
from Queue import Queue
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from FaceRecognizer import FaceRecognizer
from QueuedProcessing import Worker
from FaceRecognitionWorker import FaceRecognitionWorker
from SmileRecognitionWorker import SmileRecognitionWorker
from HairLengthDetectionWorker import HairLengthDetectionWorker
from ClothColorDetectionWorker import ClothColorDetectionWorker

from perception_msgs.msg import PersonPerceptArray
from perception_msgs.msg import PerceptionEvent

#from magma_util import get_key_with_minval
#from magma_util import get_key_with_maxval
from magma_util import FaceDetectionDLIB
from magma_util import FaceAlignment
from magma_util import RegionOfInterest


class PerceptionFace(object):
    '''
    A ROS Node for Facial Perception
    '''
    def __init__(self, min_face_size=96*96):
        self.bridge = CvBridge()

        self.min_face_size = min_face_size

        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('perception_face')

        self.last_recognition_time = time.time()

        self.data_queue = Queue()
        self.result_queue = Queue()

        # create a face detector and a face aligner
        self.face_detector = FaceDetectionDLIB()
        self.face_aligner = FaceAlignment(predictor_path=pkg_path + '/data/shape_predictor_68_face_landmarks.dat')

        self.model_path = pkg_path + '/data/nn4.small2.v1.t7'
        self.face_registry_path = pkg_path + '/data/face_registry'
        rospy.loginfo('perception_face_node - Model Path: {}'.format(self.model_path))
        rospy.loginfo('perception_face_node - Faces Path: {}'.format(self.face_registry_path))

        self.name_info_file = pkg_path + '/data/name_info.json'
        rospy.loginfo('perception_face_node - Names File: {}'.format(self.name_info_file))
        self.names = json.load(open(self.name_info_file))

        self.worker = FaceRecognitionWorker(self.model_path, self.face_registry_path,
                                            self.data_queue, self.result_queue)
        self.worker.start()

        # A Worker for Smile Recognition
        self.emo_job_queue = Queue()
        self.emo_result_queue = Queue()
        smile_recognition_weights = pkg_path + "/data/weights_theano.h5"
        rospy.loginfo('perception_face_node - Smile Weight Path: {}'.format(smile_recognition_weights))
        self.smile_recognition_worker = SmileRecognitionWorker(smile_recognition_weights,
                                                               self.emo_job_queue, self.emo_result_queue)
        self.smile_recognition_worker.start()

        # A Worker for Hairlength Detection
        self.hairlength_job_queue = Queue()
        self.hairlength_result_queue = Queue()
        hairlength_detection_weights = pkg_path + "/data/hairlength_model_keras.hd5"
        rospy.loginfo('perception_face_node - Hairlength Weight Path: %s',
					  hairlength_detection_weights)
        self.hairlength_detection_worker = HairLengthDetectionWorker(
                                                     hairlength_detection_weights,
                                                     self.hairlength_job_queue,
                                                     self.hairlength_result_queue)
        self.hairlength_detection_worker.start()

        # A Worker for Hairlength Detection
        self.clothcolor_job_queue = Queue()
        self.clothcolor_result_queue = Queue()
        clothcolor_detection_weights = pkg_path + "/data/color_model_hsv_keras.hd5"
        rospy.loginfo('perception_face_node - ClothColor Weight Path: %s',
					  clothcolor_detection_weights)
        self.clothcolor_detection_worker = ClothColorDetectionWorker(
                                                     clothcolor_detection_weights,
                                                     self.clothcolor_job_queue,
                                                     self.clothcolor_result_queue)
        self.clothcolor_detection_worker.start()

        # Topic Subscriptions
        image_sub = message_filters.Subscriber("Color_Image", Image)
        fd_sub = message_filters.Subscriber("Tracked_People", PersonPerceptArray)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, fd_sub], 5, 0.05)
        ts.registerCallback(self.callback)

        face_memorized_sub = rospy.Subscriber('/mhri/face_memorized', String, self.face_memorized_sub)

        self.pub = rospy.Publisher("/mhri/perception_face/persons", PersonPerceptArray, queue_size=1)

        self.skip_count = 0

        # Subscribe to the event channel
        rospy.Subscriber("/mhri/events", PerceptionEvent, self.handle_event)
        self.identity_recognition_completed = []
        self.categories = ['face_recognition', 'cloth_color', 'hair_length', 'eyeglasses', 'gender']
        self.recognition_completed = {}
        for category in self.categories:
            self.recognition_completed[category] = []

        self.loop_count = 0

        rospy.loginfo("PerceptionFace Initialized.")

    def face_memorized_sub(self, msg):
        self.worker.load_faces()

    def handle_event(self, event_msg):
        human_id = event_msg.data
        if event_msg.event_id == 1:
            rospy.logdebug("STOPPING FACE RECOG for %d", human_id)
            self.recognition_completed['face_recognition'].append(human_id)
        elif event_msg.event_id == 3:
            rospy.logdebug("STOPPING HAIR RECOG for %d", human_id)
            self.recognition_completed['hair_length'].append(human_id)
        elif event_msg.event_id == 2:
            rospy.logdebug("STOPPING CLOTHCOLOR RECOG for %d", human_id)
            self.recognition_completed['cloth_color'].append(human_id)

    def calc_area(self, region):
        return (region.right - region.left) * (region.bottom - region.top)

    def select_biggest_region(self, regions):
        if len(regions) <= 1:
            return regions[0]
        else:
            biggest_region = regions[0]
            biggest_area = self.calc_area(biggest_region)
            for i in range(1, len(regions)):
                cur_area = self.calc_area(regions[i])
                if cur_area > biggest_area:
                    biggest_region = regions[i]
                    biggest_area = cur_area
            return biggest_region

    def evaluate_face_quality(self, image, percept):
        # 얼굴이 없으면 건너뜀
        if percept.face_detected == 0:
            return False
        if percept.face_roi.width * percept.face_roi.height < self.min_face_size:
            rospy.logdebug("Person %d has no or small face.", percept.trk_id)
            return False
        return True

    def detect_face(self, image, percept):
        # refine tracking bbox coordinates to be inside image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)

        (img_height, img_width, channel) = cv_image.shape

        percept.trk_bbox_x = max(percept.trk_bbox_x, 0)
        percept.trk_bbox_y = max(percept.trk_bbox_y, 0)
        percept.trk_bbox_width = min(percept.trk_bbox_x + percept.trk_bbox_width, img_width) - percept.trk_bbox_x - 1
        percept.trk_bbox_height = min(percept.trk_bbox_y + percept.trk_bbox_height, img_height) - percept.trk_bbox_y - 1

        # get a person roi
        left = percept.trk_bbox_x
        top = 0
        right = left + percept.trk_bbox_width
        bottom = top + percept.trk_bbox_height

        roi = cv_image[top:bottom, left:right]

        # detect faces and select the biggest one
        face_regions = self.face_detector.detect(roi)
        if len(face_regions) == 0:
            percept.face_detected = 0
        else:
            face_region = self.select_biggest_region(face_regions)
            percept.face_detected = 1

            face_region.left = left + face_region.left
            face_region.top = top + face_region.top
            face_region.right = left + face_region.right
            face_region.bottom = top + face_region.bottom

            landmarks = self.face_aligner.detect_landmarks(cv_image, face_region)

            percept.face_roi.x_offset = max(1, landmarks[1][0])
            percept.face_roi.y_offset = max(1, min(landmarks[19][1], landmarks[24][1]))
            percept.face_roi.width = landmarks[15][0] - percept.face_roi.x_offset
            percept.face_roi.height = landmarks[9][1] - percept.face_roi.y_offset

            percept.stasm_landmarks = []
            for point in landmarks:
                percept.stasm_landmarks.append(point[0])
                percept.stasm_landmarks.append(point[1])

            percept.face_direction = 0
            fl = (landmarks[27][0] - percept.face_roi.x_offset) / float(percept.face_roi.width)
            if fl <= 0.4:
                percept.face_direction = 1;
            elif fl >= 0.6:
                percept.face_direction = 2;

            upper_lip_center = landmarks[62][1];
            lower_lip_center = landmarks[67][1];
            if abs(upper_lip_center - lower_lip_center) > 3:
                percept.mouth_opened = 1;
            else:
                percept.mouth_opened = 0;

            # get body roi
            percept.body_roi.x_offset = max(1, percept.face_roi.x_offset - (int)(percept.face_roi.width / 3))
            percept.body_roi.y_offset = max(1, percept.face_roi.y_offset + percept.face_roi.height + (int)(percept.face_roi.height / 4))
            maxX = min(img_width - 1, percept.body_roi.x_offset + (int)(1.67 * percept.face_roi.width))
            percept.body_roi.width = maxX - percept.body_roi.x_offset
            maxY = min(img_height - 1, percept.body_roi.y_offset + 2 * percept.face_roi.height)
            percept.body_roi.height = maxY - percept.body_roi.y_offset

            if percept.body_roi.height <= 0:
                percept.body_roi.width = 0

        if percept.face_detected == 1:
            return True
        else:
            return False

    def identification_completed(self, subject_id, category):
        # 인식이 완료되었으면 건너뜀
        if subject_id in self.recognition_completed[category]:
            rospy.logdebug('Person %d identification completed in %s. skip...!', subject_id, category)
            return True
        else:
            return False

    def need_to_recognize(self, image, percept):
        return self.evaluate_face_quality(image, percept) == True and self.identification_completed(percept) == False

    def callback(self, image, faces):
        '''
        ROS Callback for Face Recognition
        '''
        rospy.logdebug("perception_face_node called.")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError, e:
            rospy.logerr('{}'.format(e))

        self.loop_count += 1
        token = self.loop_count % 3

        persons_to_face_recog = []
        persons_with_good_face = []
        persons_with_face = []
        for percept in faces.person_percepts:
            rospy.logdebug('Face Size = %dx%d', percept.face_roi.width, percept.face_roi.height)

            # if no face is found on this ROI, continue to the next one
            if not self.detect_face(image, percept):
                continue

            face_quality = self.evaluate_face_quality(image, percept)
            known_face = self.identification_completed(percept.trk_id, "face_recognition")
            if face_quality is True and known_face is False:
                rospy.logdebug("Person %d has good face.", percept.trk_id)
                persons_to_face_recog.append(percept)
            if face_quality is True:
                persons_with_good_face.append(percept)
            persons_with_face.append(percept)

        if len(persons_to_face_recog) > 0:
            if self.data_queue.empty() is True:
                faces_data = []
                emo_data = []
                for percept in persons_to_face_recog:
                    faces_data.append((percept.trk_id, cv_image, percept.stasm_landmarks))
                self.data_queue.put((time.time(), faces_data))
                self.skip_count = 0
            else:
                self.skip_count += 1

        if len(persons_with_good_face) > 0:
            if self.emo_job_queue.empty() is True:
                emo_data = []
                for percept in persons_with_good_face:
                    emo_data.append((percept.trk_id, cv_image, percept.face_roi))
                self.emo_job_queue.put((time.time(), emo_data))

            if token == 1 and self.identification_completed(percept.trk_id, 'hair_length') is False and self.hairlength_job_queue.empty() is True:
                hairlength_data = []
                for percept in persons_with_good_face:
                    hairlength_data.append((percept.trk_id, cv_image, percept.stasm_landmarks))
                self.hairlength_job_queue.put((time.time(), hairlength_data))

        if len(persons_with_face) > 0:
            if token == 2 and self.identification_completed(percept.trk_id, 'cloth_color') is False and self.clothcolor_job_queue.empty() is True:
                clothcolor_data = []
                for percept in persons_with_good_face:
                    clothcolor_data.append((percept.trk_id, cv_image, percept.body_roi))
                self.clothcolor_job_queue.put((time.time(), clothcolor_data))

        if self.result_queue.empty() is False:
            timestamp, results = self.result_queue.get()
            self.result_queue.task_done()
            for percept in faces.person_percepts:
                if results.has_key(percept.trk_id) is True:
                    if results[percept.trk_id] is None:
                        percept.person_id = 'unknown'
                        percept.name = 'unknown'
                        percept.person_confidence = 1
                    else:
                        percept.person_id = results[percept.trk_id][0]
                        percept.person_confidence = results[percept.trk_id][1][percept.person_id]
                        if self.names.has_key(percept.person_id):
                            percept.name = self.names[percept.person_id]
                        else:
                            percept.name = unicode('아무개', 'utf-8')
            rospy.logdebug('Face Recognition: {} after {:4.2f}, duration={:4.2f}'.format(results, time.time() - self.last_recognition_time, time.time() - timestamp))
            self.last_recognition_time = time.time()

        if self.emo_result_queue.empty() is False:
            timestamp, results = self.emo_result_queue.get()
            self.emo_result_queue.task_done()
            for percept in faces.person_percepts:
                if results.has_key(percept.trk_id) is True:
                    percept.emotion = results[percept.trk_id][0]
                    percept.emotion_prob = results[percept.trk_id][1]
            rospy.logdebug('Emotion Recognition: {} with duration {:4.2f}'.format(results, time.time() - timestamp))

        if self.hairlength_result_queue.empty() is False:
            timestamp, results = self.hairlength_result_queue.get()
            self.hairlength_result_queue.task_done()
            for percept in faces.person_percepts:
                if results.has_key(percept.trk_id) is True:
                    percept.hair_length = results[percept.trk_id]
            rospy.logdebug('HairLength Detection: {} with duration {:4.2f}'.format(results, time.time() - timestamp))

        if self.clothcolor_result_queue.empty() is False:
            timestamp, results = self.clothcolor_result_queue.get()
            self.clothcolor_result_queue.task_done()
            for percept in faces.person_percepts:
                if results.has_key(percept.trk_id) is True:
                    percept.cloth_color = results[percept.trk_id]
            rospy.logdebug('ClothColor Detection: {} with duration {:4.2f}'.format(results, time.time() - timestamp))

        rospy.logdebug("FACE Publishing %d", len(faces.person_percepts))
        self.pub.publish(faces)


    def stop(self):
        self.worker.stop()
        self.smile_recognition_worker.stop()


if __name__ == '__main__':
    rospy.init_node('perception_face_node', anonymous=False)

    # minimum face size for face recognition is 48*48
    m = PerceptionFace(48*48)

    rospy.spin()
    m.stop()
