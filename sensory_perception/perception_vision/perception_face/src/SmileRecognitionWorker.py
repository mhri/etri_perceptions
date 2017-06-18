'''
A threaded worker for smile recognition

Author: Minsu Jang (minsu@etri.re.kr)
'''

import rospy
import numpy as np
import cv2
from QueuedProcessing import Worker
from SmileRecognizer import SmileRecognizer

class SmileRecognitionWorker(Worker):
    '''
    A threaded worker for smile recognition.
    '''
    def __init__(self, model_weight_file_path, job_queue, result_queue):
        Worker.__init__(self)
        self.smile_recognizer = SmileRecognizer(model_weight_file_path)
        self.set_queues(job_queue, result_queue)

    def recognize(self, img, x, y, w, h):
        '''
        Perform smile recognition.
        '''
        data = cv2.resize(img[y:y+h, x:x+w], (32, 32))
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        data = cv2.equalizeHist(data)
        data = data.astype(np.float) / 255.

        rospy.logdebug("SHAPE: {}".format(data.shape))
        probs = self.smile_recognizer.predict(data)
        rospy.logdebug("Smile: {:3.2f} vs Neutral {:3.2f}".format(probs[1], probs[0]))

        # Smile = 0, Neutral = 1
        emotion = 1 if probs[1] > 0.3 else 0
        emotion_prob = probs[emotion]

        return emotion, emotion_prob


    def work(self, jobs):
        results = {}
        for job_data in jobs:
            person_id = job_data[0]
            img = job_data[1]
            face_roi = job_data[2]
            emotion, prob = self.recognize(img,
                                           face_roi.x_offset,
                                           face_roi.y_offset,
                                           face_roi.width,
                                           face_roi.height)
            rospy.logdebug('SmileRecognition: {} {:3.2f}'.format(emotion, prob))
            results[person_id] = (emotion, prob)
        return results

