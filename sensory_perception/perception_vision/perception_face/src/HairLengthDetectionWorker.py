#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
A threaded worker for hair length detection

Author: Minsu Jang (minsu@etri.re.kr)
'''

import rospy
from HairLengthDetector import HairLengthDetector
from QueuedProcessing import Worker


class HairLengthDetectionWorker(Worker):
	'''
	A Threaded Worker for Hair Length Detection
	'''
	def __init__(self, model_path, data_queue, result_queue):
		Worker.__init__(self)
		self.set_queues(data_queue, result_queue)
		self.detector = HairLengthDetector(16, 2, model_path)

	def recognize(self, img, landmarks):
		return self.detector.identify_hair_length(img, landmarks)

	def work(self, faces):
		results = {}
		for face in faces:
			person_id = face[0]
			img = face[1]
			landmarks = face[2]

			hair_length = self.recognize(img, landmarks)
			
			rospy.logdebug('HairLengthDetection: %s', hair_length)

			if hair_length is not None:
				results[person_id] = hair_length
			else:
				results[person_id] = None
		return results

