#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
A threaded worker for cloth color detection

Author: Minsu Jang (minsu@etri.re.kr)
'''

import rospy
from ColorDetector import ColorDetector
from QueuedProcessing import Worker
import cv2

class ClothColorDetectionWorker(Worker):
	'''
	A Threaded Worker for Hair Length Detection
	'''
	def __init__(self, model_path, data_queue, result_queue):
		Worker.__init__(self)
		self.set_queues(data_queue, result_queue)
		self.detector = ColorDetector(120, 11, model_path)
		''' DEBUG
		self.x = 0
		'''

	def recognize(self, img):
		return self.detector.classify_color(img)

	def work(self, data):
		results = {}
		for item in data:
			person_id = item[0]
			img = item[1]
			roi = item[2]

			bX1 = roi.x_offset
			bY1 = roi.y_offset
			bX2 = bX1 + roi.width
			bY2 = bY1 + roi.height

			''' DEBUG
			fname = '/home/minsu/Developments/mhri/cloth_' + str(self.x) + '.jpg'
			rospy.loginfo('Writing a cloth image to a file: %s', fname)
			cv2.imwrite(fname, img[bY1:bY2,bX1:bX2])
			self.x = self.x + 1
			'''

			color = self.recognize(img[bY1:bY2,bX1:bX2])
			#rospy.loginfo('ClothColorDetection: %s', color)

			if color is not None:
				results[person_id] = color
			else:
				results[person_id] = None

		return results

