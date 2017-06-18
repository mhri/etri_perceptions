#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
Hair Length Detector

Author: Minsu Jang (minsu@etri.re.kr)
'''

import rospy
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

class HairLengthDetector(object):
	def __init__(self, input_shape, num_classes, model_path=None):
		self.model = self.create_model(input_shape, num_classes)
		if model_path != None:
			self.model.load_weights(model_path)
		self.styleIndex = {}
		self.styleIndex[0] = 'long'
		self.styleIndex[1] = 'short'

	def create_model(self, input_shape, num_classes):
		model = Sequential()
		model.add(Dense(300, activation='sigmoid', input_shape=(input_shape,)))
		#model.add(Dropout(0.2))
		model.add(Dense(256, activation='relu'))
		#model.add(Dropout(0.2))
		model.add(Dense(300, activation='relu'))
		#model.add(Dropout(0.2))
		model.add(Dense(num_classes, activation='softmax'))

		model.summary()
		return model

	def extract_roi(self, img, landmarks):
		mask = np.zeros(img.shape, dtype=np.uint8)
		mask[:] = 255

		# Extract Face Region
		roi_corners = []
		height, width = img.shape[:2]

		# ROI for hair learning and detection
		fX1 = landmarks[1*2]
		fY1 = min(landmarks[19*2+1], landmarks[24*2+1])
		fX2 = landmarks[15*2]
		fY2 = landmarks[9*2+1]

		roi_corners.append((fX1,fY1))
		roi_corners.append((fX1,height))
		roi_corners.append((fX2,height))
		roi_corners.append((fX2,fY1))
		poly_points = np.array([roi_corners], dtype=np.int32)
		black = (255,255,255)
		cv2.fillPoly(img, poly_points, black)

		xExtension = (fX2-fX1)/2
		fl = (landmarks[27*2]-fX1)/(float)(fX2-fX1)
		fr = (fX2-landmarks[27*2])/(float)(fX2-fX1)

		#print "ratio => ", fl, " , ", fr
		hX1 = (int)(max(1,fX1-xExtension*fl))
		hY1 = (int)(max(1,fY1-(int)((fY2-fY1)/1.5)))
		hX2 = (int)(min(width-1,fX2+(int)(xExtension*fr)))
		hY2 = (int)(min(height-1,fY2+(fY2-fY1)/3))

		return img[hY1:hY2,hX1:hX2]


	def extract_feature(self, roi):
		feature = []
		image = cv2.resize(roi, (100,152))
		ret, image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		sumV = image.sum()
		feature = []
		for x in range(0,4):
			for y in range(0,4):
				feature.append(image[y*38:(y+1)*38,x*25:(x+1)*25].sum()/(float)(sumV))
		return feature


	def identify_hair_length(self, image, landmarks):
		'''
		Arguments
		---------
		image: an opencv image in BGR
		landmarks: facial landmark points
		'''
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		roi = self.extract_roi(image, landmarks)

		'''
		fname = '/home/minsu/Developments/mhri/face_' + str(self.x) + '.jpg'
		rospy.loginfo('Writing a face image to a file: %s', fname)
		cv2.imwrite(fname, roi)
		self.x = self.x + 1
		'''
		feature = self.extract_feature(roi)
		x = np.array([feature])
		predictions = self.model.predict(x)
		#print '*****PREDICTIONS: ', predictions, ', shape=', predictions.shape
		index = np.argmax(predictions[0])
		#print '*****PREDICTION ARGMAX: ', index, ' = ', self.styleIndex[index]
		return self.styleIndex[index]
