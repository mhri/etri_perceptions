#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
Cloth Color Detector

Author: Minsu Jang (minsu@etri.re.kr)
'''

import rospy
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.externals import joblib
from nolearn.dbn import DBN


class ColorDetector:
	def __init__(self, input_shape, num_classes, model_file=None):
		self.color_index = {}
		self.color_index[0] = 'black'
		self.color_index[1] = 'blue'
		self.color_index[2] = 'brown'
		self.color_index[3] = 'green'
		self.color_index[4] = 'grey'
		self.color_index[5] = 'orange'
		self.color_index[6] = 'pink'
		self.color_index[7] = 'purple'
		self.color_index[8] = 'red'
		self.color_index[9] = 'white'
		self.color_index[10] = 'yellow'

		self.colorNames = {}
		self.colorNames = {v: k for k, v in self.color_index.items()}

		self.model = self.create_model(input_shape, num_classes)
		if model_file != None:
			self.model.load_weights(model_file)

		rospy.loginfo("ColorDetector Initialized.")


	def extract_feature(self, img):
		hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		feature = []
		hist = cv2.calcHist( [hsv], [0], None, [40], [0, 180] )
		nf = hsv.shape[0]*hsv.shape[1]
		#print "NF = ", hsv.shape, " & ", hsv.shape[0]
		hist = hist.flatten() / nf
		#print "SUM = ", sum(hist)
		feature = np.concatenate([feature,hist])
		hist = cv2.calcHist( [hsv], [1], None, [40], [0, 256] )
		nf = hsv.shape[0]*hsv.shape[1]
		hist = hist.flatten() / nf
		feature = np.concatenate([feature,hist])
		hist = cv2.calcHist( [hsv], [2], None, [40], [0, 256] )
		nf = hsv.shape[0]*hsv.shape[1]
		hist = hist.flatten() / nf
		feature = np.concatenate([feature,hist])
		np.set_printoptions(precision=4,suppress=True)
		return feature


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


	def classify_color(self, img):
		feature = self.extract_feature(img)
		x = np.array([feature])
		predictions = self.model.predict(x)
		index = np.where(predictions[0] > 0.5)[0][0]
		return self.color_index[index]
