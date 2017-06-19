#!/usr/bin/env python
#-*- encoding: utf8 -*-

import rospy
import rospkg
import threading, time
import numpy as np
import cPickle as pickle
import os.path
from std_msgs.msg import String
from perception_common.msg import CognitiveState
from perception_msgs.msg import PersonPerceptArray
import message_filters
import cv2
#from py2neo import Graph, Path, Relationship, Node
#import rdflib
#import Queue
#from matplotlib import pyplot as plt

class BeliefUpdateAgent:
	def __init__(self):
		# get an instance of RosPack with the default search paths
		rospack = rospkg.RosPack()

		#people_sub = message_filters.Subscriber("Tracked_People", PersonPerceptArray)
		#hair_sub = message_filters.Subscriber("People_HairLength_Detected", PersonPerceptArray)
		#cloth_sub = message_filters.Subscriber("People_ClothColor_Detected", PersonPerceptArray)
		#pos3d_sub = message_filters.Subscriber("People_Pos3D_Detected", PersonPerceptArray)
		#fexpr_sub = message_filters.Subscriber("People_Emotion_Detected", PersonPerceptArray)
		#ts = message_filters.ApproximateTimeSynchronizer([people_sub, hair_sub, cloth_sub], 2, 0.1)
		#ts.registerCallback(self.callback)

		self.face_sub = rospy.Subscriber("Tracked_People", PersonPerceptArray, self.callback_single)

		self.belief = {}
		self.valueIndex = {}
		self.indexValue = {}
		self.index = {}
		self.pub = rospy.Publisher("/mhri/social_perception_core/beliefs/persons", PersonPerceptArray, queue_size=1)

		self.colors = {}
		self.colors['black'] = 'dark'
		self.colors['blue'] = 'bright'
		self.colors['brown'] = 'dark'
		self.colors['green'] = 'bright'
		self.colors['grey'] = 'dark'
		self.colors['orange'] = 'colorful'
		self.colors['pink'] = 'colorful'
		self.colors['purple'] = 'colorful'
		self.colors['red'] = 'colorful'
		self.colors['white'] = 'bright'
		self.colors['yellow'] = 'bright'

		# 얼굴 아이디에 대한 정보를 로딩한다.
		self.person_info = {}
		person_info_file = os.path.join(rospkg.RosPack().get_path('social_perception_core'), 'data', 'person-info.txt')
		if os.path.isfile(person_info_file):
			lines = [line.rstrip('\n') for line in open(person_info_file)]
			for line in lines:
				strs = line.split(':')
				if len(strs) < 2:
					continue
				self.person_info[int(strs[0])] = strs[1]

		self.episodic = False

		if self.episodic is True:
			self.g = Graph()
		else:
			self.g = None


	def callback_single(self, data):
		rospy.logdebug("BeliefUpdateAgent callback_single Called!")
		for percept in data.person_percepts:
			#if percept.session_face_id is None:
			#	continue
			if self.episodic is True:
				self.memorize(percept)

			cs = CognitiveState()
			if percept.cognitive_status.upper() == 'SURE':
				cs.state = "sure"
				#self.notifyProspect('positive')
			elif percept.cognitive_status.upper() == 'UNSURE':
				cs.state = "unsure"
				#self.notifyProspect('negative')
			elif percept.cognitive_status.upper() == 'THINKING':
				cs.state = "thinking"
			else:
				cs.state = ""

			if percept.person_id in self.person_info:
				percept.name = self.person_info[percept.person_id]
			else:
				percept.name = "Unknown"

			if self.episodic is True:
				self.update_decision(percept)

		if len(data.person_percepts) > 0:
			self.curr_percepts = data.person_percepts

		#rospy.loginfo("[belief_update_agent] publishing...: %d", len(data.person_percepts))

		self.pub.publish(data)


	def callback(self, data, hair_msg, cloth_msg):
		rospy.logdebug("BeliefUpdateAgent Called!")

		hair_data = {}
		for percept in hair_msg.person_percepts:
			hair_data[percept.trk_id] = percept.hair_length

		cloth_data = {}
		for percept in cloth_msg.person_percepts:
			cloth_data[percept.trk_id] = percept.cloth_color

		'''
		pos3d_data = {}
		for percept in pos3d_msg.person_percepts:
			pos3d_data[percept.trk_id] = percept.face_pos3d
		'''
		'''
		emotion_data = {}
		for percept in emotion_msg.person_percepts:
			emotion_data[percept.trk_id] = (percept.emotion, percept.emotion_prob)
		'''
		for percept in data.person_percepts:
			#if percept.session_face_id is None:
			#	continue
			if self.episodic is True:
				self.memorize(percept)

			cs = CognitiveState()
			if percept.cognitive_status.upper() == 'SURE':
				cs.state = "sure"
				#self.notifyProspect('positive')
			elif percept.cognitive_status.upper() == 'UNSURE':
				cs.state = "unsure"
				#self.notifyProspect('negative')
			elif percept.cognitive_status.upper() == 'THINKING':
				cs.state = "thinking"
			else:
				cs.state = ""

			if percept.person_id in self.person_info:
				percept.name = self.person_info[percept.person_id]
			else:
				percept.name = "Unknown"

			if self.episodic is True:
				self.update_decision(percept)

			if cloth_data.has_key(percept.trk_id):
				percept.cloth_color = cloth_data[percept.trk_id]

			if hair_data.has_key(percept.trk_id):
				percept.hair_length = hair_data[percept.trk_id]
			'''
			if pos3d_data.has_key(percept.trk_id):
				percept.face_pos3d = pos3d_data[percept.trk_id]
				percept.frame_id = 'camera_depth_optical_frame'
			'''
			'''
			if emotion_data.has_key(percept.trk_id):
				percept.emotion = emotion_data[percept.trk_id][0]
				percept.emotion_prob = emotion_data[percept.trk_id][1]
			'''
		if len(data.person_percepts) > 0:
			self.curr_percepts = data.person_percepts

		rospy.logdebug("[belief_update_agent] publishing...: ")

		self.pub.publish(data)

	'''
	def hasBigEyes(self, landmarks):
		# ROI for hair learning and detection
		fX1 = landmarks[1*2]
		fY1 = min(landmarks[19*2+1], landmarks[24*2+1])
		fX2 = landmarks[15*2]
		fY2 = landmarks[9*2+1]
		# Eye Size
		leftEyeSizeRatio = (landmarks[46*2+1] - landmarks[44*2+1])/(float)(fY2-fY1)
		rightEyeSizeRatio = (landmarks[40*2+1] - landmarks[38*2+1])/(float)(fY2-fY1)

		return leftEyeSizeRatio > 0.05
	'''

	def get_chunk(self, id):
		if self.belief.get(id) is None:
			self.graph.Create('Person','person_session_id',id)
			self.belief[id] = {}
			return self.belief[id]
		else:
			return self.belief[id]

	def accumulateKnowledgeItem(self, chunk, item, value):
		valueId = self.get_index(item, value)
		#print 'index of (', item, ',', value, ') is ', valueId
		if chunk.get(item) is None:
			chunk[item] = []
		chunk[item].append(valueId)
		if len(chunk[item]) > 200:
			chunk[item] = chunk[item][1:]

	def decision(self, chunk, item):
		percepts = chunk.get(item)
		if percepts is None:
			return None
		else:
			hist, bin_edges = np.histogram(chunk[item],bins=range(0,self.index[item]+1))
			i = self.max_index(hist)
			return self.get_value(item,bin_edges[i])

	def get_index(self, item, value):
		if self.valueIndex.get(item) is None:
			self.valueIndex[item] = {}
			self.indexValue[item] = {}
			self.index[item] = 0
		if self.valueIndex[item].get(value) is None:
			self.valueIndex[item][value] = self.index[item]
			self.indexValue[item][self.index[item]] = value
			self.index[item] = self.index[item] + 1
		return self.valueIndex[item][value]

	def get_value(self, item, index):
		return self.indexValue[item][index]

	def max_index(self, hist):
		maxIndex = 0
		for i in range(1,len(hist)):
			if hist[i] > hist[maxIndex]:
				maxIndex = i
		return maxIndex

if __name__ == '__main__':
	rospy.init_node('visualize', anonymous=False)

	m = BeliefUpdateAgent()

	rospy.spin()
