#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
 Author: Minsu Jang (minsu@etri.re.kr)
'''

import rospy
import json
import time
import numpy
from perception_msgs.msg import *
from std_msgs.msg import Empty
from perception_base.perception_base import PerceptionBase

class PersonEngagement:
	def __init__(self, id, listener):
		self.id = id
		self.engaged = 0	# 0: not engaged, 1: waiting to be engaged,
							# 2: engaged, 3: waiting not to be engaged
		self.listener = listener
		self.last_updated = time.time()
		self.duration_threshold_ = 3

	def IsLookingAtMe(self):
		self.Updated()
		if self.engaged == 0:
			self.engaged = 1
			self.timestamp = time.time()
		elif self.engaged == 1:
			duration = time.time() - self.timestamp
			if duration > self.duration_threshold_:
				self.engaged = 2
				self.listener.Engaged(self.id)
		elif self.engaged == 2:
			pass
		elif self.engaged == 3:
			self.engaged = 1
			self.timestamp = time.time()

	def IsLookingSomewhereElse(self):
		self.Updated()
		if self.engaged == 0:
			pass
		elif self.engaged == 1:
			self.engaged = 3
			self.timestamp = time.time()
		elif self.engaged == 2:
			self.engaged = 3
			self.timestamp = time.time()
		elif self.engaged == 3:
			duration = time.time() - self.timestamp
			if duration > self.duration_threshold_:
				self.engaged = 0
				self.listener.Disengaged(self.id)

	def Updated(self):
		self.last_updated = time.time()

	def NotUpdatedForQuiteAwhile(self):
		duration = time.time() - self.last_updated
		if duration > 5:
			return True
		else:
			return False

class EngagementDetection(PerceptionBase):

	def __init__(self):
		super(PersonEngagement, self).__init__("engagement_detection")

		rospy.Subscriber(
			"/mhri/social_perception_core/beliefs/persons", PersonPerceptArray, self.handle)
		self.persons = {}
		self.lastTime = time.time()

	def FaceIsTowardsMe(self, landmarks):
		if landmarks == None or len(landmarks) == 0:
			return False
		faceDirection = True
		fX1 = landmarks[1*2]
		fX2 = landmarks[15*2]
		fl = (landmarks[27*2]-fX1)/(float)(fX2-fX1)
		fr = (fX2-landmarks[27*2])/(float)(fX2-fX1)
		if (fr-fl) >= 0.3:
			faceDirection = False
		elif (fr-fl) <= -0.4:
			faceDirection = False
		return faceDirection

	def ConvertEngagementData2Json(self, face_id):
		data = json.loads('{}')
		data['face_id'] = face_id
		return json.dumps(data)

	def Engaged(self, id):
		wr_data = self.ConvertEngagementData2Json(id)

		rospy.loginfo("PERSON_ENGAGED: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'person_engaged')

	def Disengaged(self, id):
		wr_data = self.ConvertEngagementData2Json(id)

		rospy.loginfo("PERSON_DISENGAGED: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'person_disengaged')

	def handle(self, msg):
		try:
			rospy.logdebug("No_of_People = %d", len(msg.person_percepts))

			duration = time.time() - self.lastTime
			if duration < 0.2:
				rospy.logdebug("Check Frequency: %d ==> Skip This Time...", duration)
				return

			percepts = msg.person_percepts

			for percept in percepts:
				# Engagement is measured only for people inside 1.5m boundary
				if percept.trk_id not in self.persons:
					self.persons[percept.trk_id] = PersonEngagement(percept.trk_id, self)
				engagement_manager = self.persons[percept.trk_id]
				if percept.face_detected == 1:
					landmarks = percept.stasm_landmarks
				else:
					landmarks = None
				looking_at_me = self.FaceIsTowardsMe(landmarks)
				if looking_at_me is True:
					engagement_manager.IsLookingAtMe()
				else:
					engagement_manager.IsLookingSomewhereElse()

			to_be_deleted = []
			for key in self.persons:
				engagement_manager = self.persons[key]
				if engagement_manager.NotUpdatedForQuiteAwhile() == True:
					to_be_deleted.append(key)

			for key in to_be_deleted:
				del self.persons[key]

		except rospy.ServiceException, e:
			rospy.logerror("Service call failed: %s" % e)

if __name__ == '__main__':
	rospy.init_node('engagement_detection', anonymous=False)
	m = EngagementDetection()
	rospy.spin()
