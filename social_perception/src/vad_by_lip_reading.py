#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
Detect Voice Activity by Lip Reading

Date: 2016
Author: Minsu Jang ()
'''

import rospy
import json
import time
import numpy
from perception_msgs.msg import PersonPerceptArray
from std_msgs.msg import String
from perception_base.perception_base import PerceptionBase


class VoiceActivityStateManager:
	def __init__(self, listener, face_id):
		self.mouth_state = 'closed'
		self.talking_prob = 0.0
		self.talking = False
		self.listener = listener
		self.face_id = face_id

	def UpdateTalkingState(self):
		if self.talking is True:
			if self.talking_prob < 0.3:
				self.talking = False
				self.listener.FinishedTalking(self.face_id)
		else:
			if self.talking_prob > 0.7:
				self.talking = True
				self.listener.StartedTalking(self.face_id)

	def MouthOpened(self):
		#print ' --- opened! ', self.talking_prob
		if self.mouth_state == 'closed':
			self.talking_prob = min(1.0, 0.5 + self.talking_prob * 1.5)
		else:
			self.talking_prob = min(1.0, self.talking_prob * 1.2)
		self.UpdateTalkingState()
		self.mouth_state = 'opened'

	def MouthClosed(self):
		if self.mouth_state == 'closed':
			self.talking_prob = min(1.0, self.talking_prob * 0.8)
		else:
			self.talking_prob = min(1.0, self.talking_prob * 1.5)
		self.UpdateTalkingState()
		self.mouth_state = 'closed'


class VoiceActivityDetectionByLipReading(PerceptionBase):
	# Initialization
	def __init__(self):
		super(VoiceActivityDetectionByLipReading, self).__init__("vad_by_lip_reading")
		self.vad_managers = {}
		self.IAmTalking = False

		rospy.Subscriber("People_With_Face", PersonPerceptArray, self.HandleVADState)

		self.lip_talking_started_pub = rospy.Publisher("/mhri/perception/lip_talking_started", String, queue_size=2)
		self.lip_talking_finished_pub = rospy.Publisher("/mhri/perception/lip_talking_finished", String, queue_size=2)


	def ConvertVADData2Json(self, face_id):
		data = json.loads('{}')
		data['face_id'] = face_id
		return json.dumps(data)


	def StartedTalking(self, id):
		self.lip_talking_started_pub.publish(id)

		wr_data = self.ConvertVADData2Json(id)

		rospy.loginfo("PERSON_STARTED_TALKING: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'lip_talking_started')


	def FinishedTalking(self, id):
		self.lip_talking_finished_pub.publish(id)

		wr_data = self.ConvertVADData2Json(id)

		rospy.loginfo("PERSON_FINISHED_TALKING: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'lip_talking_finished')


	def HandleVADState(self, msg):
		no_of_people = len(msg.person_percepts)
		rospy.logdebug("No_of_People = %d", no_of_people)

		percepts = msg.person_percepts

		for percept in percepts:
			if percept.face_detected == 0:
				continue
			if percept.session_face_id not in self.vad_managers:
				self.vad_managers[percept.session_face_id] = VoiceActivityStateManager(self, percept.session_face_id)
			vad_manager = self.vad_managers[percept.session_face_id]
			if percept.mouth_opened == 1:
				vad_manager.MouthOpened()
			else:
				vad_manager.MouthClosed()
			rospy.logdebug(">> Talking Prob: %5.3f --- Talking = %r", vad_manager.talking_prob, vad_manager.talking)

		to_be_deleted = []
		for key in self.vad_managers:
			found = False
			for percept in percepts:
				if key == percept.session_face_id:
					found = True
					break
			if not found:
				to_be_deleted.append(key)

		for key in to_be_deleted:
			del self.vad_managers[key]

if __name__ == '__main__':
	rospy.init_node('vad_by_lip_reading', anonymous=False)

	m = VoiceActivityDetectionByLipReading()

	rospy.spin()
