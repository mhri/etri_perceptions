#!/usr/bin/env python
#-*- encoding: utf8 -*-
'''
Social Perception for Intention Recognition

Recognized intentions are: concentrating, affirmation, negation

Author: Minsu Jang (minsu@etri.re.kr)
'''

import rospy
import json
import time
import numpy
from perception_msgs.msg import *
from std_msgs.msg import Empty
from perception_base.perception_base import PerceptionBase


class IntentionRecognition(PerceptionBase):
	# Initialization
	def __init__(self):
		super(IntentionRecognition, self).__init__("intention_recognition")
		self.minds = {}
		rospy.Subscriber("People_With_CognitiveStatus", PersonPerceptArray, self.handle)

	def get_event_data_in_json(self, face_id):
		data = json.loads('{}')
		data['face_id'] = face_id
		return json.dumps(data)

	def write_event(self, percept):
		mind_status = percept.cognitive_status.lower()
		if percept.cognitive_status == '':
			mind_status = 'concentrating'

		if mind_status.lower() == 'agree':
			event = 'intention_affirmation'
		elif mind_status.lower() == 'disagree':
			event = 'intention_negation'
		elif mind_status.lower() == 'concentrating':
			event = 'intention_concentration'

		wr_data = self.get_event_data_in_json(percept.trk_id)

		rospy.loginfo("SOCIAL EVENT: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'facial_expression_detected')


	def handle(self, msg):
		no_of_people = len(msg.person_percepts)
		rospy.logdebug("INTENTION_RECOGNITION No_of_People = %d", no_of_people)

		percepts = msg.person_percepts

		for percept in percepts:
			if percept.trk_id not in self.minds:
				self.minds[percept.trk_id] = ''
			prev_status = self.minds[percept.trk_id]
			cur_status = percept.cognitive_status
			if cur_status != '' and cur_status != prev_status:
				rospy.logdebug("INTENTION(%d): CUR=%s PREV=%s", percept.trk_id, cur_status, prev_status)
				self.minds[percept.trk_id] = cur_status
				self.write_event(percept)

		to_be_deleted = []
		for key in self.minds:
			found = False
			for percept in percepts:
				if key == percept.trk_id:
					found = True
					break
			if not found:
				to_be_deleted.append(key)

		for key in to_be_deleted:
			del self.minds[key]

if __name__ == '__main__':
	rospy.init_node('intention_recognition', anonymous=False)
	m = IntentionRecognition()
	rospy.spin()
