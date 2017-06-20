#!/usr/bin/env python
#-*- encoding: utf8 -*-

# Authored by Minsu Jang

import rospy
import json
import time
import numpy as np
from perception_msgs.msg import PersonPerceptArray
from std_msgs.msg import Empty
from perception_msgs.msg import PersonIDArray
from perception_common.msg import PersonPresenceState
from Queue import Queue
from perception_base.perception_base import PerceptionBase


class OverwritingQueue(object):
	def __init__(self, size=5):
		self.queue = []
		for i in range(size):
			self.queue.append(0)
		self.max_size = size

	def put(self, item):
		if len(self.queue) >= self.max_size:
			self.queue.pop(0)
		self.queue.append(item)

	def get_all(self):
		return self.queue


class PersonIdentityProcessor(PerceptionBase):

	def __init__(self):
		super(PersonIdentityProcessor, self).__init__("person_detection")
		rospy.Subscriber(
			"/mhri/social_perception_core/beliefs/persons", PersonPerceptArray, self.handle)

		self.persons = []
		self.distance_map = {}
		self.zone_map = {}
		self.lastTime = time.time()
		self.face_detection_state_windows = {}
		self.face_detection_states = {}


	def update_face_detection_data(self, cur_persons):
		'''
		TODO: self.face_detection_state_windows의 아이디별 삭제 코드 필요
		'''
		for cur_person in cur_persons:
			if cur_person.trk_id not in self.face_detection_state_windows:
				self.face_detection_state_windows[cur_person.trk_id] = OverwritingQueue()
			if cur_person.face_detected == 1:
				self.face_detection_state_windows[cur_person.trk_id].put(1)
			else:
				self.face_detection_state_windows[cur_person.trk_id].put(0)


	def event_occurred(self, binary_window):
		'''
		Detect an occurrence of an event based on the binomial (0: not occurred, 1: occurred) observation sequence
		'''
		average = np.average(binary_window)
		if average > 0.7:
			return True
		else:
			return False


	def maintain_face_detection_state(self, cur_persons):
		'''
		Gather faces newly detected and disappeared, and write corresponding events.
		'''
		self.update_face_detection_data(cur_persons)

		curr_detection_states = {}
		for person in cur_persons:
			event_record = self.face_detection_state_windows[person.trk_id].get_all()
			#rospy.loginfo("EVENT_RECORD[%d] = %s", person.trk_id, event_record)
			curr_detection_states[person.trk_id] = self.event_occurred(event_record)

		face_detected = []
		face_disappeared = []
		for cur_person in cur_persons:
			prev_state = None
			if cur_person.trk_id in self.face_detection_states:
				prev_state = self.face_detection_states[cur_person.trk_id]
				#rospy.loginfo('PREV_STATE = %d', prev_state)
			curr_state = curr_detection_states[cur_person.trk_id]
			#rospy.loginfo('CURR_STATE = %d', curr_state)

			if prev_state == None:
				if curr_state == 1:
					face_detected.append(cur_person.trk_id)
			else:
				if curr_state != prev_state:
					if curr_state == 0:
						face_disappeared.append(cur_person.trk_id)
					elif curr_state == 1:
						face_detected.append(cur_person.trk_id)

		self.face_detection_states = curr_detection_states
		#rospy.loginfo(">>> CURR_STATE: %s", self.face_detection_states)

		self.write_face_detection_events(face_detected, face_disappeared)


	def get_face_detection_data(self, face_ids):
		'''
		Transform face detection data into the json format.
		'''
		data = {}
		data['count'] = len(face_ids)
		data['face_id'] = []
		for face_id in face_ids:
			data['face_id'].append(face_id)
		return data


	def write_face_detection_events(self, detected, disappeared):
		'''
		얼굴 검출 이벤트를 메모리에 기록한다.
		face_detection:
			face_detected: false
			face_disappeared: false
			face_id: []
			count: 0
		'''
		if len(detected) > 0:
			# appearance
			wr_data = self.get_face_detection_data(detected)

			rospy.logdebug("face_detected: %s", wr_data)

			self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
			self.raise_event(self.conf_data.keys()[0], 'face_detected')


		if len(disappeared) > 0:
			# disappearance
			wr_data = self.get_face_detection_data(disappeared)

			rospy.logdebug("face_disappeared: %s", wr_data)

			self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
			self.raise_event(self.conf_data.keys()[0], 'face_disappeared')

	def handle(self, msg):
		try:
			rospy.logdebug("No_of_People = %d", len(msg.person_percepts))
			'''
			duration = time.time() - self.lastTime
			if duration < 0.2:
				rospy.logdebug("Check Frequency: %d ==> Skip This Time...", duration)
				return
			'''

			percepts = msg.person_percepts

			self.maintain_face_detection_state(percepts)

			self.persons = percepts

		except rospy.ServiceException, e:
			rospy.logerr("Service call failed: %s" % e)

		self.lastTime = time.time()


if __name__ == '__main__':
	# rospy.init_node('face_detection', anonymous=False)
	m = PersonIdentityProcessor()
	rospy.spin()
