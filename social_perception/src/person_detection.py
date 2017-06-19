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

		self.pps_pub = rospy.Publisher("/mhri/person_presence_state",
									   PersonPresenceState, queue_size=1)

		#rospy.Subscriber("/mhri/people_tracking/alive", PersonIDArray, self.ids_callback)

	def checkAppearance(self, percepts, persons):
		rospy.logdebug("Appear or Disappear?!!! Prev_Percepts=%d, Cur_Percepts=%d", len(persons), len(percepts))
		# check appearance
		appeared = []
		disappeared = list(persons)
		for percept in percepts:
			#rospy.loginfo('Percept: %d', percept.trk_id)
			found = False
			for person in persons:
				if percept.session_face_id == person.session_face_id:
					found = True
					disappeared.remove(person)
					break
			if found == False:
				appeared.append(percept)

		return appeared, disappeared


	def presenceToData(self, presence):
		'''
		Transform person presence data into the json format.
		'''
		data = json.loads('{}')
		data['count'] = len(presence)
		data['person_id'] = []
		for percept in presence:
			data['person_id'].append(percept.trk_id)
		return json.dumps(data)


	def write_presence(self, appeared, disappeared):
		'''
		사람 존재 이벤트를 메모리에 기록 --> 이벤트 발생
		person_detection:
			person_appeared: false
			person_disappeared: false
			person_id: []
			count: 0
		'''
		if len(appeared) == 0 and len(disappeared) == 0:
			return

		rospy.wait_for_service('/social_memory/write_data', timeout=1.0)
		wr_memory = rospy.ServiceProxy('/social_memory/write_data', WriteData)


		if len(appeared) > 0:
			# appearance
			wr_data = self.presenceToData(appeared)
			rospy.loginfo("person_appeared: %s", wr_data)

			self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
			self.raise_event(self.conf_data.keys()[0], 'person_appeared')

		if len(disappeared) > 0:
			# disappearance
			wr_data = self.presenceToData(disappeared)
			rospy.loginfo("person_disappeared: %s", wr_data)

			self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
			self.raise_event(self.conf_data.keys()[0], 'person_disappeared')

		'''
			PersonPresenceState:
				int16 count
				string[] appeared
				string[] disappeared
		'''
		# Notify Episodic Memory
		if len(appeared) > 0 or len(disappeared) > 0:
			pps = PersonPresenceState()
			pps.count = len(appeared)
			pps.appeared = []
			pps.disappeared = []
			for percept in appeared:
				pps.appeared.append(percept.trk_id)
			for percept in disappeared:
				pps.disappeared.append(percept.trk_id)
			self.pps_pub.publish(pps)

			#for percept in disappeared:
			#	self.graph.Set('Person', 'session_id', percept.trk_id, 'disappeared_at', time.time())


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

			appeared, disappeared = self.checkAppearance(percepts, self.persons)

			self.write_presence(appeared, disappeared)

			self.persons = percepts

		except rospy.ServiceException, e:
			rospy.logerr("Service call failed: %s" % e)

		self.lastTime = time.time()


if __name__ == '__main__':
	rospy.init_node('person_detection', anonymous=False)
	m = PersonIdentityProcessor()
	rospy.spin()
