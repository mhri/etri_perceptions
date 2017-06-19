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
		super(PersonIdentityProcessor, self).__init__("presence_detection")
		rospy.Subscriber(
			"/mhri/social_perception_core/beliefs/persons", PersonPerceptArray, self.handle)

		self.persons = []
		self.distance_map = {}
		self.zone_map = {}
		self.lastTime = time.time()
		self.face_detection_state_windows = {}
		self.face_detection_states = {}


	def get_zone(self, dist):
		if dist > 4:
			return 4
		elif dist > 3:
			return 3
		elif dist > 2:
			return 2
		else:
			return 1


	def running_mean(self, x):
		N = len(x)
		cumsum = np.cumsum(np.insert(x, 0, 0))
		return (cumsum[N:] - cumsum[:-N]) / N


	def maintain_distance(self, persons):
		cur_face_ids = []
		for percept in persons:
			cur_face_ids.append(percept.session_face_id)
			if percept.session_face_id not in self.distance_map:
				self.distance_map[percept.session_face_id] = []
			self.distance_map[percept.session_face_id].append(percept.face_pos3d.z)
			self.distance_map[percept.session_face_id] = self.distance_map[percept.session_face_id][-50:]

		# 사라진 사람들을 distance_map과 zone_map에서 삭제한다.
		to_be_removed = []
		for face_id in self.distance_map.keys():
			if face_id not in cur_face_ids:
				to_be_removed.append(face_id)
		for key in to_be_removed:
			del self.distance_map[key]
			del self.zone_map[key]

		# Zone이 변화된 사람들의 정보를 갱신한다.
		approached = []
		left = []
		zone_changed = []
		for face_id in self.distance_map.keys():
			if face_id not in self.zone_map:
				self.zone_map[face_id] = 100
			#print "Distance[", face_id, "] = ", self.running_mean(self.distance_map[face_id])
			if len(self.distance_map[face_id]) < 40:
				continue
			updated_zone = self.get_zone(self.running_mean(self.distance_map[face_id]))
			prev_zone = self.zone_map[face_id]
			if prev_zone != updated_zone:
				# approach and leave detection
				if updated_zone > prev_zone:
					left.append(face_id)
				elif updated_zone < prev_zone:
					approached.append(face_id)
				zone_changed.append(face_id)
				self.zone_map[face_id] = updated_zone

		# 이제 이벤트를 생성할 순서
		if len(zone_changed) > 0:
			self.write_zone_info(zone_changed)
		if len(approached) > 0:
			self.write_approached(approached)
		if len(left) > 0:
			self.write_left(left)


	# 사용자의 거리 영역 내 존재 정보를 JSON 형식으로 변환한다.
	def zoneInfoToData(self, zone_changed):
		data = json.loads('{}')
		data['count'] = len(zone_changed)
		data['face_id'] = []
		data['zone_id'] = []
		for face_id in zone_changed:
			data['face_id'].append(face_id)
			data['zone_id'].append(self.zone_map[face_id])
		return json.dumps(data)


	# 사용자가 특정 거리 영역에 들어선 정보를 메모리에 적는다.
	def write_zone_info(self, zone_changed):
		if len(zone_changed) <= 0:
			return

		wr_data = self.zoneInfoToData(zone_changed)

		rospy.logdebug("PERSON_ZONE_ENTERED: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'person_entered_zone')


	def ConvertApproachData2Json(self, id_list):
		data = json.loads('{}')
		data['count'] = len(id_list)
		data['face_id'] = []
		data['zone_id'] = []
		for face_id in id_list:
			data['face_id'].append(face_id)
		return json.dumps(data)


	# 사용자가 특정 거리 영역에 들어선 정보를 메모리에 적는다.
	def write_approached(self, approached):
		if len(approached) <= 0:
			return

		wr_data = self.ConvertApproachData2Json(approached)

		rospy.logdebug("PERSON_APPROACHED: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'person_approached')


	# 사용자가 떠난 정보를 메모리에 적는다.
	def write_left(self, left):
		if len(left) <= 0:
			return

		wr_data = self.ConvertApproachData2Json(left)

		rospy.logdebug("PERSON_LEFT: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'person_left')


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

			self.maintain_distance(percepts)

			self.persons = percepts

		except rospy.ServiceException, e:
			rospy.logerr("Service call failed: %s" % e)

		self.lastTime = time.time()


if __name__ == '__main__':
	# rospy.init_node('presence_detection', anonymous=False)
	m = PersonIdentityProcessor()
	rospy.spin()
