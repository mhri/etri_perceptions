#!/usr/bin/env python
#-*- encoding: utf8 -*-
'''
Authored by Minsu Jang
'''
import rospy
import json
import time
import numpy as np
from perception_msgs.msg import *
from std_msgs.msg import Empty
from perception_base.perception_base import PerceptionBase

class SaliencyDetection(PerceptionBase):
	# Initialization
	def __init__(self):
		super(PersonIdentityProcessor, self).__init__("saliency_detection")
		self.saliencies = {}
		self.face_pos = {}
		self.saliency_values = {}
		self.top_salient_person = ''
		self.max_saliency_value = ''
		rospy.Subscriber(
			"/mhri/perception_core/3d_pos_estimation/persons", PersonPerceptArray, self.handle)
		self.last_time_changed = time.time()

	def ConvertSaliencyData2Json(self, face_id, saliency_value, face_index):
		data = json.loads('{}')
		data['face_id'] = face_id
		data['saliency_value'] = saliency_value
		data['face_index'] = face_index
		return json.dumps(data)

	def NotifySaliencyStatus(self, face_id, saliency_value, face_index):
		wr_data = self.ConvertSaliencyData2Json(face_id, saliency_value, face_index)

		rospy.logdebug("SOCIAL EVENT: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'saliency_changed')

	def CalcSaliency(self, pos1, pos2):
		val = (abs(pos1.x-pos2.x) + abs(pos1.y-pos2.y) + abs(pos1.z-pos2.z))
		if val > 100:
			print "face-pos:", pos1.x, ",", pos1.y, ",", pos1.z, " -- ", pos2.x, ",", pos2.y, ",", pos2.z
		return val

	def handle(self, msg):
		no_of_people = len(msg.person_percepts)
		rospy.logdebug("No_of_People = %d", no_of_people)

		percepts = msg.person_percepts

		max_saliency_value = -1
		top_salient_person = ''
		top_salient_person_index = -1
		for percept in percepts:
			face_id = percept.session_face_id
			if face_id not in self.saliencies:
				self.saliencies[face_id] = []
				self.saliencies[face_id].append(0)
				self.face_pos[face_id] = percept.face_pos3d
				self.saliency_values[face_id] = 0
			else:
				#print "prev_face_pos: ", self.face_pos[face_id].x, ",", self.face_pos[face_id].y, ",", self.face_pos[face_id].z
				#print "curr_face_pos: ", percept.face_pos3d.x, ",", percept.face_pos3d.y, ",", percept.face_pos3d.z
				saliency = self.CalcSaliency(self.face_pos[face_id],percept.face_pos3d)
				self.saliencies[face_id].append(saliency)
				if len(self.saliencies[face_id]) > 30:
					self.saliencies[face_id] = self.saliencies[face_id][1:]
				self.face_pos[face_id] = percept.face_pos3d
				self.saliency_values[face_id] = np.mean(self.saliencies[face_id])
				if self.saliency_values[face_id] > max_saliency_value:
					max_saliency_value = self.saliency_values[face_id]
					top_salient_person = face_id
					top_salient_person_index = percept.index
				#print "Saliency: ", face_id, " = ", self.saliency_values[face_id]

		if max_saliency_value > 0.02:
			#print "Saliency Bigger Than Threshold: ", self.saliency_values
			pass
		#if self.top_salient_person != top_salient_person and max_saliency_value > 0.02:
		if max_saliency_value > 0.02:
			if self.top_salient_person != top_salient_person or (time.time() - self.last_time_changed) > 10:
				#print "Saliency Changed: ", top_salient_person, ", value=", max_saliency_value
				self.NotifySaliencyStatus(top_salient_person, max_saliency_value, top_salient_person_index)
				self.top_salient_person = top_salient_person
				self.max_saliency_value = max_saliency_value
				self.last_time_changed = time.time()				
		else:
			#print "Salieny Not Changed."
			pass

		to_be_deleted = []
		for key in self.face_pos:
			found = False
			for percept in percepts:
				if key == percept.session_face_id:
					found = True
					break
			if not found:
				to_be_deleted.append(key)

		for key in to_be_deleted:
			del self.saliencies[key]
			del self.face_pos[key]
			del self.saliency_values[key]

if __name__ == '__main__':
	rospy.init_node('saliency_detection', anonymous=False)
	m = SaliencyDetection()
	rospy.spin()
