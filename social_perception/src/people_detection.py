#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
Authored by Minsu Jang (minsu@etri.re.kr)
'''
import rospy
import json
import time
import numpy
from mhri_common.msg import *
from std_msgs.msg import Empty
from perception_base.perception_base import PerceptionBase


class PeopleDetection(PerceptionBase):
	# Initialization
	def __init__(self):
		super(PeopleDetection, self).__init__("people_detection")
		self.no_of_people_ = 0
		rospy.Subscriber(
			"/mhri/social_perception_core/beliefs/persons", PersonPerceptArray, self.handle)

	def ConvertPeopleData2Json(self, difference, no_of_people):
		data = json.loads('{}')
		data['difference'] = difference
		data['no_of_people'] = no_of_people
		return json.dumps(data)

	def NotifyNoOfGuestsChanged(self, difference, no_of_people):
		wr_data = self.ConvertPeopleData2Json(difference, no_of_people)
		
		rospy.loginfo("Event: %s", wr_data)

		self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
		self.raise_event(self.conf_data.keys()[0], 'number_of_guests_changed')


	def handle(self, msg):
		no_of_people = len(msg.person_percepts)
		rospy.logdebug("No_of_People = %d", no_of_people)

		if no_of_people != self.no_of_people_:
			# Publish an event
			self.NotifyNoOfGuestsChanged(no_of_people-self.no_of_people_, no_of_people)
			# Update the number of people detected
			self.no_of_people_ = no_of_people

if __name__ == '__main__':
	rospy.init_node('people_detection', anonymous=False)
	m = PeopleDetection()
	rospy.spin()
