#!/usr/bin/env python
#-*- encoding: utf8 -*-
'''
People Tracking Information Publisher.

This node accepts tracking information from OpenPTrack
and publishes messages delivering the information.

Author: Minsu Jang [minsu@etri.re.kr]
'''

import time

import rospy
import std_msgs.msg
import message_filters

from perception_msgs.msg import PersonPercept
from perception_msgs.msg import PersonPerceptArray
from perception_msgs.msg import PersonIDArray
from opt_msgs.msg import TrackArray
from opt_msgs.msg import IDArray


class OverwritingQueue(object):
    def __init__(self, size=5, initial_fill=True):
        self.queue = []
        if initial_fill is True:
            for i in range(size):
                self.queue.append(0)
        self.max_size = size

    def put(self, item):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get_all(self):
        return self.queue


class PeopleTracking(object):
    '''
    Input Topic: /tracker/tracks (type: TrackArray)
    Output Topic: /mhri/people_tracking/persons
    '''
    def __init__(self, max_no_track_count=10):
        self.pub = rospy.Publisher("/mhri/people_tracking/persons", PersonPerceptArray, queue_size=1)

        t_sub = rospy.Subscriber("/tracker/tracks", TrackArray, self.callback)
        i_sub = rospy.Subscriber("/tracker/alive_ids", IDArray, self.handle_ids)

        self.alive_ids = []
        self.prev_percepts = {}
        self.alive_count = {}

        self.max_no_track_count = max_no_track_count
        self.no_track_count = {}
        self.dead_ids = []

        self.ids = {}


    def handle_ids(self, idarray_msg):
        self.alive_ids = list(idarray_msg.ids)
        rospy.logdebug('TRACKING ALIVE IDS: %s', self.alive_ids)


    def retrieve_id(self, trk_id):
        if not self.ids.has_key(trk_id):
            self.ids[trk_id] = str(trk_id) + str(time.time()).split('.')[0]
        return self.ids[trk_id]

    def callback(self, tracks_msg):
        '''
        Callback for a tracking message.
        '''
        alive_ids_copy = list(self.alive_ids)
        alive_ids = []
        for alive_id in alive_ids_copy:
            if alive_id not in self.dead_ids:
                alive_ids.append(alive_id)
        rospy.logdebug('FILTERED ALIVE IDS: %s / DEAD IDS: %s', alive_ids, self.dead_ids)
        rospy.logdebug('  NO_TRACK_COUNT = %s', self.no_track_count)

        for alive_id in alive_ids:
            if alive_id not in self.alive_count:
                self.alive_count[alive_id] = 0
            self.alive_count[alive_id] += 1

        sorted_tracks = sorted(tracks_msg.tracks, key=lambda x: x.distance, reverse=True)

        valid_tracks = []
        for track in sorted_tracks:
            if track.id in self.alive_count and self.alive_count[track.id] > 5:
                valid_tracks.append(track)
                if track.id in self.no_track_count:
                    self.no_track_count[track.id] = 0

        percepts = PersonPerceptArray()
        percepts.header = std_msgs.msg.Header()
        percepts.header.stamp = rospy.Time.now()
        #tracks_msg.header.stamp

        if len(valid_tracks) > 0:
            for trk in valid_tracks:
                if trk.id in alive_ids:
                    alive_ids.remove(trk.id)
                percept = PersonPercept()
                percept.trk_id = self.retrieve_id(trk.id)
                percept.trk_pos_x = trk.x
                percept.trk_pos_y = trk.y
                percept.trk_pos_z = trk.distance
                percept.trk_height = trk.height
                percept.trk_confidence = trk.confidence
                percept.trk_age = trk.age
                percept.trk_bbox_x = trk.box_2D.x
                percept.trk_bbox_y = trk.box_2D.y
                percept.trk_bbox_width = trk.box_2D.width
                percept.trk_bbox_height = trk.box_2D.height
                rospy.logdebug("TRACK: id=%d, x=%f, y=%f, dist=%f, height=%f, confidence=%f, age=%f"%(trk.id, trk.x, trk.y, trk.distance, trk.height, trk.confidence, trk.age))
                rospy.logdebug("      BBox: x=%d, y=%d, width=%d, height=%d"%(trk.box_2D.x, trk.box_2D.y, trk.box_2D.width, trk.box_2D.height))
                percepts.person_percepts.append(percept)

        if len(alive_ids) > 0:
            for percept in self.prev_percepts:
                trk_id = int(percept.trk_id[0]) #!!!!!
                if self.no_track_count.has_key(trk_id) == False:
                    self.no_track_count[trk_id] = 0
                self.no_track_count[trk_id] += 1
                if self.no_track_count[trk_id] > self.max_no_track_count:
                    self.dead_ids.append(trk_id)
                    try:
                        alive_ids.remove(trk_id)
                    except Exception as e:
                        pass 
                    del self.alive_count[trk_id]
                    del self.no_track_count[trk_id]
                if trk_id in alive_ids:
                    percepts.person_percepts.append(percept)

        self.prev_percepts = list(percepts.person_percepts)

        #rospy.loginfo("TRACKING NUM Percepts = %d", len(percepts.person_percepts))
        self.pub.publish(percepts)


if __name__ == '__main__':
    rospy.init_node('perception_tracking_node', anonymous=False)

    PeopleTracking(40)

    rospy.spin()
