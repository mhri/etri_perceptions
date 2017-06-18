#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
Author: Minsu Jang (minsu@etri.re.kr)
'''

import time
import json
import rospy
from perception_msgs.msg import PersonPerceptArray
from DecayingPotential import DecayingPotential
from perception_base.perception_base import PerceptionBase


class EmotionalStateEstimation(object):
    '''
    Emotional State Estimator
    '''
    def __init__(self, subject_id):
        '''
        Arguments
        ---------
        subject_id: the owner of emotional states
        '''
        pass

    def put_evidence(self, evidence):
        pass

    def get_state(self):
        pass


class FacialExpressionDetection(PerceptionBase):
    '''
    Cognitive Decision Making based on Results from Sensory Facial Expression Recognition
    '''
    def __init__(self):
        super(FacialExpressionDetection, self).__init__("facial_expression_recognition")
        rospy.Subscriber("/mhri/social_perception_core/beliefs/persons",
                         PersonPerceptArray, self.handle)
        self.persons = {}


    def create_event_json(self, face_id, emotion_id, emotion_confidence):
        '''
        Create a JSON representation of a facial expression detection event
        '''
        data = json.loads('{}')
        data['face_id'] = face_id
        data['emotion_id'] = emotion_id
        data['emotion_confidence'] = emotion_confidence
        return json.dumps(data)


    def write_event(self, face_id, emotion_id, emotion_confidence):
        '''
        Generate an event by writing event data to the social memory.
        '''
        wr_data = self.create_event_json(face_id, emotion_id, emotion_confidence)

        rospy.loginfo("EMOTION_DETECTED: person id=%s emotion_id=%s confidence=%f",
                      face_id, emotion_id, emotion_confidence)

        self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
        self.raise_event(self.conf_data.keys()[0], 'facial_expression_detected')


    def binarize(self, potential):
        '''
        Convert the potential value into a binary decision.
        '''
        return 1 if potential > 0.6 else 0


    def handle(self, msg):
        '''
        Message handler
        '''
        try:
            percepts = msg.person_percepts

            # 미소가 검출된 사람들의 상태를 업데이트한다.
            for percept in percepts:
                if percept.face_detected == 0:
                    continue

                if percept.trk_id not in self.persons:
                    self.persons[percept.trk_id] = [DecayingPotential(0.5), time.time(), 0]
                    self.persons[percept.trk_id][0].start()

                state_vec = self.persons[percept.trk_id]

                if percept.emotion == 1: # smile/happy
                    state_vec[0].spike()

                curr_potential = state_vec[0].get_potential()
                curr_state = self.binarize(curr_potential)
                rospy.logdebug('Emotion: id=%s prev_state=%d, curr_potential=%3.2f, curr_state=%d',
                                  percept.trk_id, state_vec[2], curr_potential, curr_state)

                if state_vec[2] != curr_state:
                    rospy.logdebug('FACIAL_EXPRESSION_DETECTED will be generated...')
                    state_vec[2] = curr_state
                    self.write_event(percept.trk_id, curr_state, curr_potential if curr_state == 1 else (1-curr_potential))

                # 최종 관찰된 시점을 기록
                self.persons[percept.trk_id][1] = time.time()


            # 관찰된지 오래된 사람들은 삭제 (현재 기준 = 5초)
            to_be_deleted = []
            for key in self.persons:
                last_time_seen = self.persons[key][1]
                if (time.time() - last_time_seen) > 5.0:
                    to_be_deleted.append(key)

            for key in to_be_deleted:
                del self.persons[key]

        except rospy.ServiceException, exception:
            rospy.logerr("Service call failed: %s", exception)

    def stop(self):
        '''
        Stop all the decaying counters.
        '''
        for key in self.persons:
            self.persons[key][0].stop()


def main():
    '''
    Main runner.
    '''
    rospy.init_node('facial_expression_recognition', anonymous=False)

    fed = FacialExpressionDetection()

    rospy.spin()

    fed.stop()


if __name__ == '__main__':
    main()
