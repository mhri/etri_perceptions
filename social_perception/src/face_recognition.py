#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
Author: Minsu Jang (minsu@etri.re.kr)
'''

import json
import time
import codecs
import rospy
import rospkg
from perception_msgs.msg import PersonPerceptArray
from perception_msgs.msg import PerceptionEvent
from perception_msgs.msg import PersonIdentity
from DecisionByEvidence import DecisionByEvidence
from perception_base.perception_base import PerceptionBase


def load_jsonfile(fname):
    '''
    Load a json file.
    '''
    if fname=='-':
        fp = codecs.getreader('utf-8')(sys.stdin)
    else:
        fp = codecs.open(fname, 'rb', encoding='utf-8')
    lines  = fp.read()
    fp.close()
    jdata = json.loads(lines)
    return jdata


class IdentityRecognition(PerceptionBase):
    '''
    Identity Recognition
    '''
    def __init__(self):
        super(IdentityRecognition, self).__init__("face_recognition")

        rospy.Subscriber("People_With_Face", PersonPerceptArray, self.handle)

        self.num_evidences_to_watch = 30

        self.categories = ['face_recognition', 'cloth_color', 'hair_length', 'eyeglasses', 'gender']

        self.decision_makers = {}
        self.beliefs = {}
        for category in self.categories:
            self.decision_makers[category] = DecisionByEvidence(category)
            self.beliefs[category] = {}

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('social_perception')
        names_file_path = pkg_path + "/config/names.json"
        self.names = load_jsonfile(names_file_path)

        # 인식이 완료되었으니 인식기 그만 돌리라는 이벤트 배포
        self.pub = rospy.Publisher("/mhri/events", PerceptionEvent, queue_size=1)
        # 인식 내용을 에피소드로 기억하기 위한 이벤트 배포
        self.pi_pub = rospy.Publisher("/mhri/person_identity_state", PersonIdentity, queue_size=1)


    def notify_event(self, event_id, data):
        '''
        Notify belief update completion event.
        '''
        event = PerceptionEvent()
        event.event_id = event_id
        event.data = data
        self.pub.publish(event)


    def update_belief(self, subject_id, category, cur_value, invalid_value, event_id):
        '''
        Update Beliefs
        '''
        rospy.logdebug("UPDATE_BELIEF: %d=%s=%s=%s=%d",
                       subject_id, category, str(cur_value), str(invalid_value), event_id)
        if cur_value == invalid_value:
            rospy.logdebug('   SKIPPING....')
            return
        if self.beliefs[category].has_key(subject_id) is True:
            rospy.logdebug('   SKIPPING....')
            return

        rospy.logdebug("UPDATE_BELIEF: %d=%s=%s=%s=%d",
                       subject_id, category, str(cur_value), str(invalid_value), event_id)

        decision_maker = self.decision_makers[category]
        decision, num_votes, confidence = decision_maker.add_evidence(
            subject_id, cur_value)
        if decision != None:
            rospy.loginfo("FINISHED %s (person=%s) = %s with %d votes and confidence %f",
                          category, subject_id, decision, num_votes, confidence)
            # Belief Updated!!
            self.beliefs[category][subject_id] = (decision, num_votes, confidence)

            # Notify Episodic Memoty...
            identity = PersonIdentity()
            identity.human_id = subject_id
            identity.category = category
            if isinstance(decision, int):
                identity.i_value = decision
            elif isinstance(decision, float):
                identity.f_value = decision
            else:
                identity.value = str(decision)
            self.pi_pub.publish(identity)

            # Notify that cloth color detection has been completed
            self.notify_event(event_id, subject_id)


    def update_identity_states(self, percepts):
        '''
        Update probabilistic centainties of the identity of tracked people
        '''
        face_ids = []
        person_ids = []
        names = []
        confidences = []

        for percept in percepts:
            if percept.face_detected == 0:
                continue

            cur_trk_id = percept.trk_id
            cur_person_id = percept.person_id
            cur_person_confidence = percept.person_confidence
            cur_cloth_color = percept.cloth_color
            cur_hair_length = percept.hair_length
            cur_gender = percept.gender
            cur_eyeglasses = percept.eye_glasses

            # Face Recognition
            belief = self.beliefs['face_recognition']
            if cur_person_id != '' and belief.has_key(cur_trk_id) is False:
                decision_maker = self.decision_makers['face_recognition']
                rospy.logdebug("ADDING EVIDENCE: %d %s %f",
                               cur_trk_id, cur_person_id, cur_person_confidence)
                decision, num_votes, confidence = decision_maker.add_evidence(
                    cur_trk_id, cur_person_id, cur_person_confidence)
                if decision != None:
                    name = self.names.get(decision, '').encode('utf-8')

                    # Identification Finished!!
                    belief[cur_trk_id] = (decision, num_votes, confidence)

                    # Notify Episodic Memory...
                    # person_id
                    identity = PersonIdentity()
                    identity.human_id = cur_trk_id
                    identity.category = 'person_id'
                    identity.value = decision
                    rospy.loginfo('>>> PersonID = %s', decision)
                    self.pi_pub.publish(identity)
                    time.sleep(0.001)
                    rospy.loginfo("****publishing: %s", identity)
                    # name
                    identity = PersonIdentity()
                    identity.human_id = cur_trk_id
                    identity.category = 'name'
                    identity.value = self.names.get(decision, '').encode('utf-8')
                    self.pi_pub.publish(identity)
                    time.sleep(0.001)
                    #rospy.loginfo("****publishing: %s", identity)
                    # confidence
                    identity = PersonIdentity()
                    identity.human_id = cur_trk_id
                    identity.category = 'confidence'
                    identity.f_value = confidence
                    self.pi_pub.publish(identity)
                    time.sleep(0.001)
                    #rospy.loginfo("****publishing: %s", identity)

                    # Notify face recognizer to stop recognizing current person
                    self.notify_event(1, cur_trk_id)

                    face_ids.append(cur_trk_id)
                    person_ids.append(decision)
                    #name = unicode(self.names.get(decision, ''), 'utf-8')
                    name = self.names.get(decision, '')
                    query = '/' + decision + '/name'
                    name = rospy.get_param(query, 'undefined')
                    rospy.loginfo('GET NAME: %s = %s.', query, name)
                    rospy.loginfo('NAME TYPE = %s\n', type(name))
                    names.append(name)
                    confidences.append(confidence)

            self.update_belief(cur_trk_id, 'cloth_color', cur_cloth_color, '', 2)
            self.update_belief(cur_trk_id, 'hair_length', cur_hair_length, '', 3)
            self.update_belief(cur_trk_id, 'eyeglasses', cur_eyeglasses, 0, 4)
            self.update_belief(cur_trk_id, 'gender', cur_gender, 0, 5)

        if len(face_ids) > 0:
            self.write_face_recognition_event(face_ids, person_ids, names, confidences)


    def	write_face_recognition_event(self, face_ids, person_ids, names, confidences):
        if len(face_ids) <= 0 or len(person_ids) <= 0 or len(names) <= 0 or len(confidences) <= 0:
            return

        wr_data = self.get_identification_data(face_ids, person_ids, names, confidences)

        rospy.loginfo('FACE_RECOGNIZED EVENT: %s\n', wr_data)

        self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
        self.raise_event(self.conf_data.keys()[0], 'face_recognized')


    def get_identification_data(self, face_ids, person_ids, names, confidences):
        data = json.loads('{}')
        data['count'] = len(face_ids)
        data['face_id'] = face_ids
        data['person_id'] = person_ids
        data['name'] = names
        data['confidence'] = confidences
        return json.dumps(data)


    def GetGenderString(self, gender):
        if gender == 1:
            return 'male'
        elif gender == 2:
            return 'female'
        else:
            return 'unknown'


    def GetEyeglassesString(self, eye_glasses):
        if eye_glasses == 0:
            return 'unknown'
        elif eye_glasses == 1:
            return 'True'
        elif eye_glasses == 2:
            return 'False'


    def percepts_to_json(self, percepts):
        '''
        Convert percept data into json format.
        '''
        data = json.loads('{}')
        data["count"] = len(percepts)
        data["name"] = []
        data["person_id"] = []
        data["confidence"] = []
        data["face_pos"] = []
        data["gender"] = []
        data["cloth_color"] = []
        data["eyeglasses"] = []
        data["height"] = []
        data["hair_style"] = []
        data["session_face_id"] = []
        data["emotion"] = []
        for percept in percepts:
            cur_trk_id = percept.trk_id
            person_id = percept.person_id
            name = self.names[person_id] if self.names.has_key(person_id) else person_id
            cloth_color = percept.cloth_color
            hair_length = percept.hair_length
            gender = percept.gender
            eye_glasses = percept.eye_glasses
            confidence = percept.person_confidence
            if cur_trk_id in self.beliefs['face_recognition']:
                person_id = self.beliefs['face_recognition'][cur_trk_id][0]
                confidence = self.beliefs['face_recognition'][cur_trk_id][2]
                name = self.names[person_id] if self.names.has_key(person_id) else person_id
            if cur_trk_id in self.beliefs['cloth_color']:
                cloth_color = self.beliefs['cloth_color'][cur_trk_id][0]
            if cur_trk_id in self.beliefs['hair_length']:
                hair_length = self.beliefs['hair_length'][cur_trk_id][0]
            if cur_trk_id in self.beliefs['gender']:
                gender = self.beliefs['gender'][cur_trk_id][0]
            if cur_trk_id in self.beliefs['eyeglasses']:
                eye_glasses = self.beliefs['eyeglasses'][cur_trk_id][0]
            data["person_id"].append(person_id)
            data["name"].append(name)
            data["gender"].append(self.GetGenderString(gender))
            data["cloth_color"].append(cloth_color)
            data["eyeglasses"].append(self.GetEyeglassesString(eye_glasses))
            data["hair_style"].append(hair_length)
            data["confidence"].append(confidence)
            face_pos = []
            face_pos.append(percept.face_pos3d.x)
            face_pos.append(percept.face_pos3d.y)
            face_pos.append(percept.face_pos3d.z)
            face_pos.append(percept.frame_id)
            data["face_pos"].append(face_pos)
            data["height"].append(0)
            data["session_face_id"].append(percept.trk_id)
            if percept.emotion == 1:
                data["emotion"].append("happy")
            else:
                data["emotion"].append("unhappy")
            return json.dumps(data)


    def write_identification_event(self, percepts):
        '''
        Write identification information to the social memory.
        '''
        wr_data = self.conf_data['person_identification']['data']
        wr_data = self.percepts_to_json(percepts)

        self.save_to_memory(self.conf_data.keys()[0], data=wr_data)



    def handle(self, msg):
        '''
        Message handler
        '''
        try:
            if len(msg.person_percepts) > 0:
                self.update_identity_states(msg.person_percepts)
                self.write_identification_event(msg.person_percepts)
        except rospy.ServiceException, exception:
            rospy.logerr("Service call failed: %s", exception)


def main():
    '''
    Main runner.
    '''
    rospy.init_node('face_recognition', anonymous=False)

    IdentityRecognition()

    rospy.spin()


if __name__ == '__main__':
    main()
