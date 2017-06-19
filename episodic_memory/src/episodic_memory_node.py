#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
Episodic Memory

Author: Minsu Jang (minsu@etri.re.kr)
'''

import time
import json
import os
from os.path import expanduser
from shutil import copy
from random import randint
import rospy
import rospkg
from std_msgs.msg import String
from perception_common.msg import PersonPresenceState
from perception_common.msg import MSetInfo
from perception_msgs.msg import PersonIdentity
from knowledge_graph import KnowledgeGraph
from perception_base.perception_base import PerceptionBase


def list_files(root_dir, suffix=None):
    """
    Walks a directory recursively and returns a list of files.
    """
    files = []
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        for fname in file_list:
            if fname[0] == '.':
                continue
            if suffix != None and not fname.endswith(suffix):
                continue
            file_path = dir_name + "/" + fname
            files.append(file_path)
    return files


class EpisodicMemoryNode(PerceptionBase):
    '''
    Episodic Memory Node
    '''
    def __init__(self):
        super(EpisodicMemoryNode, self).__init__("episodic_memory")

        rospy.Subscriber('/mhri/person_presence_state', PersonPresenceState, self.pps_callback)
        rospy.Subscriber('/mhri/person_identity_state', PersonIdentity, self.pi_callback)
        rospy.Subscriber('/mhri/mset', MSetInfo, self.mset_callback)

        self.graph = KnowledgeGraph()

        #self.pub = rospy.Publisher("/mhri/events", PerceptionEvent, queue_size=1)

        self.persons = {}
        self.episodes = {}

        self.load_episodic_memory()

        rospack = rospkg.RosPack()
        perception_face_pkg_path = rospack.get_path('perception_face')
        self.face_registry_path = perception_face_pkg_path + '/data/face_registry'

        self.names_file_path = perception_face_pkg_path + '/data/name_info.json'
        self.load_people_names(self.names_file_path)

        self.face_memorized_pub = rospy.Publisher('/mhri/face_memorized', String, queue_size=1)

        rospy.loginfo("episodic_memory_node initialized.")


    def load_people_names(self, names_file_path):
        rospy.loginfo('episodic_memory - Names File: %s', names_file_path)
        self.names = json.load(open(names_file_path))


    def save_person_name(self, person_id, name):
        if person_id not in self.names:
            self.names[person_id] = name
        with open(self.names_file_path, 'w') as names_file:
            json.dump(self.names, names_file)


    def load_episodic_memory(self):
        '''
        Load all the memory in the episodic memory.
        '''
        query = 'match (x:Episode) return x.person_id, x.name, x.confidence, x.eyeglasses,\
                 x.cloth_color, x.hair_length, x.start_time, x.end_time, x.gender'
        categories = ['name', 'confidence', 'eyeglasses', 'cloth_color',
                      'hair_length', 'start_time', 'end_time', 'gender']
        result = self.graph.Run(query)
        records = []
        for record in result:
            records.append(record)
        rospy.loginfo("EPISODIC_MEMORY: Total No of episodes = %d", len(records))
        persons = {}
        for record in records:
            person_id = record['x.person_id']
            if person_id not in persons:
                persons[person_id] = {}
            for category in categories:
                if category not in persons[person_id]:
                    persons[person_id][category] = {}
                value = record['x.' + category]
                if value not in persons[person_id][category]:
                    persons[person_id][category][value] = 0
                persons[person_id][category][value] += 1
        for person_id in persons:
            for category in categories:
                max_num = 0
                max_val = ''
                values = persons[person_id][category]
                for key in values:
                    rospy.loginfo('EPISODES(%s): %s is %s = %d', person_id, category, key, values[key])
                    if key == '':
                        continue
                    if max_val == '' or values[key] >= max_num:
                        max_num = values[key]
                        max_val = key
                persons[person_id][category] = max_val
                rospy.loginfo('EPISODES(%s): Decision for %s is %s', person_id, category, max_val)
                if category == 'name':
                    rospy.set_param('/{0}/{1}'.format(str(person_id), str(category)), max_val) # 170118
                    rospy.loginfo('/%s/%s = %s', str(person_id), str(category), max_val)
        self.episodes = persons
        rospy.loginfo("EPISODIC_MEMORY: Memory of %d people recalled.", len(self.episodes.keys()))


    def person_appeared(self, trk_id):
        '''
        새로운 사람의 등장!
        '''
        if trk_id not in self.persons:
            self.persons[trk_id] = {}
            self.persons[trk_id]['start_time'] = time.time()


    def person_disappeared(self, trk_id):
        '''
        사람 사라짐!
        '''
        if trk_id in self.persons:
            self.persons[trk_id]['end_time'] = time.time()
            self.memorize_episode(trk_id)
        else:
            rospy.logerr("I DO NOT KNOW A PERSON WITH AN ID=%s", trk_id)


    def memorize_episode(self, trk_id):
        '''
        Store an episode with a person(trk_id).
        '''
        if trk_id not in self.persons:
            rospy.logerr("I DO NOT KNOW A PERSON WITH AN ID=%s", trk_id)
            return

        episode = self.persons[trk_id]

        # if the person has not been identified, no memory lives on.
        # if an episode ended in 10 seconds, no memory lives on.
        duration = episode['end_time'] - episode['start_time']
        if not episode.has_key('person_id') or duration < 10:
            rospy.loginfo('EPISODIC_MEMORY: No Memory with %s', trk_id)
            return

        rospy.loginfo("MEMORIZING EPISODE (%s):======\n%s\n=======", trk_id, episode)
        self.graph.Create('Episode', 'human_id', trk_id)
        self.graph.Set('Episode', 'human_id', trk_id, 'person_id', episode['person_id'])
        self.graph.Set('Episode', 'human_id', trk_id, 'confidence', episode.get('confidence', 0))
        self.graph.Set('Episode', 'human_id', trk_id, 'name', episode.get('name', ''))
        self.graph.Set('Episode', 'human_id', trk_id, 'gender', episode.get('gender', 0))
        self.graph.Set('Episode', 'human_id', trk_id, 'eyeglasses', episode.get('eyeglasses', 0))
        self.graph.Set('Episode', 'human_id', trk_id, 'cloth_color', episode.get('cloth_color', ''))
        self.graph.Set('Episode', 'human_id', trk_id, 'hair_length', episode.get('hair_length', ''))
        self.graph.Set('Episode', 'human_id', trk_id, 'start_time', episode.get('start_time', 0))
        self.graph.Set('Episode', 'human_id', trk_id, 'end_time', episode.get('end_time', 0))

        # 초면인 경우 얼굴을 기억한다.
        if trk_id == episode['person_id']:
            self.memorize_face(trk_id)
            self.face_memorized_pub.publish("A")
            rospy.set_param('/{0}/{1}'.format(trk_id, 'name'), episode.get('name', '')) # 170214

    def memorize_face(self, trk_id):
        '''
        얼굴을 기억한다.
        '''
        files = []
        home_dir = expanduser("~") + '/.ros'
        img_file_names = list_files(home_dir, '.jpg')
        for file_name in img_file_names:
            pid = file_name.split('/')[-1].split('_')[-1].split('.')[0]
            if pid == trk_id:
                files.append(file_name)
        rospy.loginfo(">>>>FILE NO: %s = %d", trk_id, len(files))
        if len(files) == 0:
            return
        elif len(files) < 3:
            count = len(files)
        else:
            count = 3
        for num in range(count):
            i = randint(0, len(files)-1)
            file_name = files[i]
            rospy.loginfo('>>>>FILE: %s', file_name)
            copy(file_name, self.face_registry_path)
            files.remove(file_name)


    def identity_recognized(self, trk_id, person_id):
        '''
        얼굴이 인식되었을 때 호출되는 콜백
        '''
        if trk_id not in self.persons:
            rospy.logerr("I DO NOT KNOW A PERSON WITH AN ID=%s", trk_id)
        else:
            if person_id == 'unknown':
                person_id = trk_id
            rospy.loginfo('>>> EPISODIC_MEMORY_NODE: person_id=%s', person_id)
            self.persons[trk_id]["person_id"] = person_id
            # 기억을 기반으로 소셜 이벤트 생성
            if person_id in self.episodes:
                self.write_memory_recall_event()


    def	write_memory_recall_event(self):
        '''
        Generate a social event by recalling episodic memory.
        '''
        wr_data = self.get_memory_data()

        rospy.loginfo("EPISODIC MEMORY_RECALLED: %s", wr_data)

        self.save_to_memory(self.conf_data.keys()[0], data=wr_data)
        self.raise_event(self.conf_data.keys()[0], 'memory_recalled')


    def get_memory_data(self):
        '''
        Convert retrieved memory into a JSON document.
        '''
        data = json.loads('{}')
        data['human_id'] = []
        data['person_id'] = []
        data['name'] = []
        data['gender'] = []
        data['eyeglasses'] = []
        data['cloth_color'] = []
        data['hair_length'] = []
        data['confidence'] = []
        for trk_id in self.persons:
            if 'person_id' not in self.persons[trk_id]:
                continue
            person_id = self.persons[trk_id]['person_id']
            if person_id in self.episodes:
                memory = self.episodes[person_id]
                data['human_id'].append(trk_id)
                data['person_id'].append(person_id)
                data['name'].append(memory['name'])
                data['gender'].append(memory['gender'])
                data['eyeglasses'].append(memory['eyeglasses'])
                data['cloth_color'].append(memory['cloth_color'])
                data['hair_length'].append(memory['hair_length'])
                data['confidence'].append(memory['confidence'])
        return json.dumps(data)


    def pi_callback(self, pi_msg):
        '''
        pi_msg structure:
            string human_id
            string category
            string value
            int i_value
            float f_value
        '''
        if pi_msg.human_id not in self.persons:
            rospy.logerr("I DO NOT KNOW A PERSON WITH AN ID=%s", pi_msg.human_id)
            return
        rospy.logdebug("EPISODIC_MEMORY: PI MSG = human_id=%s category=%s value=%s i_value=%i f_value=%f",
                      pi_msg.human_id, pi_msg.category, pi_msg.value, pi_msg.i_value, pi_msg.f_value)
        if pi_msg.category == 'person_id':
            rospy.loginfo('EPISODIC_MEMORY_NODE: person id notified: %s', pi_msg.value)
            self.identity_recognized(pi_msg.human_id, pi_msg.value)
        else:
            if pi_msg.i_value != 0:
                self.persons[pi_msg.human_id][pi_msg.category] = pi_msg.i_value
            elif pi_msg.f_value != 0:
                self.persons[pi_msg.human_id][pi_msg.category] = pi_msg.f_value
            else:
                self.persons[pi_msg.human_id][pi_msg.category] = pi_msg.value
            '''
            else:
                rospy.logerr("EPISODIC_MEMORY: Invalid Value = %s %s %s",
                             pi_msg.value, pi_msg.i_value, pi_msg.f_value)
            '''


    def pps_callback(self, pps_msg):
        '''
        pps_msg 구조
        ------------
            Header header
            int16 count
            string[] appeared
            string[] disappeared
        '''
        for human_id in pps_msg.appeared:
            self.person_appeared(human_id)

        for human_id in pps_msg.disappeared:
            self.person_disappeared(human_id)


    def mset_callback(self, mset_msg):
        '''
        mset_msg 구조
        -------------
            string Key
            string value
        '''
        rospy.loginfo("mset_callback called: %s %s", mset_msg.key, mset_msg.value)
        keys = mset_msg.key.split('/')
        tag = keys[0]
        human_id = keys[1]

        # TODO: 형태소 분석기 활용에 적정 장소
        if tag == 'name':   # 사용자의 이름 정보
            if human_id in self.persons and self.persons[human_id].has_key('person_id'):
                rospy.loginfo("mset_callback: name is going to be updated...")
                self.persons[human_id]['name'] = mset_msg.value
                person_id = self.persons[human_id]['person_id']
                self.save_person_name(person_id, mset_msg.value)
                self.update_memory_on_name(person_id, mset_msg.value)
            else:
                rospy.loginfo("mset_callback: name is not updated...")

    def update_memory_on_name(self, person_id, name):
        self.graph.Set('Episode', 'person_id', person_id, 'name', name)


def main():
    '''
    Main runner.
    '''
    # rospy.init_node('episodic_memory_node', anonymous=False)

    EpisodicMemoryNode()

    rospy.spin()


if __name__ == '__main__':
    main()
