'''
Decision Making based on Multiple Evidences

Author: Minsu Jang (minsu@etri.re.kr)
'''

import numpy as np
import rospy

class DecisionByEvidence(object):
    '''
    Decision Maker based on the frequency of evidences
    '''
    def __init__(self, category_name, min_num_evidences=30):
        '''
        Arguments
        ---------
        min_num_evidences: minimum number of evideces necessary for making a decision
        '''
        self.category_name = category_name
        self.evidences = {}
        self.num_evidences = {}
        self.major_evidences = {}
        self.min_num_evidences = min_num_evidences

    def add_evidence(self, subject, evidence, confidence=1.):
        '''
        Add an evidence about a subject with an optional confidence value.
        '''
        if not self.evidences.has_key(subject):
            self.evidences[subject] = {}
            self.num_evidences[subject] = 0
            self.major_evidences[subject] = None

        if not self.evidences[subject].has_key(evidence):
            self.evidences[subject][evidence] = []

        self.evidences[subject][evidence].append(confidence)
        self.num_evidences[subject] += 1
        rospy.logdebug('DECISION_MAKER(%d:%s) NUM_EVIDENCES=%d', subject, self.category_name, self.num_evidences[subject])
        if self.major_evidences[subject] is None:
            self.major_evidences[subject] = evidence
        major_evidence = self.major_evidences[subject]
        if len(self.evidences[subject][evidence]) > len(self.evidences[subject][major_evidence]):
            self.major_evidences[subject] = evidence

        return self.make_decision(subject)

    def make_decision(self, subject):
        '''
        Make a decision by the frequency of evidences.

        Arguments
        ---------
        subject: the subject
        '''
        if self.major_evidences.has_key(subject) and self.num_evidences[subject] > self.min_num_evidences:
            decision = self.major_evidences[subject]
            num_occurrences = len(self.evidences[subject][decision])
            confidence = np.average(np.array(self.evidences[subject][decision]))
            return decision, num_occurrences, confidence
        else:
            return None, None, None

    def get_min_num_evidence(self):
        '''
        Get the minimum number of evidences to make a decision.
        '''
        return self.min_num_evidences
