#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
MHRI Visualization

Author: Minsu Jang (minsu@etri.re.kr)
'''

import Queue
import rospy
import cv2
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from perception_msgs.msg import PersonPerceptArray
from perception_msgs.msg import PersonIdentity
from perception_common.msg import PersonPresenceState


class MhriVisualization:
    def __init__(self):
        self.bridge = CvBridge()

        self.count = 0
        self.writer = None
        self.recording = []
        self.in_recording = False
        self.img_size = (0, 0)
        self.frame_count = 0
        self.file_count = 0
        self.queue = Queue.Queue()
        self.recording_files = []
        self.saving = False

        image_sub = message_filters.Subscriber("/kinect2_head/sd/image_color_rect", Image)
        fd_sub = message_filters.Subscriber("People_Percepts", PersonPerceptArray)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, fd_sub], 10, 0.05)
        ts.registerCallback(self.callback)

        rospy.Subscriber('/mhri/person_presence_state', PersonPresenceState, self.pps_callback)
        rospy.Subscriber('/mhri/person_identity_state', PersonIdentity, self.pi_callback)

        self.persons = {}
        

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
        if pi_msg.category == 'person_id':
            self.persons[pi_msg.human_id]['person_id'] = pi_msg.value
        else:
            if pi_msg.i_value != 0:
                self.persons[pi_msg.human_id][pi_msg.category] = pi_msg.i_value
            elif pi_msg.f_value != 0:
                self.persons[pi_msg.human_id][pi_msg.category] = pi_msg.f_value
            else:
                self.persons[pi_msg.human_id][pi_msg.category] = pi_msg.value

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
            if human_id not in self.persons:
                self.persons[human_id] = {}

        for human_id in pps_msg.disappeared:
            if human_id in self.persons:
                del self.persons[human_id]


    def cbPresence(self, msg):
        #ospy.loginfo("RECEIVED COGNITIVE STATE = %s", msg.state)
        pass

    def callback(self, image, faces):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            #cv_image = cv2.flip(cv_image, 1)
            height, width = cv_image.shape[:2]
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.5
            if width > 1024:
                scale = 0.7
            fontColor = (0, 0, 0)
            thickness = 1
        except CvBridgeError, e:
            print e

        if self.img_size[0] == 0:
            self.img_size = (cv_image.shape[1], cv_image.shape[0])

        for percept in faces.person_percepts:
            human_x = percept.trk_bbox_x
            human_y = percept.trk_bbox_y
            human_width = percept.trk_bbox_width
            human_height = percept.trk_bbox_height
            cv2.rectangle(cv_image,
                          (human_x, human_y),
                          (human_x+human_width, human_y+human_height),
                          (0, 255, 0), 1)

            mouthOpenTxt = 'Mouth: ??'
            face_pose = 'Face Pose:??'
            emotion_str = 'Emotion: ??'

            if percept.face_detected == 1:
                landmarks = percept.stasm_landmarks

                # Face Outline
                for i in range(1, 17):
                    cv2.line(cv_image,
                             (landmarks[i*2], landmarks[i*2+1]),
                             (landmarks[(i-1)*2], landmarks[(i-1)*2+1]),
                             (255, 255, 0))

                # Nose Line
                self.drawLines(cv_image, landmarks, 0, 16)
                self.drawLines(cv_image, landmarks, 27, 30)
                self.drawLines(cv_image, landmarks, 17, 21)
                self.drawLines(cv_image, landmarks, 22, 26)
                self.drawLines(cv_image, landmarks, 30, 35)
                self.drawLines(cv_image, landmarks, 36, 41)
                self.drawLines(cv_image, landmarks, 42, 47)
                self.drawLines(cv_image, landmarks, 48, 59)
                self.drawLines(cv_image, landmarks, 60, 67)

                # ROI for distance estimation
                pcl = percept.face_pos3droi.x_offset
                pct = percept.face_pos3droi.y_offset
                pcr = pcl + percept.face_pos3droi.width
                pcb = pct + percept.face_pos3droi.height
                cv2.rectangle(cv_image, (pcl,pct), (pcr,pcb), (255,0,0), 2)

                # ROI for hair learning and detection
                fX1 = landmarks[1*2]
                fY1 = min(landmarks[19*2+1], landmarks[24*2+1])
                fX2 = landmarks[15*2]
                fY2 = landmarks[9*2+1]
                cv2.rectangle(cv_image, (fX1,fY1), (fX2,fY2), (255,255,0), 1)

                hX1 = max(1,fX1-(fX2-fX1)/3)
                hY1 = max(1,fY1-(int)((fY2-fY1)/1.5))
                hX2 = min(width-1,fX2+(fX2-fX1)/3)
                hY2 = min(height-1,fY2+(fY2-fY1)/4)
                cv2.rectangle(cv_image, (hX1,hY1), (hX2,hY2), (255,0,255), 1)

                #print "MOUTH: ", percept.mouth_opened
                if percept.mouth_opened == 1:
                    mouthOpenTxt = "Mouth: OPENED"
                else:
                    mouthOpenTxt = "Mouth: CLOSED"

                face_pose = "Face Pose: Front"
                fX1 = landmarks[1*2]
                fX2 = landmarks[15*2]
                fl = (landmarks[27*2]-fX1)/(float)(fX2-fX1)
                fr = (fX2-landmarks[27*2])/(float)(fX2-fX1)
                if (fr-fl) >= 0.3:
                    face_pose = "Face Pose: Left"
                elif (fr-fl) <= -0.4:
                    face_pose = "Face Pose: Right"

                if percept.emotion == 1:
                    emotion_str = "Smile"
                else:
                    emotion_str = "Neutral"

            overlay = cv_image.copy()
            # draw a rectangle for displaying sensory perceptions
            rx = max(0, (human_x + int(human_width / 2)) - 110)
            ry = max(0, (human_y + human_height) - 240)
            cv2.rectangle(overlay, (rx, ry), (rx+220, ry+240), (73, 241, 244), -1)

            opacity = 0.5
            cv2.addWeighted(overlay, opacity, cv_image, 1 - opacity, 0, cv_image)

            # display the ID
            cv2.putText(cv_image, "ID: " + str(percept.trk_id), (rx, ry+20),
                        font, scale, fontColor, thickness=thickness)

            # display the gender
            gender = percept.gender
            eye_glasses = percept.eye_glasses
            hair_length = percept.hair_length
            cloth_color = percept.cloth_color
            person_id = percept.person_id
            confidence = percept.person_confidence
            if self.persons.has_key(percept.trk_id):
                gender = self.persons[percept.trk_id].get('gender', gender)
                eye_glasses = self.persons[percept.trk_id].get('eyeglasses', eye_glasses)
                hair_length = self.persons[percept.trk_id].get('hair_length', hair_length)
                cloth_color = self.persons[percept.trk_id].get('cloth_color', cloth_color)
                person_id = self.persons[percept.trk_id].get('person_id', person_id)
                confidence = self.persons[percept.trk_id].get('confidence', confidence)

            cv2.putText(cv_image, "Person ID: " + person_id
                        + "(" + '{:4.3f}'.format(confidence) + ")",
                        (rx, ry+40), font, scale, fontColor, thickness=thickness)

            if gender == 0:
                genderText = "Gender: ??"
            elif gender == 1:
                genderText = "Gender: MALE"
            else:
                genderText = "Gender: FEMALE"
            cv2.putText(cv_image, genderText, (rx, ry+60), font,
                        scale, fontColor, thickness=thickness)

            if eye_glasses == 0:
                eyeGlassesText = "EyeGlasses: ??"
            elif eye_glasses == 1:
                eyeGlassesText = "EyeGlasses: YES"
            else:
                eyeGlassesText = "EyeGlasses: NO"
            cv2.putText(cv_image, eyeGlassesText, (rx, ry+80), font,
                        scale, fontColor, thickness=thickness)

            clothColorText = "ClothColor: " + cloth_color.upper()
            cv2.putText(cv_image, clothColorText, (rx, ry+100), font,
                        scale, fontColor, thickness=thickness)

            cv2.putText(cv_image, "Hair: " + hair_length.upper(),
                        (rx, ry+120), font, scale, fontColor, thickness=thickness)

            cv2.putText(cv_image, "Emotion: " + emotion_str, (rx, ry+140),
                        font, scale, fontColor, thickness=thickness)

            mind_status = percept.cognitive_status
            cv2.putText(cv_image, "Mind: " + mind_status.upper(),
                        (rx, ry+160), font, scale, fontColor, thickness=thickness)

            cv2.putText(cv_image, mouthOpenTxt, (rx, ry+180), font,
                        scale, fontColor, thickness=thickness)

            #print "Eye Size = ", percept.eye_size
            if percept.eye_size == 1:
                bigEyesText = "Eyes: BIG"
            else:
                bigEyesText = "Eyes: SMALL"
            cv2.putText(cv_image, bigEyesText, (rx, ry+200), font,
                        scale, fontColor, thickness=thickness)

            cv2.putText(cv_image, face_pose, (rx, ry+220), font,
                        scale, fontColor, thickness=thickness)

            cv2.putText(cv_image, "Distance: " + str(percept.trk_pos_z),
                        (rx, ry+240), font, scale, fontColor, thickness=thickness)

        cv2.imshow('mhri_viz', cv_image)
        cv2.waitKey(1)


    def drawLines(self, img, landmarks, startIndex, endIndex):
        for i in range(startIndex+1, endIndex+1):
            cv2.line(img,
                     (landmarks[i*2], landmarks[i*2+1]),
                     (landmarks[(i-1)*2], landmarks[(i-1)*2+1]),
                     (255, 255, 0))


if __name__ == '__main__':
    rospy.init_node('visualize', anonymous=False)
    m = MhriVisualization()
    rospy.spin()
