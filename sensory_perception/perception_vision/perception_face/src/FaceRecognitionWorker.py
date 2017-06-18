'''
A threaded worker for face recognition

Author: Minsu Jang (minsu@etri.re.kr)
'''

import rospy
import time
from FaceRecognizer import FaceRecognizer
from QueuedProcessing import Worker
from magma_util import get_key_with_minval, get_key_with_maxval
from magma_util import ImageWriter


class FaceRecognitionWorker(Worker):
	'''
	A Threaded Worker for Face Recognition
	'''
	def __init__(self, model_path, registry_path, data_queue, result_queue):
		Worker.__init__(self)
		self.set_queues(data_queue, result_queue)
		self.face_recognizer = FaceRecognizer(model_path, registry_path)

		self.image_writer = ImageWriter()
		self.image_writer.start()

	def load_faces(self):
		self.face_recognizer.load_registered_faces()
		
	def recognize(self, img, landmarks):
		landmark_points = []
		for i in range(len(landmarks)/2):
			landmark_points.append((landmarks[i*2],landmarks[i*2+1]))
		matching_scores, image_to_save = self.face_recognizer.recognize(img, landmark_points)
		return matching_scores, image_to_save

	def work(self, faces):
		results = {}
		for face in faces:
			person_id = face[0]
			img = face[1]
			landmarks = face[2]
			matching_scores, image_to_save = self.recognize(img, landmarks)
			
			rospy.logdebug('FaceRecognition: {}'.format(matching_scores))
			if image_to_save is not None:
				rospy.logdebug('FaceRecognitionWorker: Saving an image...')
				self.image_writer.put(image_to_save, str(time.time()) + '_' + str(person_id) + '.jpg')
			else:
				rospy.logdebug('FaceRecognitionWorker: No image to save...')
				
			if matching_scores is not None:
				results[person_id] = (get_key_with_maxval(matching_scores), matching_scores)
			else:
				results[person_id] = None
		return results

