'''
Smile Recognizer using a CNN

* This class utilizes SmileCNN project @ https://github.com/kylemcdonald/SmileCNN).

Author: Minsu Jang (minsu@etri.re.kr)
'''

import logging
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D



class SmileRecognizer(object):
    '''
    Smile Recognizer using a CNN
    '''
    def __init__(self, model_weight_file_path):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(">>> LOADING FROM: {}".format(model_weight_file_path))
        #self.model = model_from_json(open(model_description_file_path).read())
        self.model = self.create_model()
        self.model.load_weights(model_weight_file_path)
        self.model.summary()

        self.class_names = ['Neutral', 'Smiling']


    def predict(self, image):
        '''
        Classify input image and returns the class-wise prediction probabilities.
        '''
        return self.model.predict(np.array([image]))[0]


    def create_model(self):
        '''
        Create a CNN model.
        '''
        img_rows, img_cols = 32, 32
        nb_filters = 32
        nb_pool = 2
        nb_conv = 3
        nb_classes = 2

        model = Sequential()

        model.add(Reshape((1, img_rows, img_cols), input_shape=(img_rows, img_cols)))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
