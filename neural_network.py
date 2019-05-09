from tensorflow.python import keras
from tensorflow.contrib import saved_model
from tensorflow.python.keras.preprocessing import image

import cv2 as cv
import tensorflow as tf

import numpy as np

import pandas as pd

tf.enable_eager_execution()


class defaultNetwork:
    def __init__(self):
        self.model: keras.Model = keras.applications.resnet50.ResNet50(weights='imagenet')

        for layer in self.model.layers:
            layer.trainable = False

        # self.model.summary()

    def infer(self, img):
        img = cv.resize(img, dsize=(224, 224))

        img = np.expand_dims(img, axis=0)

        img = keras.applications.resnet50.preprocess_input(img)

        preds = self.model.predict(img, steps=1)

        decoded = keras.applications.resnet50.decode_predictions(preds, top=3)[0]

        return decoded[0][1]

        # print('Predicted:', decoded)
        #
        # assert (0.75 < decoded[0][2] < 0.9), 'Model failed!'
        #
        # print('Model passed!')


class expandingNetwork:
    def __init__(self, number_of_class):
        try:
            self.number_of_class = int(number_of_class)
        except TypeError:
            self.number_of_class = 1

        resnet_output: keras.Model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                                          pooling='avg', input_shape=(244, 244, 3))

        for layer in resnet_output.layers:
            layer.trainable = False

        model_output = keras.layers.Dense(units=self.number_of_class, activation='softmax',
                                          name='final_dense_layer')(resnet_output.output)

        self.model = keras.Model(inputs=resnet_output.input, outputs=model_output)

        self.model.summary()

    def train(self, dataset):
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Nadam(),
                           metrics=[keras.metrics.mean_squared_error, keras.metrics.categorical_accuracy])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = self.model.fit(dataset, epochs=30,
                                 steps_per_epoch=1, callbacks=[early_stop])

    def save_model(self):
        saved_model.save_keras_model(model=self.model, saved_model_path='savemodel')
