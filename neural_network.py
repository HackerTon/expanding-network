import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib

from tensorflow.python import keras
from tensorflow.contrib import saved_model
from tensorflow.python.keras.layers import Dense, InputLayer

tf.enable_eager_execution()


class defaultNetwork:
    def __init__(self):
        self.model: keras.Model = keras.applications.resnet50.ResNet50(weights='imagenet')

        for layer in self.model.layers:
            layer.trainable = False

        # self.model.summary()

    def infer(self, img):
        img = cv2.resize(img, dsize=(224, 224))

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

        savemodel_path = pathlib.Path('savemodel')

        if savemodel_path.exists():
            self.classifier_model = saved_model.load_keras_model(saved_model_path='savemodel')
            print('Savedmodel loaded!')
        else:
            resnet_model: keras.Model = keras.applications.resnet50.ResNet50(
                include_top=False,
                weights='imagenet',
                pooling='avg',
                input_shape=(224, 224, 3)
            )

            for layer in resnet_model.layers:
                layer.trainable = False

            last_output = Dense(
                units=self.number_of_class,
                activation='softmax',
                name='final_dense_layer'
            )(resnet_model.output)

            self.classifier_model = keras.Model(inputs=resnet_model.input, outputs=last_output)
            print('Created new model!')

    def train(self, dataset, steps_per_epoch):
        self.classifier_model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.sgd(lr=1e-4, momentum=0.997),
            metrics=[keras.metrics.categorical_accuracy]
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='categorical_accuracy',
            min_delta=0, patience=10,
            verbose=1,
            mode='auto'
        )

        self.classifier_model.fit(
            x=dataset,
            epochs=30,
            steps_per_epoch=steps_per_epoch,
            callbacks=[early_stopping]
        )

    def infer(self, image):
        image = cv2.resize(
            src=image,
            dsize=(224, 224),
        )
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image /= 255.0

        probablity = self.classifier_model.predict(x=image, steps=1)

        return probablity

    def save_model(self):
        saved_model.save_keras_model(model=self.classifier_model, saved_model_path='savemodel')

