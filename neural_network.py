import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import shutil
import pathlib

from tensorflow.python import keras
from tensorflow.contrib import saved_model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, add, Input, MaxPool2D


# Keras Resnet: BGR
# Keras Inception: RGB

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


def resnet_block(input, filters, kernel_size, stride=1, padding='same'):
    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, activation='relu')(input)
    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, activation='relu')(y)

    return add([input, y])


class resnet:
    def __init__(self):
        input = Input(shape=(224, 224, 3))

        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(input)

        for i in range(2):
            if i == 0:
                x = Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(x)

            x = resnet_block(x, filters=64, kernel_size=3)

        self.model = keras.Model(inputs=input, outputs=x)

        self.model.summary()


class expandingNetwork:
    def __init__(self, number_of_class, model_type='inception'):
        try:
            self.number_of_class = int(number_of_class)
        except TypeError:
            self.number_of_class = 1

        savemodel_path = pathlib.Path('savemodel')

        if savemodel_path.exists():
            self.classifier_model = saved_model.load_keras_model(saved_model_path='savemodel')
            print('Savedmodel loaded!')

            output_of_dense = self.classifier_model.output_shape[1]

            if output_of_dense is not self.number_of_class:
                output = Dense(
                    units=self.number_of_class,
                    activation='softmax',
                    name='final_output_layer'
                )(self.classifier_model.get_layer(name='global_average_pooling2d').output)

                output_layer = self.classifier_model.layers.pop()

                output_variable = output_layer.get_weights()

                self.classifier_model = keras.Model(inputs=self.classifier_model.input, outputs=output)

                random_weights = np.random.uniform(.005, .006, size=(output_variable[0].shape[0], 1))
                random_bias = np.random.uniform(.005, .006, size=1)

                output_weights = np.concatenate((output_variable[0], random_weights), axis=1)
                output_bias = np.concatenate((output_variable[1], random_bias))

                local_array = self.classifier_model.layers.pop().get_weights()

                self.classifier_model.layers.pop().set_weights([output_weights, output_bias])
        else:
            if model_type == 'inception':
                self.classifier_model = self.inception()
            elif model_type == 'resnet':
                self.classifier_model = self.resnet()
            else:
                self.classifier_model = None
            print('Created new model!')

        self.classifier_model.summary()

    def resnet(self):
        resnet_model: keras.Model = keras.applications.resnet50.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(200, 200, 3)
        )

        for layer in resnet_model.layers:
            layer.trainable = False

        last_output: keras.layers.Layer = GlobalAveragePooling2D()(resnet_model.output)

        last_output = Dense(
            units=self.number_of_class,
            activation='softmax',
            name='final_dense_layer'
        )(last_output)

        model = keras.Model(inputs=resnet_model.input, outputs=last_output)

        print('Resnet')

        return model

    def inception(self):
        inception_model: keras.Model = keras.applications.inception_v3.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(200, 200, 3)
        )

        for layer in inception_model.layers:
            layer.trainable = False

        last_output: keras.layers.Layer = GlobalAveragePooling2D()(inception_model.output)

        last_output = Dense(
            units=self.number_of_class,
            activation='softmax',
            name='final_dense_layer'
        )(last_output)

        model = keras.Model(inputs=inception_model.input, outputs=last_output)

        print('Inception')

        return model

    def train(self, dataset, steps_per_epoch, validation_data):
        self.classifier_model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.sgd(lr=1e-6, momentum=0.997),
            metrics=['accuracy']
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0, patience=10,
            verbose=1,
            mode='auto'
        )

        self.classifier_model.fit(
            x=dataset,
            epochs=128,
            steps_per_epoch=steps_per_epoch,
            # callbacks=[early_stopping],
            validation_data=validation_data
        )

    def infer(self, image, network_type='inception'):
        image = cv2.resize(src=image, dsize=(200, 200))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        if network_type == 'inception':
            # RGB
            image = image[..., ::-1]
            image /= 127.5
            image -= 1.
        elif network_type == 'resnet':
            # BGR
            mean = [103.939, 116.779, 123.68]

            image -= mean
        else:
            image /= 255.

        probablity = self.classifier_model.predict(x=image)

        return probablity

    def save_model(self):
        savemodel_path = pathlib.Path('savemodel')

        if savemodel_path.exists():
            shutil.rmtree('savemodel')

        saved_model.save_keras_model(model=self.classifier_model, saved_model_path='savemodel')

        print('Model saved!')
