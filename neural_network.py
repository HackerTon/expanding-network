import tensorflow as tf
from tensorflow import keras


class expandingNetwork:
    def __init__(self, number_of_class):
        try:
            self.number_of_class = int(number_of_class)
        except TypeError:
            self.number_of_class = 1

        resnet_output = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                             pooling='avg', input_shape=(244, 244, 3))

        model_output = keras.layers.Dense(units=self.number_of_class, activation='softmax')(resnet_output.output)

        self.model = keras.Model(inputs=resnet_output.input, outputs=model_output)

        self.model.summary()

    def train(self, dataset):
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Nadam(),
                           metrics=[keras.metrics.mean_squared_error, keras.metrics.categorical_accuracy])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.model.fit(dataset, epochs=30,
                       steps_per_epoch=1, callbacks=[early_stop])
