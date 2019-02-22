import tensorflow as tf
from tensorflow import keras

# Read image from a image data
def _read_image(image, value):
    imgString = tf.read_file(image)
    image = tf.image.decode_and_crop_jpeg(imgString, crop_window=[47, 80, 66, 200], channels=0)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image - tf.reduce_mean(input_tensor=image)
    return image, value

# Generate a tf.data.Dataset from csv
def _function_input2(filename, batch_size):
    dataset: tf.data.Dataset = tf.data.experimental.CsvDataset(filenames=filename,
                                                               record_defaults=[tf.string, tf.float32],
                                                               header=True)

    dataset = dataset.apply(tf.data.experimental.map_and_batch(_read_image,
                                                               batch_size=batch_size,
                                                               num_parallel_batches=1))

    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))

    return dataset

class expanding_network:
    def __init__(self, number_of_class):
        try:
            self.number_of_class = int(number_of_class)
        except TypeError:
            self.number_of_class = 1

        resnet_output = keras.applications.resnet50.ResNet50(include_top=False,
                                                             weights='imagenet',
                                                             pooling='avg',
                                                             input_shape=(244, 244, 3))

        model_output = keras.layers.Dense(units=self.number_of_class,
                                          activation='softmax')(resnet_output.output)

        self.model = keras.Model(inputs=resnet_output.input,
                                 outputs=model_output)

        self.model.summary()

    def preprocessing(self, folder):
        keras.applications.resnet50.preprocess_input()

    def train(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Nadam,
                           metrics=[keras.losses.mean_absolute_error, keras.losses.mean_squared_error])


        # TODO Continue to write the training method