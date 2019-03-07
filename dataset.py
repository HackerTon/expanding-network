import tensorflow as tf
from tensorflow import keras


# Read image from a image data
def _read_image(image, value):
    img_string = tf.read_file(image)
    image = tf.image.decode_and_crop_jpeg(img_string, crop_window=[47, 80, 66, 200], channels=0)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image - tf.reduce_mean(input_tensor=image)
    return image, value


def import_dataset_from_generator(generator, batch_size):
    dataset = tf.data.Dataset.from_generator(generator, (tf.string, tf.int64),
                                             (tf.TensorShape([]), tf.TensorShape([])))

    dataset = dataset.apply(tf.data.experimental.map_and_batch(_read_image, batch_size=batch_size,
                                                               num_parallel_batches=1))

    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))

    return dataset
