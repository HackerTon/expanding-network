import os
import time
import cv2
import glob
import pathlib
import tensorflow as tf

# Debugging
# tf.enable_eager_execution()

# Default folder for current system
HOME = '/home/hackerton/'


class folder:
    def __init__(self, base, number_of_class):
        self.number_of_class = number_of_class
        self.local_time = time.time()
        self.dataset_size = 0

        if os.path.exists(base):
            self.base_directory = base
        else:
            print('Path does not exists')
            self.base_directory = os.path.join(HOME, 'Documents/opencv_images')

        print(self.base_directory)

    def count(self):
        base_path = pathlib.Path(self.base_directory)

        subd = []

        for subdirectory in base_path.iterdir():
            subd.append(len(list(subdirectory.iterdir())))

        return subd

    def save(self, index_folder, img):
        self.local_time = time.time()

        try:
            index_folder = int(index_folder)

            if 0 <= index_folder <= self.number_of_class:
                local_path = os.path.join(self.base_directory, str(index_folder))

                local_dir = pathlib.Path(local_path)

                if not local_dir.exists():
                    print(f'Directory does not exists: {local_dir}')

                    try:
                        local_dir.mkdir()
                    except Exception as e:
                        print(e)

                write_or_failed = cv2.imwrite(os.path.join(local_path, str(self.local_time) + '.jpeg'), img)

                if write_or_failed:
                    print('Write successful!')
                else:
                    print('Write failed!')
            else:
                print('You try to index more than the classes avaiable!')

        except TypeError as e:
            print('Save failed!')
            print(e)

    def generator(self):
        current_iter = 0

        while current_iter != self.number_of_class:
            for image_addr in os.listdir(os.path.join(self.base_directory, str(current_iter))):
                yield os.path.join(self.base_directory, os.path.join(str(current_iter), image_addr)), current_iter

            current_iter += 1

    def dataset_generation(self, batch_size=1, network_type='inception'):
        # Get the size of whole dataset
        self.dataset_size = len(glob.glob(os.path.join(self.base_directory, '**/*.jpeg'), recursive=True))

        if batch_size < 0:
            batch_size = self.dataset_size
            print('Batch_size defaulted!')
        elif batch_size > self.dataset_size:
            print('batch_size is bigger than self.dataset_size')
            print('Batch_size defaulted!')
            batch_size = self.dataset_size
        else:
            batch_size = batch_size

        if network_type == 'inception':
            _image_read = self._read_image_inception
        elif network_type == 'resnet':
            _image_read = self._read_image_resnet
        else:
            _image_read = None

        dataset = tf.data.Dataset.from_generator(
            generator=self.generator,
            output_types=(tf.string, tf.int32),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
        )

        dataset = dataset.shuffle(buffer_size=self.dataset_size)
        dataset = dataset.repeat()
        dataset = dataset.map(map_func=_image_read, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def _read_image_inception(self, image, index):
        image_bytes = tf.read_file(image)
        image = tf.image.decode_jpeg(contents=image_bytes, channels=3)
        # image = tf.image.random_crop(value=image, size=[100, 100, 3])
        image = tf.image.resize_images(images=image, size=[200, 200])
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Normalization process
        image = image[..., ::-1]
        image /= 127.5
        image -= 1.

        return image, tf.one_hot(indices=index, depth=self.number_of_class)

    def _read_image_resnet(self, image, index):
        image_bytes = tf.read_file(image)
        image = tf.image.decode_jpeg(contents=image_bytes, channels=3)
        # image = tf.image.random_crop(value=image, size=[100, 100, 3])
        image = tf.image.resize_images(images=image, size=[200, 200])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.random_flip_left_right(image)

        # Normalization process
        mean = [103.939, 116.779, 123.68]

        image -= mean

        return image, tf.one_hot(indices=index, depth=self.number_of_class)
