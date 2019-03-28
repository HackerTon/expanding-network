import os, time
import numpy as np
# Colorama for print styling
from colorama import Fore, Back, Style
import cv2
import tensorflow as tf

# Debugging
# tf.enable_eager_execution()

# Default folder for current system
HOME = '/home/hackerton/'


class folder:
    def __init__(self, base, number_of_class):
        self.number_of_class = number_of_class
        self.local_time = time.time()

        if os.path.exists(base):
            self.base_directory = base
        else:
            print(Fore.RED + 'Path does not exists')
            self.base_directory = os.path.join(HOME, 'Documents/opencv_images')

        print(Fore.WHITE + self.base_directory)

    def save(self, index_folder, img):
        self.local_time = time.time()

        if isinstance(index_folder, int) and 0 <= index_folder < self.number_of_class:
            local_file = os.path.join(self.base_directory, str(index_folder))

            print(local_file)

            if os.path.exists(local_file):
                pass
            else:
                print(Fore.RED + f'Directory does not exits: {local_file}')

                try:
                    os.mkdir(local_file)
                except OSError:
                    print(Fore.RED + 'ERROR')

            cv2.imwrite(os.path.join(local_file, str(self.local_time)) + '.png', img)
            print('WRITE IMAGE SUCCESSFUL')
        else:
            print('Index more the classes')

    def generator(self):
        current_iter = 0

        while current_iter != self.number_of_class:
            for image_addr in os.listdir(os.path.join(self.base_directory, str(current_iter))):
                yield os.path.join(self.base_directory, os.path.join(str(current_iter), image_addr)), current_iter

            current_iter += 1

    def dataset_generation(self, batch_size=1):
        total_amount = 0

        # Get number of images
        for x in range(self.number_of_class):
            array = os.listdir(os.path.join(self.base_directory, str(x)))

            total_amount += len(array)

        if batch_size < 0:
            batch_size = total_amount

        if batch_size > total_amount:
            print('Batch is bigger than total_amount')
            batch_size = total_amount
        else:
            batch_size = batch_size

        dataset = tf.data.Dataset.from_generator(self.generator, (tf.string, tf.int64),
                                                 (tf.TensorShape([]), tf.TensorShape([])))

        dataset = dataset.apply(tf.data.experimental.map_and_batch(self._read_image, batch_size=batch_size,
                                                                   num_parallel_batches=1))

        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))

        return dataset

    def _read_image(self, image, value):
        image_string = tf.read_file(image)

        image = tf.image.decode_png(contents=image_string, channels=3)

        image = tf.image.resize_images(images=image, size=[244, 244])

        image = tf.image.convert_image_dtype(image, tf.float32)

        # Normalization process
        image = image - tf.reduce_mean(input_tensor=image)

        return image, value

# folder_1 = folder('/home/hackerton/mylife', number_of_class=3)
# dataset = folder_1.dataset_generation(batch_size=5)
# print(dataset)