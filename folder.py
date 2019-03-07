import os, time
import numpy as np
# Colorama for print styling
from colorama import Fore, Back, Style
import cv2

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