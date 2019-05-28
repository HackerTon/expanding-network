import numpy as np

from camera import camera
from folder import folder
from neural_network import expandingNetwork, defaultNetwork
from cv2 import imshow, waitKey, destroyAllWindows, putText, LINE_8
from cv2 import imread, imreadmulti, cvtColor, COLOR_BGR2RGB

BATCH_SIZE = 15
NETWORK = 'resnet'


def main():
    camera_1 = camera()
    folder_1 = folder('/home/hackerton/mylife', number_of_class=3)

    network_1 = expandingNetwork(number_of_class=3, model_type=NETWORK)

    enable_realtime_prediction = False

    while True:
        ret, image = camera_1.play()

        if NETWORK == 'inception':
            infer_image = cvtColor(image, COLOR_BGR2RGB)
        else:
            infer_image = image

        if enable_realtime_prediction is True:
            probability = network_1.infer(image=infer_image, network_type=NETWORK)

            index = np.argmax(probability[0])

            ls_person = ('a', 'c', 'p')

            putText(
                img=image,
                text=str(ls_person[index]),
                org=(30, 30),
                fontFace=0,
                fontScale=1,
                color=(255, 255, 255),
                lineType=LINE_8
            )

            print(f'Probability: {probability[0]}, Index: {index}')

        count = folder_1.count()

        print(count[::-1])

        imshow('window', image)
        result = waitKey(1)

        if result == ord('p'):
            enable_realtime_prediction = True if enable_realtime_prediction is False else False
            print('Enable' if enable_realtime_prediction is True else 'Disable')

        if result == ord('q'):
            destroyAllWindows()
            break
        elif result == ord('s'):
            folder_1.save(0, img=image)
        elif result == ord('d'):
            folder_1.save(1, img=image)
        elif result == ord('f'):
            folder_1.save(2, img=image)
        elif result == ord('t'):
            destroyAllWindows()
            camera_1.stop()

            dataset = folder_1.dataset_generation(batch_size=BATCH_SIZE, network_type=NETWORK)
            steps_per_epoch = folder_1.dataset_size / BATCH_SIZE

            network_1.train(dataset=dataset, steps_per_epoch=steps_per_epoch, validation_data=None)
            network_1.save_model()


if __name__ == '__main__':
    main()
