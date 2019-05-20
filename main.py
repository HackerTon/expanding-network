from camera import camera
from folder import folder
from neural_network import expandingNetwork
from neural_network import defaultNetwork

from cv2 import imshow, waitKey, destroyAllWindows, putText, LINE_8


def main():
    camera_1 = camera()
    folder_1 = folder('/home/hackerton/mylife', number_of_class=3)

    network_1 = expandingNetwork(number_of_class=3)

    enable_realtime_prediction = False

    while True:
        ret, image = camera_1.play()

        if enable_realtime_prediction is True:
            probability = network_1.infer(image)
            putText(
                img=image,
                text=str(probability),
                org=(30, 30),
                fontFace=0,
                fontScale=1,
                color=(0, 0, 0),
                lineType=LINE_8
            )
            print(probability)

        imshow('window', image)
        result = waitKey(1)

        if result == ord('p'):
            enable_realtime_prediction = True if enable_realtime_prediction is False else False
            print('Enable' if enable_realtime_prediction is True else 'Disable')

        if result == ord('q'):
            destroyAllWindows()
            break
        elif result == ord('s'):
            folder_1.save('s', img=image)
        elif result == ord('d'):
            folder_1.save(1, img=image)
        elif result == ord('f'):
            folder_1.save(2, img=image)
        elif result == ord('t'):
            dataset = folder_1.dataset_generation(batch_size=5)

            steps_per_epoch = folder_1.dataset_size / 5

            network_1.train(dataset=dataset, steps_per_epoch=steps_per_epoch)

            network_1.save_model()


if __name__ == '__main__':
    main()
