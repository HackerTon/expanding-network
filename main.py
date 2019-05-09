from camera import camera
from folder import folder
from neural_network import expandingNetwork
from neural_network import defaultNetwork

from cv2 import imshow, waitKey, destroyAllWindows, putText, LINE_8


def main():
    camera_1 = camera()
    folder_1 = folder('/home/hackerton/mylife', number_of_class=3)

    # network_1 = expandingNetwork(number_of_class=3)

    network_1 = defaultNetwork()

    enable_realtime_prediction = False

    while True:
        ret, image = camera_1.play()

        if enable_realtime_prediction is True:
            probability = network_1.infer(image)
            putText(image, str(probability), (30, 30), 0,
                    1, (0, 0, 0), LINE_8)

        imshow('window', image)
        result = waitKey(1)

        if enable_realtime_prediction is False:
            enable_realtime_prediction = True if result == ord('p') else False

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

            network_1.train(dataset=dataset)

            network_1.save_model()


if __name__ == '__main__':
    main()
