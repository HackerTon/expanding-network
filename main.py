from camera import camera
from folder import folder
from neural_network import expandingNetwork

from cv2 import imshow, waitKey, destroyAllWindows


def main():
    camera_1 = camera()
    folder_1 = folder('/home/hackerton/mylife', number_of_class=3)

    network_1 = expandingNetwork(number_of_class=3)

    while True:
        ret, image = camera_1.play()
        imshow('window', image)

        result = waitKey(1)

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


if __name__ == '__main__':
    main()
