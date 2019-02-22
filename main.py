from camera import camera
from folder import folder
from neural_network import expanding_network

from cv2 import imshow, waitKey, destroyAllWindows


def main():
    camera_1 = camera()
    folder_1 = folder('/home/hackerton/mylife',
                      number_of_class=3)

    network_1 = expanding_network(number_of_class=3)

    while True:
        ret, image = camera_1.play()
        imshow('window', image)

        print(f'Function: ret')

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
            folder_1.load()


if __name__ == '__main__':
    main()
