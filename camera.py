import cv2


class camera:
    def __init__(self):
        self.videocapture = cv2.VideoCapture(0)

    def play(self):
        if self.videocapture.isOpened():
            return self.videocapture.read()
