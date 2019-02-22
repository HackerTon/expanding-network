import cv2

class camera:
    def __init__(self):
        self.videocapture = cv2.VideoCapture(0)

    def play(self):
        return self.videocapture.read()