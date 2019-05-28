import cv2


class camera:
    def __init__(self):
        self.videocapture = cv2.VideoCapture(0)
        self.running = False

    def play(self):
        self.running = True

        if self.running is True:
            if self.videocapture.isOpened():
                return self.videocapture.read()
            else:
                self.videocapture.open(0)
                return self.videocapture.read()
        else:
            return []

    def stop(self):
        self.running = False
        self.videocapture.release()
