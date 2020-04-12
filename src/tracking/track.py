import random

class Track(object):
    def __init__(self, id, detections):
        self.id = id
        self.track = detections
        self.color = (int(random.random() * 256),
                      int(random.random() * 256),
                      int(random.random() * 256))
        self.terminated = False

    def get_track(self):
        return self.track

    def add_detection(self, detection):
        self.track.append(detection)

    def last_detection(self):
        return self.track[-1]