import random


class Track(object):
    def __init__(self, id, detections, camera=0):
        self.id = id
        self.track = detections
        self.color = (int(random.random() * 256),
                      int(random.random() * 256),
                      int(random.random() * 256))
        self.terminated = False
        self.prev_track = None
        self.next_track = None
        self.camera = camera

    def get_track(self):
        return self.track

    def add_detection(self, detection):
        self.track.append(detection)

    def last_detection(self):
        return self.track[-1]

    def set_prev_track(self, track):
        self.prev_track = track

    def get_prev_track(self):
        return self.prev_track

    def set_next_track(self, track):
        self.next_track = track

    def get_next_track(self):
        return self.next_track
