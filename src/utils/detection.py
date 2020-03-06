class Detection(object):
    def __init__(self, frame, id, label, parked, xtl, ytl, xbr, ybr, confidence=1):
        self.frame = frame
        self.id = id
        self.label = label
        self.parked = parked
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.confidence = confidence

    def __str__(self):
        bbox = self.get_bbox()
        return '\n frame={0}, id={1}, label={2}, parked={3}, confidence={4} bbox={5}'.format(self.frame, self.id, self.label, self.parked, self.confidence, bbox)

    def get_bbox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    def get_width(self):
        return abs(self.xbr - self.xtl)

    def get_height(self):
        return abs(self.ytl - self.ybr)
