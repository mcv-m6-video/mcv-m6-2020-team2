class Detection:

    def __init__(self, frame, id, label, xtl, ytl, xbr, ybr, score=None, parked=None):
        self.frame = frame
        self.id = id
        self.label = label
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.score = score
        self.parked = parked

    @property
    def bbox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    @property
    def width(self):
        return abs(self.xbr - self.xtl)

    @property
    def height(self):
        return abs(self.ytl - self.ybr)

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (int((self.xtl + self.xbr) / 2), int((self.ybr + self.ytl) / 2))

    def __str__(self):
        return f'frame={self.frame}, id={self.id}, label={self.label}, bbox={self.bbox}, confidence={self.score}, parked={self.parked}'
