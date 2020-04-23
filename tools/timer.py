import time


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        pass

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def log_duration(self):
        print('Duration is: {}'.format(self.end_time - self.start_time))
