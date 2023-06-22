import random
from typing import Tuple
from Sensor import Sensor
import Constants
from Object_2d import Object_2d
import numpy as np

class Emmiter(Object_2d):
    def __init__(self, location: Tuple[int, int], signal: np.ndarray, repetitions: int, dt=0):
        self.signal = signal # the signal is the
        self.dt=dt
        super().__init__(location)

    def emit_signal(self,sensor_list):
        for sensor in sensor_list:
            distance=self.distance(sensor)
            decay = 1 / distance
            sensor.add_signal(int(Constants.SAMPLINGRATE*(self.dt+distance/Constants.C)), self.signal, decay)
        return

    def emit_signal_random_block(self,sensor_list, probability):
        for sensor in sensor_list:
            distance=self.distance(sensor)
            if random.random() < probability:
                decay = 1 / distance
                sensor.add_signal(int(Constants.SAMPLINGRATE*(self.dt+distance/Constants.C)), self.signal, decay)
        return

    def get_signal(self):
        return self.signal