import numpy as np
from Sensor import Sensor
from Emmiter import Emmiter
class Channel():
    def __init__(self, signal: np.ndarray, time: int, emmiter, c: float, d: float):
        self.signal = signal
        self.time = time # the time of the creation of the emmision
        self.emmiter = emmiter
        self.c = c
        self.distance_units = d

    def add_signal_to_sensors(self, sensor_list):
        # adds the signal from time t to the sensor in time t + d/c
        for sensor in sensor_list:
            distance = self.emmiter.distance(sensor)
            decay = 1 / (distance * self.distance_units)**2
            sensor.add_signal(round(distance/self.c), self.signal, decay)

    #def time_development(self): Not needed. I created a model with no time developement.
        # will help in add_signal_to_sensors
    #    pass
