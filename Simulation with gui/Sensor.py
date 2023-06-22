import numpy as np
from scipy import signal
import Constants
import matplotlib.pyplot as plt

from Object_2d import Object_2d as Obj


class Sensor(Obj):
    TIME_ERR = 10 ** -6

    @staticmethod
    def correlation(sensor1, sensor2, samplingrate=Constants.SAMPLINGRATE, rectsize=Constants.THRESHOLDWIDTH,
                    alpha=Constants.THRESHOLD):
        corr = signal.fftconvolve(sensor1.get_signal(), sensor2.get_signal()[::-1], "same")
        rect = np.linspace(alpha / rectsize, alpha / rectsize, rectsize)
        threshold = signal.fftconvolve(abs(corr), rect, "same")[rectsize // 2:len(corr) - rectsize // 2]
        corr = corr[rectsize // 2:len(corr) - rectsize // 2]
        peaks = signal.find_peaks(corr - threshold, height=0)
        u = (peaks[0])
        for i in range(len(u)):
            u[i] -= len(corr) // 2
        return u / samplingrate
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(np.arange(-(len(threshold)//2), len(threshold)//2), threshold)
        # ax.plot(np.arange(-(len(corr) // 2), len(corr) // 2), corr)
        # plt.show()
        # index = corr.argmax() - len(corr) // 2
        # return index / samplingrate
        # return (peaks - np.linspace((len(corr)//2, len(corr) // 2, len(peaks))) / samplingrate

    @staticmethod
    def correlation_to_signal(signal1, signal2, samplingrate=Constants.SAMPLINGRATE, rectsize=Constants.THRESHOLDWIDTH,
                              alpha=Constants.THRESHOLD):
        corr = signal.fftconvolve(signal1, signal2[::-1], "same")
        rect = np.linspace(alpha / rectsize, alpha / rectsize, rectsize)
        threshold = signal.fftconvolve(abs(corr), rect, "same")[rectsize // 2:len(corr) - rectsize // 2]
        corr = corr[rectsize // 2:len(corr) - rectsize // 2]
        peaks = signal.find_peaks(corr - threshold, height=0)
        u = (peaks[0])
        u -= len(corr) // 2
        return u / samplingrate

    @staticmethod
    def correlation_cmplx(sensor1, sensor2):
        pass

    def __init__(self, size=Constants.NUMSAMPLES, noise=1, dt=0, location=(0, 0), time=0, random=True,
                 time_err=TIME_ERR):
        super().__init__(location)
        self.signal = np.zeros(size)
        self.noise = noise
        # self.time_delay=dt //I think this line is not necessary.
        self.time = time  # I think this line is also not necessary.
        if random == True:
            self.time = np.random.random(1) * time_err

    def add_signal(self, index, signal, decay):
        index_delay = index + int(self.time * Constants.SAMPLINGRATE)
        signal_length = len(signal)
        length = min((index_delay + signal_length), len(self.signal)) - (index_delay)
        if length > 0:
            self.signal[index_delay:min((index_delay + signal_length), len(self.signal))] += decay * np.array(signal)[
                                                                                                     0:length]

    def cor2me(self, sensor):
        return self.correlation(self, sensor)

    def get_hyperbole(self, sensor):
        return Hyperbole.Hyperbole(self, sensor)

    def get_limits(self, distance, samplingrate=Constants.SAMPLINGRATE):
        pass

    def get_signal(self):
        return self.signal

    def set_signal(self, emitter):
        pass
