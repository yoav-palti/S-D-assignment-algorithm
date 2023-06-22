import numpy as np
import Constants
class Signal():
    def __init__(self,one_cycle_signal: np.ndarray, repetition_number: int):
        self.one_cycle_signal = one_cycle_signal  # the one cycle signal
        self.time = len(one_cycle_signal)
        self.repetition_number = repetition_number # np.inf can be used in order to create an infinite signal

    def repeaeted_signal(self, t):
        # emmits a signal at time length t
        if t < self.repetition_number * len(self.one_cycle_signal):
            signal_time = t % self.time
            return self.one_cycle_signal[signal_time]
        return 0

    def original_signal(self):
        return self.one_cycle_signal

    def __len__(self):
        return self.time * self.repetition_number

    def __array__(self) -> np.ndarray:
        # casting to np.array
        return np.repeat(self.one_cycle_signal, self.repetition_number)

    @staticmethod
    def random_signal(strength=1, bandwidth=Constants.BANDWIDTH, length=Constants.NUMSAMPLES, samplingrate=Constants.SAMPLINGRATE):
        """
        a static method to create a random signal with a given bandwidth, sampling rate  and length
        return: numpy array of length : length
        """
        rng = np.random.default_rng()
        randomrate=samplingrate//bandwidth
        randed = rng.standard_normal(length//randomrate+1)
        random_sample=np.zeros(length)
        for i in range(length):
            random_sample[i]=strength*randed[i//randomrate]
        return Signal(random_sample,1)

    def __getitem__(self, item):
        # emmits a signal at time length t
        if item < self.repetition_number * len(self.one_cycle_signal):
            signal_time = item % self.time
            return self.one_cycle_signal[signal_time]
        else:
            raise IndexError('index out of bounds')
