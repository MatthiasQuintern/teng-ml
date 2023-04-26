import numpy as np
from scipy.interpolate import interp1d

class Normalize:
    """
    normalize so that all values are between low and high
    """
    def __init__(self, low=0, high=1):
        assert(low < high)
        self.low = low
        self.high = high
    def __call__(self, a):
        min_ = np.min(a)
        a = a - min_
        max_ = np.max(a)
        if max_ != 0:
            a = (a / max_)
        # now normalized between 0 and 1
        a *= (self.high - self.low)
        a -= self.low
        return a


class ConstantInterval:
    """
    Interpolate the data to have a constant interval / sample rate,
    so that 1 index step is always equivalent to a certain time step
    Expects: timestamps, idata, vdata
    """
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, timestamps, data):
        interp = interp1d(timestamps, data)
        new_stamps = np.arange(0, timestamps[-1], self.interval)
        print(f"old=({timestamps.size}) {timestamps}, new=({new_stamps.size}){new_stamps}")

        new_vals = interp(new_stamps)
        return np.vstack((new_stamps, new_vals)).T
    @staticmethod

    def get_average_interval(timestamps):
        avg_interval = np.average([ timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))])
        return avg_interval
        # sug_interval = 0.5 * avg_interval
        # print(f"Average interval: {avg_interval}, Suggestion: {sug_interval}")

