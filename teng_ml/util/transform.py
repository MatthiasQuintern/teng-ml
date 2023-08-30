import numpy as np
from scipy.interpolate import interp1d
from torch import mul

class Normalize:
    """
    normalize so that all values are between low and high
    """
    def __init__(self, low=0, high=1):
        assert(low < high)
        self.low = low
        self.high = high
    def __call__(self, data):
        min_ = np.min(data)
        data = data - min_   # smallest point is 0 now
        max_ = np.max(data)
        if max_ != 0:
            data = (data / max_)
        # now normalized between 0 and 1
        data *= (self.high - self.low)
        data += self.low
        return data

    def __repr__(self):
        return f"Normalize(low={self.low}, high={self.high})"

class NormalizeAmplitude:
    """
    scale data so that all values are between -high and high
    """
    def __init__(self, high=1):
        self.high = high

    def __call__(self, data):
        min_ = np.min(data)
        max_ = np.max(data)
        scale = np.max([np.abs(min_), np.abs(max_)])
        if scale != 0:
            data = data / scale * self.high
        return data
    def __repr__(self):
        return f"NormalizeAmplitude(high={self.high})"


class Multiply:
    def __init__(self, multiplier):
        self.multiplier = multiplier
    def __call__(self, data):
        return data * self.multiplier
    def __repr__(self):
        return f"Multiply(multiplier={self.multiplier})"


class ConstantInterval:
    """
    Interpolate the data to have a constant interval / sample rate,
    so that 1 index step is always equivalent to a certain time step
    """
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, data):
        """
        array: [timestamps, data1, data2...]
        """
        timestamps = data[:,0]
        new_stamps = np.arange(timestamps[0], timestamps[-1], self.interval)
        ret = new_stamps
        for i in range(1, data.shape[1]):  # 
            interp = interp1d(timestamps, data[:,i])
            new_vals = interp(new_stamps)
            ret = np.vstack((ret, new_vals))
        return ret.T

    @staticmethod
    def get_average_interval(timestamps):
        avg_interval = np.average([ timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))])
        return avg_interval
        # sug_interval = 0.5 * avg_interval
        # print(f"Average interval: {avg_interval}, Suggestion: {sug_interval}")

    def __repr__(self):
        return f"ConstantInterval(interval={self.interval})"
