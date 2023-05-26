import numpy as np

class DataSplitter:
    r"""
    Split a numpy array into smaller arrays of size datapoints_per_split
    If data.shape(0) % datapoints_per_split != 0, the remaining datapoints are dropped
    """
    def __init__(self, datapoints_per_split):
        self.split_size = datapoints_per_split

    def __call__(self, data: np.ndarray):
        """
        data: [[t, i, v]]
        """
        ret_data = []
        for i in range(self.split_size, data.shape[0], self.split_size):
            ret_data.append(data[i-self.split_size:i, :])
        if len(ret_data) == 0:
            raise ValueError(f"data has only {data.shape[0]}, but datapoints_per_split is set to {self.split_size}")
        return ret_data

    def __repr__(self):
        return f"DataSplitter({self.split_size})"
