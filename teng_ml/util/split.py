import numpy as np

class DataSplitter:
    r"""
    Split a numpy array into smaller arrays of size datapoints_per_split
    If data.shape(0) % datapoints_per_split != 0, the remaining datapoints are dropped
    """
    def __init__(self, datapoints_per_split, drop_if_smaller_than=-1):
        """
        @param drop_if_smaller_than: drop the remaining datapoints if the sequence would be smaller than this value. -1 means drop_if_smaller_than=datapoints_per_split
        """
        self.split_size = datapoints_per_split
        self.drop_threshhold = datapoints_per_split if drop_if_smaller_than == -1 else drop_if_smaller_than

    def __call__(self, data: np.ndarray):
        """
        data: [[t, i, v]]
        """
        ret_data = []
        for i in range(self.split_size, data.shape[0], self.split_size):
            ret_data.append(data[i-self.split_size:i, :])

        rest_start = len(ret_data) * self.split_size
        if len(data) - rest_start >= self.drop_threshhold:
            ret_data.append(data[rest_start:,:])

        if len(ret_data) == 0:
            raise ValueError(f"data has only {data.shape[0]}, but datapoints_per_split is set to {self.split_size}")
        return ret_data

    def __repr__(self):
        return f"DataSplitter({self.split_size})"
