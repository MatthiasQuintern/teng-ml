
from os import path, listdir
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# groups: date, name, voltage, distance, index
re_filename = r"(\d{4}-\d{2}-\d{2})_([a-zA-Z]+)_(\d{1,2}(?:\.\d*)?)V_(\d+(?:\.\d*)?)mm(\d+).csv"

class LabelConverter:
    def __init__(self, class_labels):
        self.class_labels = class_labels.copy()
        self.class_labels.sort()

    def get_one_hot(self, label):
        """return one hot vector for given label"""
        vec = np.zeros(len(self.class_labels), dtype=np.float32)
        vec[self.class_labels.index(label)] = 1.0
        return vec

    def __getitem__(self, index):
        return self.class_labels[index]

    def __contains__(self, value):
        return value in self.class_labels

    def get_labels(self):
        return self.class_labels.copy()



class Datasample:
    def __init__(self, date: str, label: str, voltage: str, distance: str, index: str, label_vec, datapath: str):
        self.date = date
        self.label = label
        self.voltage = float(voltage)
        self.distance = float(distance)
        self.index = int(index)
        self.label_vec = label_vec
        self.datapath = datapath
        self.data = None

    def __repr__(self):
        size = self.data.size if self.data else "Unknown"
        return f"{self.label}-{self.index}: dimension={size}, recorded at {self.date} with U={self.voltage}V, d={self.distance}mm"

    def _load_data(self):
        df = pd.read_csv(self.datapath)
        self.data = df.to_numpy()

    def get_data(self):
        """[[timestamps, idata, vdata]]"""
        if not self.data:
            self._load_data()
        return self.data

class Dataset:
    """
    Store the whole dataset, compatible with torch.data.Dataloader
    """
    def __init__(self, datasamples, transforms=None):
        self.datasamples = datasamples
        self.transforms = transforms
        # self.labels = [ d.label_vec for d in datasamples ]
        # self.data = [ d.get_data() for d in datasamples ]

    def __getitem__(self, index):
        data, label = self.datasamples[index].get_data(), self.datasamples[index].label_vec
        if type(self.transforms) == list:
            for t in self.transforms:
                data = t(data)
        elif self.transforms:
            data = self.transforms(data)
        # TODO
        return data[:400], label

    def __len__(self):
        return len(self.datasamples)

def load_datasets(datadir, labels: LabelConverter, transforms=None, voltage=None, train_to_test_ratio=0.7, random_state=None):
    """
    load all data from datadir that are in the format: yyyy-mm-dd_label_x.xV_xxxmm.csv
    """
    datasamples = []
    files = listdir(datadir)
    files.sort()
    for file in files:
        match = re.fullmatch(re_filename, file)
        if not match: continue

        label = match.groups()[1]
        if label not in labels: continue

        sample_voltage = float(match.groups()[2])
        if voltage and voltage != sample_voltage: continue

        datasamples.append(Datasample(*match.groups(), labels.get_one_hot(label), datadir + "/" + file))
    train_samples, test_samples = train_test_split(datasamples, train_size=train_to_test_ratio, shuffle=True, random_state=random_state)
    train_dataset = Dataset(train_samples, transforms=transforms)
    test_dataset = Dataset(test_samples, transforms=transforms)
    return train_dataset, test_dataset
