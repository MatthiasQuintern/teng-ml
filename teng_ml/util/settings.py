from ..util.data_loader import LabelConverter
from ..util.split import DataSplitter

class MLSettings:
    """
    Manage model and training settings for easy saving and loading
    """
    def __init__(self,
                 num_features=1,
                 num_layers=1,
                 hidden_size=1,
                 bidirectional=True,
                 optimizer=None,
                 scheduler=None,
                 loss_func=None,
                 transforms=[],
                 splitter=None,
                 num_epochs=10,
                 batch_size=5,
                 labels=LabelConverter([]),
             ):
        self.num_features = num_features
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.bidirectional = bidirectional
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.transforms = transforms
        self.splitter = splitter
        self.batch_size = batch_size
        self.labels = labels

    def get_name(self):
        """
        F = num_features
        L = num_layers
        H = hidden_size
        B = bidirectional
        T = #transforms
        S = splitter
        E = #epochs
        """
        return f"F{self.num_features}L{self.num_layers}H{self.hidden_size:02}B{'1' if self.bidirectional else '0'}T{len(self.transforms)}S{self.splitter.split_size if type(self.splitter) == DataSplitter is not None else 0:03}E{self.num_epochs:03}"
