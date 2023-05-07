from ..util.data_loader import LabelConverter
import time
import torch

class EpochTracker:
    """
    Track progress through epochs and generate statistics
    """
    def __init__(self, labels: LabelConverter):
        # Training
        self.accuracy = []
        self.loss = []
        self.times = []  # timestamps for each epoch end
        self.trainings = []
        self.training_indices = [[]]  # epoch, batch_nr, (correct_indices, predicted_indices), ind:ex_nr
        self._current_epoch = 0

        self.labels = labels

        # Testing
        self.tests = []  # (correct_indices, predicted_indices)

    def train_begin(self):
        """for time tracking"""
        self.times.append(time.time())

    # TRAINING
    def train(self, correct_indices: torch.Tensor, predicted_indices: torch.Tensor):
        self.training_indices[self._current_epoch].append((correct_indices, predicted_indices))

    def next_epoch(self, loss):
        self.times.append(time.time())
        self.loss.append(loss)
        correct_predictions = 0
        total_predictions = 0
        for predicted_indices, correct_indices in self.training_indices[self._current_epoch]:
            correct_predictions += (predicted_indices == correct_indices).sum().item()
            total_predictions += predicted_indices.size(0)
        accuracy = 100 * correct_predictions / total_predictions
        self.accuracy.append(accuracy)
        self._current_epoch += 1
        self.training_indices.append([])

    def get_last_epoch_summary_str(self):
        """call after next_epoch()"""
        return f"Epoch {self._current_epoch:3}: Accuracy={self.accuracy[-1]:.2f}, Loss={self.loss[-1]:.3f}, Training duration={self.times[-1] - self.times[0]:.2f}s"
    def get_last_epoch_summary(self):
        """
        @returns accuracy, loss, training time
        """
        return self.accuracy[-1], self.loss[-1], self.times[-1] - self.times[0]

    def get_training_count_per_label(self):
        count_per_label = [ 0 for _ in range(len(self.labels)) ]
        for i in range(len(self.training_indices)):
            for j in range(len(self.training_indices[i])):
                for k in range(self.training_indices[i][j][0].size(0)):
                    # epoch, batch_nr, 0 = correct_indices, correct_index_nr
                    count_per_label[self.training_indices[i][j][0][k]] += 1
        return count_per_label

    def __len__(self):
        return len(self.accuracy)

    def __getitem__(self, idx):
        return (self.accuracy[idx], self.loss[idx])

    # TESTING
    def test(self, correct_indices: torch.Tensor, predicted_indices: torch.Tensor):
        """
        @param correct_indices and predicted_indices: 1 dim Tensor
        """
        for i in range(correct_indices.size(0)):
            self.tests.append((correct_indices[i], predicted_indices[i]))


    def get_test_statistics(self):
        # label i, label_j was predicted when label_i was correct 
        statistics = [ [ 0 for _ in range(len(self.labels))] for _ in range(len(self.labels)) ]
        for corr, pred in self.tests:
            statistics[corr][pred] += 1
        print(statistics)
        return statistics
