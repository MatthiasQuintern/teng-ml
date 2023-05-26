from ..util.data_loader import LabelConverter
import matplotlib.pyplot as plt
import time
import torch
import numpy as np

class EpochTracker:
    """
    Track accuracy, loss, learning_rate etc. during model training
    Can also be used for validation (which will probably be only one epoch)
    """
    def __init__(self, labels: LabelConverter):
        self.labels = labels

        self.times: list[float] = []            # (epoch)
        self.predictions = [[]]  # (epoch, batch_nr, (correct_indices | predicted_indices), ind:ex_nr)
        self.loss: list[float] = []             # (epoch)
        self.learning_rate: list[float] = []    # (epoch)
        self.epochs: list[int] = []             # 1 based for FINISHED epochs
        self._current_epoch = 0    # 0 based

        # after training
        self.accuracies: list[float] = []       # (epoch)

    def begin(self):
        self.times.append(time.time())

    def end(self):
        self.times.append(time.time())
        # if end_epoch was called before end:
        if len(self.predictions[-1]) == 0:
            self.predictions.pop()
            self._current_epoch -= 1
        else:  # if end_epoch was not called
            self.epochs.append(len(self.epochs) + 1)
            self._calculate_accuracies(self._current_epoch)


        s = f"Summary: After {self.epochs[-1]} epochs: "
        s += f"Accuracy={self.accuracies[-1]:.2f}%"
        s += f", Total time={self.get_total_time():.2f}s"
        return s



    def get_total_time(self):
        if len(self.times) > 1: return self.times[-1] - self.times[0]
        else: return -1

    #
    # EPOCH
    #
    def end_epoch(self, loss, learning_rate):
        """
        loss and learning_rate of last epoch
        call before scheduler.step()
        """
        self.times.append(time.time())
        self.epochs.append(len(self.epochs) + 1)
        if type(loss) == torch.Tensor: self.loss.append(loss.item())
        else: self.loss.append(loss)
        self.learning_rate.append(learning_rate)
        self._calculate_accuracies(self._current_epoch)

        self._current_epoch += 1
        self.predictions.append([])

    def get_epoch_summary_str(self, ep=-1):
        """call after next_epoch()"""
        m = max(ep, 0)  # if ep == -1, check if len is > 0
        assert(len(self.epochs) > m)
        s = f"Epoch {self.epochs[ep]:3}"
        if len(self.accuracies) > m:s += f", Accuracy={self.accuracies[ep]:.2f}%"
        if len(self.loss) > m:      s += f", Loss={self.loss[ep]:.3f}"
        if len(self.loss) > m:      s += f", lr={self.learning_rate[ep]:.4f}"
        if len(self.times) > m+1:   s += f", dt={self.times[ep] - self.times[ep-1]:.2f}s"
        return s

    def add_prediction(self, correct_indices: torch.Tensor, predicted_indices: torch.Tensor):
        """for accuracy calculation"""
        self.predictions[self._current_epoch].append((correct_indices.detach().numpy(), predicted_indices.detach().numpy()))

    #
    # STATISTICS
    #
    def get_count_per_label(self, epoch=-1):
        """
        the number of times where <label> was the correct label, per label
        @returns shape: (label)
        """
        count_per_label = [ 0 for _ in range(len(self.labels)) ]
        for corr, _ in self.predictions[epoch]:
            for batch in range(len(corr)):
                count_per_label[corr[batch]] += 1
        return count_per_label

    def get_predictions_per_label(self, epoch=-1):
        """
        How often label_i was predicted, when label_j was the correct label
        @returns shape: (label_j, label_i)
        """
        statistics = [ [ 0 for _ in range(len(self.labels)) ] for _ in range(len(self.labels)) ]
        for corr, pred in self.predictions[epoch]:
            for batch in range(len(corr)):
                statistics[corr[batch]][pred[batch]] += 1
        return statistics

    def plot_training(self, title="Training Summary", model_dir=None, name="img_training"):
        """
        @param model_dir: Optional. If given, save to model_dir as svg
        """
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, layout="tight")

        ax[0].plot(self.epochs, self.accuracies, color="red")
        ax[0].set_ylabel("Accuracy")

        ax[1].plot(self.epochs, self.learning_rate, color="green")
        ax[1].set_ylabel("Learning Rate")

        ax[2].plot(self.epochs, self.loss, color="blue")
        ax[2].set_ylabel("Loss")

        fig.suptitle(title)
        ax[2].set_xlabel("Epoch")
        plt.tight_layout()
        if model_dir is not None:
            fig.savefig(f"{model_dir}/{name}.svg")

        return fig, ax

    def plot_predictions(self, title="Predictions per Label", ep=-1, model_dir=None, name="img_training_predictions"):
        """
        @param model_dir: Optional. If given, save to model_dir as svg
        @param ep: Epoch, defaults to last
        """
        # Normalize the data
        predictions_per_label = self.get_predictions_per_label(ep)
        normalized_predictions = predictions_per_label / np.sum(predictions_per_label, axis=1, keepdims=True)

        N = len(self.labels)
        label_names = self.labels.get_labels()

        fig, ax = plt.subplots(layout="tight")

        im = ax.imshow(normalized_predictions, cmap='Blues')  # cmap='BuPu'
        ax.set_xticks(np.arange(N))
        ax.set_yticks(np.arange(N))
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Correct Label')

        # horizontal lines between labels to better show that the sum of a row is 1
        for i in range(1, N):
            ax.axhline(i-0.5, color='black', linewidth=1)

        # rotate the x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # create annotations
        for i in range(N):
            for j in range(N):
                text = ax.text(j, i, round(normalized_predictions[i, j], 2),
                               ha="center", va="center", color="black")

        # add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        ax.set_title(title)
        plt.tight_layout()
        if model_dir is not None:
            fig.savefig(f"{model_dir}/{name}.svg")
        return fig, ax

    #
    # CALCULATION
    #
    def _calculate_accuracies(self, ep):
        correct_predictions = 0
        total_predictions = 0
        for correct_indices, predicted_indices in self.predictions[ep]:
            correct_predictions += (predicted_indices == correct_indices).sum().item()
            total_predictions += len(predicted_indices)
        accuracy = correct_predictions / total_predictions * 100
        while len(self.accuracies) <= ep:
            self.accuracies.append(-1)
        self.accuracies[ep] = accuracy
