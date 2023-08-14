import torch
import torch.nn.utils.rnn as rnn
import numpy as np

class PadSequences:
    def __call__(self, batch):
        # batch = [(data, label)]
        # sort by length
        sorted_batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
        sequences = [torch.Tensor(sample[0]) for sample in sorted_batch]
        labels = torch.Tensor(np.array([sample[1] for sample in sorted_batch]))
        lengths = torch.IntTensor(np.array([seq.shape[0] for seq in sequences]))
        sequences_padded = rnn.pad_sequence(sequences, batch_first=True)
        return sequences_padded, lengths, labels
