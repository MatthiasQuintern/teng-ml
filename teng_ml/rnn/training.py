from os import makedirs, path
import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..util.settings import MLSettings
from ..tracker.epoch_tracker import EpochTracker
from ..util.file_io import get_next_digits
from ..util.string import class_str, optimizer_str

from ..util import model_io as mio


def select_device(force_device=None):
    """
    Select best device and move model
    """
    if force_device is not None:
        device = force_device
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            # else "mps"
            # if torch.backends.mps.is_available()
            else "cpu"
        )
    # print(device, torch.cuda.get_device_name(device), torch.cuda.get_device_properties(device))
    return device


def train(model, optimizer, scheduler, loss_func, train_loader: DataLoader, st: MLSettings, print_interval=1, print_continuous=False, training_cancel_points=[]) -> EpochTracker:
    epoch_tracker = EpochTracker(st.labels)
    epoch_tracker.begin()
    for ep in range(st.num_epochs):
        loss = -1
        for i, (data, lengths, y) in enumerate(train_loader):
            # data = batch, seq, features
            x = data[:,:,[2]].float()   # select voltage data
            # print(f"x({x.shape}, {x.dtype})=...")
            # print(f"y({y.shape}, {y.dtype})=...")
            # pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
            # out = model(pack)  # really slow
            out = model(x, lengths)

            with torch.no_grad():
                predicted = torch.argmax(out, dim=1, keepdim=False)  # -> [ label_indices ]
                correct = torch.argmax(y, dim=1, keepdim=False)  # -> [ label_indices ]
                # print(f"predicted={predicted}, correct={correct}")
                # train_total += y.size(0)
                # train_correct += (predicted == correct).sum().item()
                epoch_tracker.add_prediction(correct, predicted)
                # predicted2 = torch.argmax(out, dim=1, keepdim=True)  # -> [ label_indices ]
                # print(f"correct={correct}, y={y}")
            loss = loss_func(out, correct)
            # loss = loss_func(out, y)


            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            # predicted = torch.max(torch.nn.functional.softmax(out), 1)[1]
        epoch_tracker.end_epoch(loss, optimizer.param_groups[0]["lr"])
        if (ep+1) % print_interval == 0:
            if print_continuous: end='\r'
            else: end='\n'
            print(f"Training:", epoch_tracker.get_epoch_summary_str(), end=end)
        # cancel training if model is not good enough
        if len(training_cancel_points) > 0 and ep+1 == training_cancel_points[0][0]:
            if epoch_tracker.accuracies[-1] < training_cancel_points[0][1]:
                print(f"Training cancelled because the models accuracy={epoch_tracker.accuracies[-1]:.2f} < {training_cancel_points[0][1]} after {ep+1} epochs.")
                break;
            training_cancel_points.pop(0)

        if scheduler is not None:
            scheduler.step()
    print("Training:", epoch_tracker.end())
    return epoch_tracker


def validate(model, test_loader: DataLoader, st: MLSettings) -> EpochTracker:
    epoch_tracker = EpochTracker(st.labels)
    epoch_tracker.begin()
    with torch.no_grad():
        for i, (data, y) in enumerate(test_loader):
            # print(ep, "Test")
            x = data[:,[2]].float()
            out = model(x)

            predicted = torch.argmax(out, dim=1, keepdim=False)  # -> [ label_indices ]
            if y.shape[0] == 2:  # batched
                correct = torch.argmax(y, dim=1, keepdim=False)  # -> [ label_indices ]
            else:  # unbatched
                correct = torch.argmax(y, dim=0, keepdim=True)  # -> [ label_indices ]

            epoch_tracker.add_prediction(correct, predicted)
    print("Validation:", epoch_tracker.end())
    return epoch_tracker


def train_validate_save(model, optimizer, scheduler, loss_func, train_loader: DataLoader, test_loader: DataLoader, st: MLSettings, models_dir, print_interval=1, print_continuous=False, show_plots=False, training_cancel_points=[]):
    # assumes model and data is already on correct device
    # train_loader.to(device)
    # test_loader.to(device)

    # store optimizer, scheduler and loss_func in settings
    st.optimizer = optimizer_str(optimizer)
    st.scheduler = class_str(scheduler)
    st.loss_func = class_str(loss_func)

    model_name = st.get_name()

    def add_tab(s):
        return "\t" + str(s).replace("\n", "\n\t")
    print(100 * '=')
    print("model name:", model_name)
    print(f"model:\n", add_tab(model))
    print(f"loss_func: {st.loss_func}")
    print(f"optimizer: {st.optimizer}")
    print(f"scheduler: {st.scheduler}")


    print(100 * '-')
    training_tracker = train(model, optimizer, scheduler, loss_func, train_loader, st, print_interval=print_interval, print_continuous=print_continuous, training_cancel_points=training_cancel_points)
    # print("Training: Count per label:", training_tracker.get_count_per_label())
    # print("Training: Predictions per label:", training_tracker.get_predictions_per_label())

    print(100 * '-')
    validation_tracker = validate(model, test_loader, st)
    # print("Validation: Count per label:", validation_tracker.get_count_per_label())
    # print("Validation: Predictions per label:", validation_tracker.get_predictions_per_label())


    digits = get_next_digits(f"{model_name}_", models_dir)
    model_dir = f"{models_dir}/{model_name}_{digits}"
    # do not put earlier, since the dir should not be created if training is interrupted
    if not path.isdir(model_dir):  # should always run, if not the digits function did not work
        makedirs(model_dir)

    fig, _ = validation_tracker.plot_predictions("Validation: Predictions", model_dir=model_dir, name="img_validation_predictions")
    fig, _ = training_tracker.plot_predictions("Training: Predictions", model_dir=model_dir, name="img_training_predictions")
    fig, _ = training_tracker.plot_training(model_dir=model_dir)

    if show_plots:
        plt.show()
    plt.close('all')

    # save the settings, results and model
    mio.save_settings(model_dir, st)
    mio.save_tracker_validation(model_dir, validation_tracker)
    mio.save_tracker_training(model_dir, training_tracker)
    mio.save_model(model_dir, model)
