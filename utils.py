import os
import random

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tonic
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from spikingjelly.activation_based import functional
from torch.utils.data import Dataset
from tqdm import tqdm


class DVSLip_Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        class_subset=None,
        transform=None,
        target_transform=None,
        train=True,
        T=30,
    ):
        self.classes_to_keep = class_subset
        self.target_transform = target_transform
        self.T = T

        self.dataset = tonic.datasets.DVSLip(
            save_to=dataset_path, train=train, transform=transform
        )
        targs = np.array(self.dataset.targets)
        data = np.array(self.dataset.data)

        # class_subset allows to specify which classes to keep if we whant to use only part of the dataset
        if class_subset is not None:
            idx = []
            for c in class_subset:
                idx += list(np.where(targs == c)[0])
            idx = np.array(idx)
            targs = targs[idx]
            data = data[idx]

            cpt = 0
            for c in class_subset:
                targs = np.where(targs == c, cpt, targs)
                cpt += 1
            self.dataset.targets = targs
            self.dataset.data = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]

        with torch.no_grad():
            # Initialize voxel grid
            voxel_grid = torch.zeros(self.T, 88, 88, dtype=torch.float32)
            voxel_grid = voxel_grid.flatten()

            # normalize the event timestamps so that they lie between 0 and T
            last_stamp = data["t"][-1]
            first_stamp = data["t"][0]
            deltaT = float(last_stamp - first_stamp)
            if deltaT == 0:
                deltaT = 1.0
            data["t"] = (self.T - 1) * (data["t"] - first_stamp) / deltaT
            ts = torch.from_numpy(data["t"].copy())
            xs = torch.from_numpy(data["x"].copy()).long()
            ys = torch.from_numpy(data["y"].copy()).long()
            pols = torch.from_numpy(data["p"].copy()).float()

            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < self.T
            valid_indices &= tis >= 0
            voxel_grid.index_add_(
                dim=0,
                index=xs[valid_indices]
                + ys[valid_indices] * 88
                + tis_long[valid_indices] * 88 * 88,
                source=vals_left[valid_indices],
            )

            valid_indices = (tis + 1) < self.T
            valid_indices &= tis >= 0

            voxel_grid.index_add_(
                dim=0,
                index=xs[valid_indices]
                + ys[valid_indices] * 88
                + (tis_long[valid_indices] + 1) * 88 * 88,
                source=vals_right[valid_indices],
            )

            voxel_grid = voxel_grid.view(self.T, 1, 88, 88)

        data = voxel_grid

        if self.target_transform:
            target = self.target_transform(target)

        return data, target


# Used to load and preprocess the i3s dataset, as in Marcel's code from https://github.com/marceey/MSTP_on_i3s.git
class i3s_Dataset(Dataset):
    def __init__(
        self, dataset_path, transform=None, target_transform=None, train=True, T=30
    ):
        self.target_transform = target_transform
        self.transform = transform
        self.length = T
        self.train = train

        self.class_label = {}
        self.files = []
        self.file_labels = []
        class_cpt = 0
        self.data_dir = dataset_path + "/train/" if train else dataset_path + "/test/"

        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f[-4:] == ".npy":
                    class_name = root.split("/")[-1]
                    if self.class_label.get(class_name) is None:
                        self.class_label[class_name] = class_cpt
                        class_cpt += 1
                    self.files.append(os.path.join(root, f))
                    self.file_labels.append(self.class_label[class_name])
                    print(os.path.join(root, f), self.class_label[class_name])

        print("Number of classes found:", len(self.class_label.values()))

    def __len__(self):
        return len(self.files)

    def events_to_voxel_grid_pytorch(self, events, num_bins, width, height, device):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        assert events.shape[1] == 4
        assert num_bins > 0
        assert width > 0
        assert height > 0

        with torch.no_grad():
            events_torch = torch.from_numpy(events).float()
            events_torch = events_torch.to(device)

            voxel_grid = torch.zeros(
                num_bins, height, width, dtype=torch.float32, device=device
            )
            if events_torch.shape[0] == 0:
                return voxel_grid

            voxel_grid = voxel_grid.flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]
            deltaT = float(last_stamp - first_stamp)

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (
                (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            )
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(
                dim=0,
                index=xs[valid_indices]
                + ys[valid_indices] * width
                + tis_long[valid_indices] * width * height,
                source=vals_left[valid_indices],
            )

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(
                dim=0,
                index=xs[valid_indices]
                + ys[valid_indices] * width
                + (tis_long[valid_indices] + 1) * width * height,
                source=vals_right[valid_indices],
            )

            voxel_grid = voxel_grid.view(num_bins, height, width)
        return voxel_grid

    def events_to_voxel_all(
        self, events, seq_len, num_bins, width, height, device
    ):  # frame_nums
        voxel_len = seq_len * num_bins  # min(seq_len, frame_nums)
        voxel_grid_all = np.zeros((num_bins * seq_len, 1, height, width))
        voxel_grid = self.events_to_voxel_grid_pytorch(
            events, voxel_len, width, height, device
        )
        voxel_grid = voxel_grid.unsqueeze(1).cpu().numpy()
        voxel_grid_all[:voxel_len] = voxel_grid
        return voxel_grid_all

    def CenterCrop(self, event_low, size):
        w, h = event_low.shape[-1], event_low.shape[-2]
        th, tw = size
        x1 = int(round((w - tw)) / 2.0)
        y1 = int(round((h - th)) / 2.0)
        event_low = event_low[..., y1 : y1 + th, x1 : x1 + tw]

        return event_low

    def RandomCrop(self, event_low, size):
        th, tw = size
        x1 = random.randint(0, 8)
        y1 = random.randint(0, 8)
        event_low = event_low[..., y1 : y1 + th, x1 : x1 + tw]
        return event_low

    def HorizontalFlip(self, event_low):
        if random.random() > 0.5:
            event_low = np.ascontiguousarray(event_low[..., ::-1])
        return event_low

    def __getitem__(self, idx):
        # data = np.load(self.files[idx])
        target = self.file_labels[idx]
        events_input = np.load(self.files[idx])

        events_input = events_input[
            np.where(
                (events_input[:, 0] >= 58)
                & (events_input[:, 0] < 154)
                & (events_input[:, 1] >= 32)
                & (events_input[:, 1] < 128)
            )
        ]  # 16, 112, 16, 112
        events_input[:, 0] -= 58  # 16
        events_input[:, 1] -= 32  # 16

        t, x, y, p = (
            events_input[:, 3],
            events_input[:, 0],
            events_input[:, 1],
            events_input[:, 2],
        )
        events_input = np.stack([t, x, y, p], axis=-1)

        # convert events to voxel_grid
        event_voxel_low = self.events_to_voxel_all(
            events_input, self.length, 1, 96, 96, device="cpu"
        )  # (30*num_bins[0], 96, 96)
        # event_voxel_high = self.events_to_voxel_all(events_input, self.length, self.args.num_bins[1], 96, 96, device='cpu') # (30*num_bins[1], 96, 96)

        # data augmentation
        if self.train:
            event_voxel_low = self.RandomCrop(
                event_voxel_low, (88, 88)
            )  # event_voxel_high, (88, 88)
            event_voxel_low = self.HorizontalFlip(event_voxel_low)
        else:
            event_voxel_low = self.CenterCrop(
                event_voxel_low, (88, 88)
            )  # event_voxel_high, (88, 88)

        data = torch.FloatTensor(event_voxel_low)
        return data, target


# Does a center crop from (128, 128) to (96, 96), then a random crop to (88, 88), and flips the image horizontaly with a probability of 0.5
def center_random_crop(events):
    transform = tonic.transforms.Compose(
        [
            tonic.transforms.CenterCrop(
                sensor_size=tonic.datasets.DVSLip.sensor_size, size=(96, 96, 2)
            ),
            tonic.transforms.RandomCrop(
                sensor_size=(96, 96, 2), target_size=(88, 88, 2)
            ),
            tonic.transforms.RandomFlipLR(sensor_size=(88, 88, 2), p=0.5),
        ]
    )
    return transform(events)


# Only does a center crop from (128, 128) to (88, 88)
def center_crop(events):
    transform = tonic.transforms.Compose(
        [
            tonic.transforms.CenterCrop(
                sensor_size=tonic.datasets.DVSLip.sensor_size, size=(88, 88, 2)
            ),
        ]
    )
    return transform(events)


# Train for one epoch
def train(model, device, train_loader, optimizer, num_labels=70, scheduler=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.transpose(0, 1).float().to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data).mean(0)

        label_onehot = torch.nn.functional.one_hot(target, num_labels).float()
        loss = torch.nn.functional.cross_entropy(output, label_onehot)
        loss.backward()

        optimizer.step()

        functional.reset_net(model)
        model.clamp_parameters()

        train_loss += loss.item()
        total += target.size(0)
        correct += (output.argmax(1) == target).sum().item()

    if scheduler is not None:
        scheduler.step()

    accuracy = 100.0 * correct / total
    train_loss /= i

    return train_loss, accuracy


def test(model, device, test_loader, num_labels=70):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.transpose(0, 1).float().to(device), target.to(device)

            output = model(data).mean(0)

            label_onehot = torch.nn.functional.one_hot(target, num_labels).float()
            loss = torch.nn.functional.cross_entropy(output, label_onehot)

            functional.reset_net(model)

            test_loss += loss.item()
            total += target.size(0)
            correct += (output.argmax(1) == target).sum().item()

    accuracy = 100.0 * correct / total
    test_loss /= i

    return test_loss, accuracy


def model_memory_usage(model):
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    return mem


# Generates a video from voxel grid
def generate_video(data):
    img = []
    frames = []
    fig = plt.figure()
    for i in range(len(data)):
        frames.append([plt.imshow(data[i], cmap=cm.Greys_r, animated=True)])

    ani = animation.ArtistAnimation(
        fig, frames, interval=50, blit=True, repeat_delay=1000
    )
    plt.show()
