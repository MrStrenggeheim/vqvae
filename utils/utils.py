import time

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.amos import AmosDataset


def load_amos(dconfig):
    train = AmosDataset(
        dconfig["path"],
        "train",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(dconfig["image_size"]),
                transforms.CenterCrop(dconfig["image_size"]),
                transforms.Normalize(0.5, 0.5),
            ]
        ),
        index_range=dconfig["index_range"],
        slice_range=dconfig["slice_range"],
        only_labeled=dconfig["only_labeled"],
    )

    val = AmosDataset(
        dconfig["path"],
        "val",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(dconfig["image_size"]),
                transforms.CenterCrop(dconfig["image_size"]),
                transforms.Normalize(0.5, 0.5),
            ]
        ),
        index_range=dconfig["index_range"],
        slice_range=dconfig["slice_range"],
        only_labeled=dconfig["only_labeled"],
    )

    train_loader = DataLoader(
        train,
        batch_size=dconfig["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=16,
    )
    val_loader = DataLoader(
        val,
        batch_size=dconfig["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=16,
    )

    return train, val, train_loader, val_loader


def calc_x_train_var(train_data):
    print("Calculating x_train_var")
    all_images = []
    for img, _ in train_data:
        img = img / 255.0
        all_images.append(img.flatten())
    all_images = np.concatenate(all_images)
    return np.var(all_images)


def readable_timestamp():
    return time.ctime().replace("  ", " ").replace(" ", "_").replace(":", "_").lower()
