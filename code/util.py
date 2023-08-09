from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torchvision.transforms as T

torch.manual_seed(3)


def load_images(img_folder: str) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    os.chdir(img_folder)
    files = os.listdir(".")
    files.remove("labels.txt")
    files = sorted(files)
    labels = []
    imgs = []
    with open("../data/labels.txt", "r") as f:
        for line in f:
            label = line.strip().split("\t")
            if len(label) != 1:
                labels.append(int(label[1]) - 1)
    for file in files:
        with Image.open(file) as f:
            imgs.append([np.asarray(f)])
    x, y = np.asarray(imgs), np.asarray(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=3)
    return x_train, x_test, y_train, y_test


def preprocess(imgs: torch.Tensor) -> torch.Tensor:
    mean = imgs.mean()
    std = imgs.std()
    normalize = T.Compose([  # best so far: with color jitter
        T.Normalize(mean, std),
        T.ColorJitter()
    ])
    return normalize(imgs)


class CNN(Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                               padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=5)
        self.fc1 = nn.Linear(in_features=256 * 10, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        # x = F.dropout2d(x, p=0.3)  # best so far is commented out
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.flat(x)
        x = self.fc1(x)

        # best so far is 0.75; 80% at epoch 419 with 3 linear layers
        # x = F.dropout(x, p=0.6)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
