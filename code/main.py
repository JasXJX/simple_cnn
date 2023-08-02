import util
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(3)


def train(features_src: np.ndarray, labels_src: np.ndarray, model, ep_n, lr):
    features, labels = util.preprocess(torch.asarray(features_src,
                                                     dtype=torch.float)), \
                       torch.asarray(labels_src, dtype=torch.long)
    train_set = TensorDataset(features, labels)
    train_loader = DataLoader(train_set, batch_size=5, shuffle=False)
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    avg_loss = []
    for ep in tqdm(range(ep_n), "Training..."):
        model.train()
        loss_vals = []
        for i, (inputs, expected) in enumerate(train_loader):
            optim.zero_grad()
            output = model(inputs)
            loss = loss_func(output, expected)
            loss.backward()
            optim.step()
            loss_vals.append(loss.item() * len(inputs))
        avg_loss.append(sum(loss_vals)/len(features))
    plt.plot(avg_loss)
    return model


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = util.load_images("data")
    model = util.CNN()
    model = train(x_train, y_train, model, ep_n=1000, lr=0.001)
