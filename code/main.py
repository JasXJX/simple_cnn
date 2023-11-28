import util
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(3)
if torch.cuda.is_available() :
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def train(features_src: np.ndarray, labels_src: np.ndarray, model, ep_n, lr):
    # features, labels = util.preprocess(torch.asarray(features_src,
    #                                                  dtype=torch.float,
    #                                                  device=device)), \
    #                    torch.asarray(labels_src,
    #                                  dtype=torch.long,
    #                                  device=device)

    features, labels = torch.asarray(features_src,
                                     dtype=torch.float,
                                     device=device), \
                       torch.asarray(labels_src,
                                     dtype=torch.long,
                                     device=device)

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
        # tqdm.write(f"Training Accuracy: {test(features, labels, model)}")
    plt.plot(avg_loss)
    plt.show()
    return model


def test(inputs, labels, model) -> float:
    model.eval()
    # labels = labels.to(torch.device('cpu'))
    # pred = model(inputs).to(torch.device('cpu'))

    pred = model(inputs)
    _, pred_class = torch.max(pred, dim=-1)
    acc = torch.sum(pred_class == labels).item() / len(labels)
    return acc


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = util.load_images("data")
    model = util.CNN()
    model.to(device)
    model = train(x_train, y_train, model, ep_n=500, lr=0.0001)
    x_test_t, y_test_t = torch.asarray(x_test, dtype=torch.float).to(device), \
        torch.asarray(y_test, dtype=torch.long).to(device)
    print(f"Test set score: {test(x_test_t, y_test_t, model)}")
