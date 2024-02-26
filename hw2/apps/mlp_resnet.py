import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
from needle.data import MNISTDataset, DataLoader
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
      nn.Residual(
        fn=nn.Sequential(
          nn.Linear(dim, hidden_dim),
          norm(hidden_dim),
          nn.ReLU(), 
          nn.Dropout(drop_prob),
          nn.Linear(hidden_dim, dim),
          norm(dim),
        )
      ),
      nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(
              dim=hidden_dim, 
              hidden_dim=hidden_dim // 2, 
              norm=norm, 
              drop_prob=drop_prob,
            ) for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    
    criterion = nn.SoftmaxLoss()
    num_examples, avg_loss, avg_error = 0, 0., 0.
    for i, batch in enumerate(dataloader):
        if opt:
            opt.reset_grad()

        x, y = batch # (n, 784), (n,)
        logits = model(x) # (n, num_classes)
        loss = criterion(logits, y)

        avg_loss += float(loss.numpy()) * x.shape[0]
        avg_error += (np.argmax(logits.numpy(), axis=1) != y.numpy()).sum()
        num_examples += x.shape[0]

        if opt:
            loss.backward()
            opt.step()

    return avg_error / num_examples, avg_loss / num_examples
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model = MLPResNet(dim=28 * 28, hidden_dim=hidden_dim, num_classes=10)
    optim = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(train_set, batch_size, shuffle=False)

    for i in range(epochs):
        train_err, train_loss = epoch(train_loader, model, optim)
    
    test_err, test_loss = epoch(test_loader, model, None)

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
