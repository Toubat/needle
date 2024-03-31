import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        self.conv2d = nn.Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            device=device,
            dtype=dtype,
        )
        self.bn = nn.BatchNorm2d(
            dim=out_channels,
            device=device,
            dtype=dtype,
        )
        self.relu = nn.ReLU()

    def forward(self, x: ndl.Tensor):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        layer1 = nn.Sequential(
            ConvBatchNorm(3, 16, 7, 4, device=device, dtype=dtype),
            ConvBatchNorm(16, 32, 3, 2, device=device, dtype=dtype),
        )
        res_block1 = nn.Residual(
            fn=nn.Sequential(
                ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype),
                ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype),
            )
        )
        layer2 = nn.Sequential(
            ConvBatchNorm(32, 64, 3, 2, device=device, dtype=dtype),
            ConvBatchNorm(64, 128, 3, 2, device=device, dtype=dtype),
        )
        res_block2 = nn.Residual(
            fn=nn.Sequential(
                ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype),
                ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype),
            )
        )

        self.conv_layer = nn.Sequential(
            layer1,
            res_block1,
            layer2,
            res_block2,
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.conv_layer(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        out = self.classifier(x)
        return out
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)