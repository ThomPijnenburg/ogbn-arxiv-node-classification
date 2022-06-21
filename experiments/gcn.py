import time
import matplotlib.pyplot as plt

# Progress bar
from tqdm.notebook import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

# OGB
from ogb.nodeproppred import NodePropPredDataset

# sklearn
from sklearn import metrics

from nodeclf.models import GCN
from nodeclf.util import plot_confusion_matrix
from nodeclf.util import accuracy


# HYPERPARAMS

hidden_dim = 32
lr = 0.01
weight_decay = 5e-4
epochs = 100


model = GCN(c_in=features.shape[1],
            c_hidden=hidden_dim,
            c_out=labels.max() + 1)

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)


if torch.cuda.is_available():
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    train_idx = train_idx.cuda()
    valid_idx = valid_idx.cuda()
    test_idx = test_idx.cuda()


def train_loop(epoch, writer):
    t_start = time.time()

    model.train()
    optimizer.zero_grad()
    output = model(features, adj)

    loss_train = F.nll_loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])

    loss_train.backward()
    optimizer.step()

    # check valid performance

    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[valid_idx], labels[valid_idx])
    acc_val = accuracy(output[valid_idx], labels[valid_idx])

    if (epoch % 50) == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t_start))

    writer.add_scalar("Loss/gcn-train", loss_train.item(), epoch)
    writer.add_scalar("Loss/gcn-valid", loss_val.item(), epoch)
    writer.add_scalar("Accuracy/gcn-train", acc_train.item(), epoch)
    writer.add_scalar("Accuracy/gcn-valid", acc_val.item(), epoch)


def test():
    model.eval()
    output = model(features, adj)

    loss_test = F.nll_loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    print("Test set results: loss= {:.4f}, accuracy= {:.4f}".format(
        loss_test.item(), acc_test.item()))

    plot_confusion_matrix(labels[test_idx].cpu(), torch.argmax(
        output[test_idx], dim=1).squeeze().cpu().detach().numpy())


def train_gcn(epochs: 100):
    t_start = time.time()

    writer = SummaryWriter(log_dir="logs")

    for epoch_i in tqdm(range(epochs)):
        train_loop(epoch_i, writer)

    print("Finished training in: {:.4f}s".format(time.time() - t_start))
