import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from time import time

from nodeclf.models import MLP
from nodeclf.util import accuracy
from nodeclf.util import plot_confusion_matrix


lr = 0.001
weight_decay = 5e-4

model = MLP(n_layers=6,
            c_in=features.shape[1],
            c_hidden=64,
            dropout=0.1,
            num_classes=labels.max() + 1)

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


def train_loop_mlp(epoch, writer):
    t_start = time.time()

    model.train()
    optimizer.zero_grad()
    output = model(features[train_idx])

    loss_train = F.nll_loss(output, labels[train_idx])
    acc_train = accuracy(output, labels[train_idx])

    loss_train.backward()
    optimizer.step()

    # check valid performance

    model.eval()
    output = model(features[valid_idx])

    loss_val = F.nll_loss(output, labels[valid_idx])
    acc_val = accuracy(output, labels[valid_idx])

    if (epoch % 50) == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t_start))

    writer.add_scalar("Loss/mlp-train", loss_train.item(), epoch)
    writer.add_scalar("Loss/mlp-valid", loss_val.item(), epoch)
    writer.add_scalar("Accuracy/mlp-train", acc_train.item(), epoch)
    writer.add_scalar("Accuracy/mlp-valid", acc_val.item(), epoch)


def train_mlp(epochs=100):
    t_start = time.time()

    writer = SummaryWriter(log_dir="logs")

    for epoch_i in tqdm(range(epochs)):
        train_loop_mlp(epoch_i, writer)

    print("Finished training in: {:.4f}s".format(time.time() - t_start))


train_mlp(epochs=1000)
