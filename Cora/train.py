from __future__ import division
from __future__ import print_function
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data,accuracy
from tree_utils import createDict
from model import FewGatModel


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during pass')
parser.add_argument('--seed', type=int, default=72, help='Random seed')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay(L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate(1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for leaky_relu')
parser.add_argument('--patience', type=int, default=100, help='Patience')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load data
adj, features, labels, temp_labels, idx_train, idx_val, idx_test = load_data('data/cora/', 'cora')
weight_dict = createDict('data/cora/cora.content', 'data/cora/cora.cites')

model = FewGatModel(feat_dim=features.shape[1],hidden_dim=args.hidden,num_class=int(labels.max())+1,dropout=args.dropout,
                         alpha=args.alpha,weight_dict=weight_dict,concat=True)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    weight_dict = weight_dict.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train_all = loss_train
    loss_train_all.backward()
    optimizer.step()
    if not args.fastmode:
        model.eval()
        output = model(features,adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item(), acc_val.data.item(), loss_train_all.data.item(), acc_train.data.item()

def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    return acc_test.data.item()

if __name__ == '__main__':
    t_total = time.time()
    loss_values = []
    acc_values = []
    loss_trains = []
    acc_trains = []
    bad_counter = 0
    best = 10000
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_value, acc_value, loss_train, acc_train = train(epoch)
        loss_values.append(loss_value)
        loss_trains.append(loss_train)
        acc_trains.append(acc_train)
        acc_values.append(acc_value)
        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
            if bad_counter == args.patience:
                break
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('Loading {}th epoch'.format(best_epoch))

    # Testing
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    acc_test = compute_test()
