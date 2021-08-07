from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, load_data2, accuracy
from models import GCN



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', required = False, default='cora',
                    help='dataset name')
parser.add_argument('--augmentation',action='store_true', default=False,
                    help='for Augmentation')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = args.dataset
if dataset == 'cora':
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data2(dataset)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

def data_augment(feat,idx):
    '''
    label이 동일한 group끼리 corrupt하는 함수
    data leakage 문제를 해결해주기 위해 train에만 적용!!
    '''
    n_class = labels[idx].max().item()+1 #0부터 시작하므로

    for num in range(n_class):
        df_feature = pd.DataFrame(feat.numpy())
        df_feature['label'] = labels[idx] #train 에만 적용!!!
        labeld_df = df_feature[df_feature['label'].apply(lambda x: x==num)]
        shuffled_df = labeld_df.sample(frac=1)
        shuffled_df.index = labeld_df.index
        if num ==0:
            all_df = shuffled_df
        else:
            all_df = pd.concat([all_df, shuffled_df])
    all_df.sort_index(inplace=True)
    all_df.drop(columns='label', inplace=True)
    return torch.tensor(np.array(all_df))

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

loss_list=[]
val_loss_list=[]

if args.augmentation ==True:
    sys.stdout = open(f'results/{dataset}/{args.epochs} epoch_{args.hidden} hidden units_log(augmentation).txt', 'w')
else:
    sys.stdout = open(f'results/{dataset}/{args.epochs} epoch_{args.hidden} hidden units_log.txt', 'w')
    
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if args.augmentation == True:
        #augment
        augment_features = data_augment(features[idx_train],idx_train)
        features[idx_train] = augment_features #feature train idx만 augment 적용해서 대체!
        # augment_features2 = data_augment(features[idx_val],idx_val)
        # features[idx_val] = augment_features2 #feature train idx만 augment 적용해서 대체!
        if epoch ==0:
            print("Augmentation completed!")
        output = model(features, adj)
    else:
        output = model(features, adj)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss_list.append(loss_train)
    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)
    model.eval()
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    val_loss_list.append(loss_val)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(f"{dataset} Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

def visualize():
    if not os.path.exists(f"results/{dataset}"):
        os.makedirs(f"results/{dataset}")
    else:
        print("exist")
    style.use("ggplot")
    plt.plot(loss_list, linewidth=2.0, color="blue", label="train_BCEloss")
    plt.plot(val_loss_list, linewidth=2.0, color="red", label="val_BCEloss")
    plt.legend()
    #plt.show()
    if args.augmentation == True:
        #augment
        plt.savefig(f"results/{dataset}/{args.epochs} epoch_{args.hidden} hidden units_loss(augmentation).png")
    else:
        plt.savefig(f"results/{dataset}/{args.epochs} epoch_{args.hidden} hidden units_loss.png")

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

#loss
visualize()
sys.stdout.close()