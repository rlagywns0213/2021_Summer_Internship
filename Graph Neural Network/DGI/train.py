import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from utils import load_data, load_data2 
from models import DGI
from layers import LogisticRegression
import yaml
import os

# config
with open('configuration.yaml') as f:
  configuration = yaml.load(f)

epochs = configuration['epochs']
learning_rate = configuration['learning_rate']
weight_decay = configuration['weight_decay']
hidden = configuration['hidden']

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=epochs,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=learning_rate,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=weight_decay,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=hidden,
                    help='Number of hidden units.')
parser.add_argument('--dataset', required = False, default='BlogCatalog',
                    help='dataset name')


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

n_class = labels.max().item()+1 #0부터 시작하므로
n_nodes = features.shape[0]
# Model and optimizer
model = DGI(nfeat=features.shape[1],
            nhid=args.hidden)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

loss_list=[]
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
     # Corrupt input graph (shuffling for row)
    corrupted_idx = np.random.permutation(n_nodes)
    corrupted_features = features[corrupted_idx, :]
    output = model(features, corrupted_features, adj)
    target = torch.cat([torch.ones(n_nodes), torch.zeros(n_nodes)]).unsqueeze(0)

    BCE = nn.BCEWithLogitsLoss()
    loss_train = BCE(output, target)
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    loss_list.append(loss_train)


def classification_test():
    #node classification
    embeds  = model.embed(features,adj)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0,  idx_val]
    test_embs = embeds[0, idx_test]

    train_labels = labels[idx_train]
    val_labels = labels[idx_val]
    test_labels = labels[idx_test]

    accuracy = 0.0
    print("classification Start")

    for _ in range(50):
        classifier = LogisticRegression(args.hidden,n_class)
        optimizer = optim.Adam(classifier.parameters(), lr = 0.01)

        for _ in range(200):
            classifier.train()
            optimizer.zero_grad()
            logits = classifier(train_embs)
            CE= nn.CrossEntropyLoss()
            loss = CE(logits, train_labels)
            loss.backward()
            optimizer.step()
        
        logits = classifier(test_embs)
        pred_labels = torch.argmax(logits, dim=1)
        accuracy += torch.sum(pred_labels == test_labels)/ test_labels.shape[0]
    accuracy /= 50
    print(f"Accuracy: {accuracy}")



def visualize():
    if not os.path.exists(f"results/{dataset}"):
        os.makedirs(f"results/{dataset}")
    else:
        print("exist")
    style.use("ggplot")
    plt.plot(loss_list, linewidth=2.0, color="red", label="BCEloss")
    plt.legend()
    plt.show()
    plt.savefig(f"results/{dataset}/loss(normalization).png")
  
if __name__ =="__main__":

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
        
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    classification_test()
    visualize()