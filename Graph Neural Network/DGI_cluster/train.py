import time
import argparse
import sys
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
from sklearn.cluster import KMeans

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


def data_augment(feat,cluster):
    '''
    label이 동일한 group끼리 corrupt하는 함수
    '''
    for num in range(n_class):
        df_feature = pd.DataFrame(feat.numpy())
        df_feature['label'] = cluster
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


loss_list=[]

#if args.augmentation ==True:
#    sys.stdout = open(f'results/{dataset}/{args.epochs} epoch_{args.hidden} hidden units_log(augmentation).txt', 'w')
#else:
#    sys.stdout = open(f'results/{dataset}/{args.epochs} epoch_{args.hidden} hidden units_log.txt', 'w')

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

def clustering_node():
    embeds  = model.embed(features,adj)
    df_embeds = pd.DataFrame(embeds[0].numpy())
    kmeans = KMeans(n_clusters=n_class, random_state = 2)
    km_cluster = kmeans.fit_predict(df_embeds)
    return km_cluster

def train_augment(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
     # Corrupt input graph (shuffling for row)
    corrupted_idx = np.random.permutation(n_nodes)
    corrupted_features = features[corrupted_idx, :]
    if args.augmentation == True:
        if epoch==0:
            print("Augmentation completed!")
        augment_features = data_augment(features,clusters_for_augment)
        output = model(augment_features, corrupted_features, adj)
    else:
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


classify_loss_list=[]
classify_val_loss_list=[]

    
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

    for times in range(50):
        classifier = LogisticRegression(args.hidden,n_class)
        optimizer = optim.Adam(classifier.parameters(), lr = 0.01)
        
        
        for _ in range(200):
            classifier.train()
            optimizer.zero_grad()
            logits = classifier(train_embs)
            CE= nn.CrossEntropyLoss()
            class_loss_train = CE(logits, train_labels)
            class_loss_train.backward()
            optimizer.step()
            if times == 49:
                classify_loss_list.append(class_loss_train)

            classifier.eval()
            logits = classifier(val_embs)
            class_loss_val = CE(logits, val_labels)
            if times == 49:
                classify_val_loss_list.append(class_loss_val)        

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
    plt.figure()
    style.use("ggplot")
    plt.plot(loss_list, linewidth=2.0, color="red", label="BCEloss")
    plt.legend()
    #plt.show()
    if args.augmentation ==True:
        plt.savefig(f"results/{dataset}/{args.epochs} epoch_{args.hidden} hidden units loss(augmentation).png")
    else:
        plt.savefig(f"results/{dataset}/{args.epochs} epoch_{args.hidden} hidden units loss.png")
  
  
def visualize_classification():
    if not os.path.exists(f"results/{dataset}/classification"):
        os.makedirs(f"results/{dataset}/classification")
    style.use("ggplot")
    plt.figure()
    plt.plot(classify_loss_list, linewidth=2.0, color="blue", label="train_BCEloss")
    plt.plot(classify_val_loss_list, linewidth=2.0, color="red", label="val_BCEloss")
    plt.legend()
    #plt.show()
    if args.augmentation == True:
        #augment
        plt.savefig(f"results/{dataset}/classification/{args.epochs} epoch_{args.hidden} hidden units_loss(augmentation).png")
    else:
        plt.savefig(f"results/{dataset}/classification/{args.epochs} epoch_{args.hidden} hidden units_loss.png")


if __name__ =="__main__":

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    loss_list = []

    clusters_for_augment = clustering_node()
    model = DGI(nfeat=features.shape[1],
            nhid=args.hidden)
    for epoch in range(args.epochs):
        train_augment(epoch)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    classification_test()
    visualize() # representation loss png save
    embeds  = model.embed(features,adj)
    DGI.visualize_feature1(embeds, labels, args.dataset, args.epochs, args.hidden, args.augmentation) #pca save
    DGI.visualize_feature2(embeds, labels, args.dataset, args.epochs, args.hidden, args.augmentation) #t-sne save
    #sys.stdout.close()
    visualize_classification()