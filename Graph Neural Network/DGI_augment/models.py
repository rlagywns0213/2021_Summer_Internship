import torch.nn as nn
from layers import Discriminator, Readout, GCN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


class DGI(nn.Module):
    def __init__(self, nfeat, nhid):
        super(DGI, self).__init__()

        self.gcn = GCN(nfeat, nhid)
        # self.out = nn.Linear(nclass, nclass)
        self.readout = Readout()
        self.sigmoid = nn.Sigmoid()
        self.discriminator = Discriminator(nhid)

    def forward(self, x_1, x_2, adj):
        h_1 = self.gcn(x_1, adj) #input gcn 통과
        h_2 = self.gcn(x_2, adj) #corrupt input gcn 통과

        s = self.readout(h_1) #input에 대해 global feature 추출
        s = self.sigmoid(s) # readout 활성화함수
        score = self.discriminator(s, h_1, h_2) #score반환
        return score

     # Detach the return variables
    def embed(self,h, adj):
        h = self.gcn(h, adj)
        s = self.readout(h)
        return h.detach()

####################################PCA Representation########################################################
    def visualize_feature1(embeds,labels,dataset, epoch, hidden, aug):
        x_coords = []
        y_coords = []
        colors = []
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeds[0])
        for x,y in pca_result:
                x_coords.append(x)
                y_coords.append(y)
            
        color = ["darkviolet", "yellow", "aqua", "silver","springgreen","pink","darkred"]
        labels_list = list(np.array(labels))
        for i in labels_list:
            colors.append(color[i])
        plt.figure()
        plt.scatter(x_coords, y_coords, c=colors, s=5)
        plt.title(f"Representations of Graph_epoch:{epoch}")
        if aug ==True:
            plt.savefig(f"results/{dataset}/aug_representations_PCA_{epoch} epoch_{hidden} hidden units.png")
        else:
            plt.savefig(f"results/{dataset}/representations_PCA_{epoch} epoch_{hidden} hidden units.png")

        print("PCA representation png saved!!")


####################################T-SNE Representation#######################################################
    def visualize_feature2(embeds,labels,dataset, epoch, hidden, aug):
        x_coords = []
        y_coords = []
        colors = []
        tsne = TSNE(n_components=2)
        tsne_x  = tsne.fit_transform(embeds[0])
        for x,y in tsne_x :
                x_coords.append(x)
                y_coords.append(y)
            
        color = ["darkviolet", "yellow", "aqua", "silver","springgreen","pink","darkred"]
        labels_list = list(np.array(labels))
        for i in labels_list:
            colors.append(color[i])
        plt.figure()
        plt.scatter(x_coords, y_coords, c=colors, s=5)
        plt.title(f"Representations of Graph_epoch:{epoch}")
        if aug ==True:
            plt.savefig(f"results/{dataset}/aug_representations_TSNE_{epoch} epoch_{hidden} hidden units.png")
        else:
            plt.savefig(f"results/{dataset}/representations_TSNE_{epoch} epoch_{hidden} hidden units.png")

        print("TSNE representation png saved!!")

