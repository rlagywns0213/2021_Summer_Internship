import torch.nn as nn
from layers import Discriminator, Readout, GCN


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

