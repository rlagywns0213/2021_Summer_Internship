import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GCN(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(GCN, self).__init__()
        self.fc = nn.Linear(n_features, n_hidden)
        self.activ = nn.PReLU() 
        self.init_weight()

    def init_weight(self) :
        torch.nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, x, adj):
        x = self.fc(x)
        x = torch.unsqueeze(torch.spmm(adj, torch.squeeze(x, 0)), 0)
        x = self.activ(x)
        return x

# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj):
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             output = nn.PReLU(output)
#             return output

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, h):
        return torch.mean(h, dim=1)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_1, h_2, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_1) #global feature
        sc_1 = torch.squeeze(self.f_k(h_1, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_2, c_x), 2)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat([sc_1, sc_2],dim=1)

        return logits


class LogisticRegression(nn.Module) :
    def __init__(self, n_hidden, n_class) :
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(n_hidden, n_class)
        self.init_weight()

    def init_weight(self) :
        torch.nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, h) :
        return self.fc(h)
