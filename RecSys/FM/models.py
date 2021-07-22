import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    """
    Reference
    : https://github.com/Namkyeong/RecSys_paper/blob/main/FactorizationMachine/FactorizationMachine_matrix.ipynb
    """
    def __init__(self, field_dims, latent_dims):
        super(FactorizationMachine, self).__init__()
        
        self.w_0 = nn.Parameter(nn.init.normal_(torch.zeros((1, ))), requires_grad=True) 
        self.w_i = nn.Parameter(nn.init.normal_(torch.zeros((1, field_dims)), std=1.0/field_dims), requires_grad = True)
        self.v_ij = nn.Parameter(nn.init.normal_(torch.zeros((field_dims, latent_dims)), std=1.0/latent_dims), requires_grad = True) 
        
        
    def forward(self, x):
 
        temp_1 = self.w_0 + torch.matmul(x, self.w_i.T) 
        
        square_of_sum = torch.sum(torch.matmul(x, self.v_ij), dim = 1) ** 2
        sum_of_square = torch.sum(torch.matmul(x, self.v_ij) ** 2, dim = 1)
        temp_2 = (square_of_sum - sum_of_square).view(-1, 1)
        
        result = temp_1 + 0.5 * temp_2
        
        return result
    
    def init_weight(self):
        pass
