# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.vim import VMamba, MambaConfig
torch.manual_seed(42)
   
class MambaCross(nn.Module):
    def __init__(self, hor_dim, ver_dim, feat_dim, seq_len, hidden_sizes,
                 mamba_layer, pooling='avg', activation='SiLU', drop_ratio=0.1):
        super(MambaCross, self).__init__()
        self.W = nn.Parameter(torch.randn(feat_dim, feat_dim))  
        
        # -------mamba_encoder
        self.config_hor = MambaConfig(d_model=hor_dim, expand_factor=1, n_layers=mamba_layer)
        self.config_ver = MambaConfig(d_model=ver_dim, expand_factor=1, n_layers=mamba_layer)
        self.mamba_hor = VMamba(self.config_hor)
        self.mamba_ver = VMamba(self.config_ver)
        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("Unsupported pooling function")
        
        # ------predict_decoder
        if activation == 'SiLU':
            self.act = F.silu
        elif activation == 'Leaky':
            self.act = nn.LeakyReLU(0.1)
        elif activation == 'Tanh':
            self.act = F.tanh
        else:
            self.act = F.relu

        self.hidden_layers = nn.ModuleList()
        prev_size = seq_len
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, 1)
        self.r = drop_ratio
        self.reset_para()
    
    def reset_para(self):
        nn.init.xavier_uniform_(self.W) 
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, x_Ab, x_Ag):
        #-----Mamba fusion
        contacts = torch.matmul(torch.matmul(x_Ab, self.W), x_Ag.transpose(1, 2))
        x_Ab = self.mamba_hor(contacts)
        x_Ag = self.mamba_ver(contacts.transpose(1, 2))     
        x_Ab = self.pool(x_Ab)
        x_Ag = self.pool(x_Ag)
        x = torch.cat([x_Ab.squeeze(-1), x_Ag.squeeze(-1)], dim=-1)
        
        #------MLP decoder
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x = self.act(layer(x))
                x = torch.dropout(x, self.r, train=False)
        x = torch.squeeze(self.output_layer(x))  # last layer
        
        return torch.sigmoid(x)
