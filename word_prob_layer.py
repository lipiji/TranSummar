# -*- coding: utf-8 -*-
#pylint: skip-file
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils_pg import *
from transformer import MultiheadAttention

class WordProbLayer(nn.Module):
    def __init__(self, hidden_size, dict_size, device, copy, coverage, dropout):
        super(WordProbLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.device = device
        self.copy = copy
        self.coverage = coverage
        self.dropout = dropout
    
        if self.copy:
            self.external_attn = MultiheadAttention(self.hidden_size, 1, self.dropout, weights_dropout=False)
            self.proj = nn.Linear(self.hidden_size * 3, self.dict_size)
            self.v = nn.Parameter(torch.Tensor(1, self.hidden_size * 3))
            self.bv = nn.Parameter(torch.Tensor(1))
        else:
            self.proj = nn.Linear(self.hidden_size, self.dict_size)
        
        self.init_weights()

    def init_weights(self):
        init_linear_weight(self.proj)
        if self.copy:
            init_xavier_weight(self.v) 
            init_bias(self.bv)

    def forward(self, h, y_emb=None, memory=None, mask_x=None, xids=None, max_ext_len=None):

        if self.copy:
            atts, dists = self.external_attn(query=h, key=memory, value=memory, key_padding_mask=mask_x, need_weights = True)
            pred = T.softmax(self.proj(T.cat([h, y_emb, atts], -1)), dim=-1)
            if max_ext_len > 0:
                ext_zeros = Variable(torch.zeros(pred.size(0), pred.size(1), max_ext_len)).to(self.device)
                pred = T.cat((pred, ext_zeros), -1)
            g = T.sigmoid(F.linear(T.cat([h, y_emb, atts], -1), self.v, self.bv))
            xids = xids.transpose(0, 1).unsqueeze(0).repeat(pred.size(0), 1, 1) 
            pred = (g * pred).scatter_add(2, xids, (1 - g) * dists)
        else:
            pred = T.softmax(self.proj(h), dim=-1)
            dists = None
        return pred, dists
