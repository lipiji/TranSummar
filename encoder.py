import sys
import numpy as np
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable

from utils_pg import *
from transformer.layers import MultiHeadedAttention, PositionwiseFeedForward, EncoderLayer
from transformer.model import Encoder

class LocalEncoder(nn.Module):
    def __init__(self, w_emb, p_emb, d_model=256, d_ff=1024, h=8, dropout=0.1, N=6):
        super(LocalEncoder, self).__init__()  
        self.w_emb = w_emb
        self.p_emb = p_emb
        self.dropout = nn.Dropout(dropout)

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), N)
        
    def forward(self, x, p, mask_x):
        emb_x = self.w_emb(x)
        emb_p = self.p_emb(p)
        return self.encoder(self.dropout(emb_x + emb_p), mask_x)  


