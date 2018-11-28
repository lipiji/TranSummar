import numpy as np
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable

from utils_pg import *
from transformer.layers import MultiHeadedAttention, PositionwiseFeedForward, EncoderLayer
from transformer.model import Encoder

class LocalEncoder(nn.Module):
    def __init__(self, w_emb, p_emb_w, p_emb_s, d_model=256, d_ff=1024, h=8, dropout=0.1, N=6):
        super(LocalEncoder, self).__init__()  
        self.w_emb = w_emb
        self.p_emb_w = p_emb_w
        self.p_emb_s = p_emb_s
        self.dropout = nn.Dropout(dropout)

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), N)
        
    def forward(self, x, p, ps, mask_x):
        emb_x = self.w_emb(x)
        emb_p_w = self.p_emb_w(p)
        emb_p_s = self.p_emb_s(ps) 
        return self.encoder(self.dropout(emb_x + emb_p_w + emb_p_s), mask_x)  


