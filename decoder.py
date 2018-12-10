import numpy as np
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable

from utils_pg import *
from transformer.layers import MultiHeadedAttention, PositionwiseFeedForward, DecoderLayer
from transformer.model import Decoder

class LocalDecoder(nn.Module):
    def __init__(self, w_emb, p_emb_w, p_emb_s, vocab, d_model=256, d_ff=1024, h=8, dropout=0.1, N=6):
        super(LocalDecoder, self).__init__()  
        self.w_emb = w_emb
        self.p_emb_w = p_emb_w
        self.p_emb_s = p_emb_s
        self.dropout = nn.Dropout(dropout)

        self_attn = MultiHeadedAttention(h, d_model)
        src_attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.decoder = Decoder(DecoderLayer(d_model, self_attn, src_attn, ff, dropout), N)
        self.proj = nn.Linear(d_model, vocab) 

    def forward(self, x, p, ps, memory, mask_x, mask_y):
        emb_x = self.w_emb(x)
        emb_p_w = self.p_emb_w(p)
        #emb_p_s = self.p_emb_s(ps) 
        h = self.decoder(self.dropout(emb_x + emb_p_w), memory, mask_x, mask_y)  
        pred = T.softmax(self.proj(h), dim=-1)
        return pred
