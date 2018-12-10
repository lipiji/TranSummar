import numpy as np
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils_pg import *
from transformer.layers import MultiHeadedAttention, PositionwiseFeedForward, DecoderLayer
from transformer.model import Decoder

class LocalDecoder(nn.Module):
    def __init__(self, device, w_emb, p_emb_w, p_emb_s, vocab, d_model=256, d_ff=1024, h=8, dropout=0.1, N=6):
        super(LocalDecoder, self).__init__()  
        self.w_emb = w_emb
        self.p_emb_w = p_emb_w
        self.p_emb_s = p_emb_s
        self.dropout = nn.Dropout(dropout)
        self.copy = True
        self.device = device

        self_attn = MultiHeadedAttention(h, d_model)
        src_attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.decoder = Decoder(DecoderLayer(d_model, self_attn, src_attn, ff, dropout), N)
        self.proj = nn.Linear(d_model, vocab) 

        self.Wc_att = nn.Parameter(torch.Tensor(d_model, d_model))
        self.b_att = nn.Parameter(torch.Tensor(d_model))

        self.W_comb_att = nn.Parameter(torch.Tensor(d_model, d_model))
        self.U_att = nn.Parameter(torch.Tensor(1, d_model))
        
        if self.copy:
            self.v = nn.Parameter(torch.Tensor(1, d_model*3))
            self.bv = nn.Parameter(torch.Tensor(1))
            init_xavier_weight(self.v) 
            init_bias(self.bv)

        init_xavier_weight(self.Wc_att)
        init_bias(self.b_att)
        init_xavier_weight(self.W_comb_att)
        init_xavier_weight(self.U_att)

    def copy_attn(self, memory, hs, xid, x_mask):
        
        memory = memory.transpose(0, 1)
        x_mask = x_mask.squeeze().transpose(0, 1).unsqueeze(-1)

        def _get_word_atten(pctx, h1, x_mask):
            h = F.linear(h1, self.W_comb_att)
            unreg_att = T.tanh(pctx + h) * x_mask
            unreg_att = F.linear(unreg_att, self.U_att)

            word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim = True)[0]) * x_mask
            sum_word_atten = T.sum(word_atten, 0, keepdim = True)
            word_atten =  word_atten / sum_word_atten
            return word_atten

        def recurrence(h, pctx, context, x_mask):
            # len(x) * batch_size * 1
            word_atten = _get_word_atten(pctx, h, x_mask)
            atted_ctx = T.sum(word_atten * context, 0)

            word_atten_ = T.transpose(word_atten.view(x_mask.size(0), -1), 0, 1) # B * len(x)
            return atted_ctx, word_atten_

        pctx = F.linear(memory, self.Wc_att, self.b_att)
        atts, dists, xids = [], [], []
        steps = range(hs.size(1))
        for i in steps:
            att, att_dist = recurrence(hs[:,i,:], pctx, memory, x_mask)
            atts += [att]
            dists += [att_dist]
            xids += [xid]
        
        atts = T.stack(atts).view(hs.size(1), *atts[0].size())
        dists = T.stack(dists).view(hs.size(1), *dists[0].size())
        xids = T.stack(xids).view(hs.size(1), *xids[0].size())
       
        atts = atts.transpose(0, 1)
        dists = dists.transpose(0, 1)
        xids = xids.transpose(0, 1)
        return atts, dists, xids 

    def forward(self, x, p, ps, memory, mask_x, mask_y, xid, max_ext_len):
        emb_x = self.w_emb(x)
        emb_p_w = self.p_emb_w(p)
        #emb_p_s = self.p_emb_s(ps) 
        h = self.decoder(self.dropout(emb_x + emb_p_w), memory, mask_x, mask_y)  
        pred = T.softmax(self.proj(h), dim=-1)
        
        if self.copy:
            atts, dists, xids = self.copy_attn(memory, h, xid, mask_x)   
            if max_ext_len > 0:
                ext_zeros = Variable(torch.zeros(pred.size(0), pred.size(1), max_ext_len)).to(self.device)
                pred = T.cat((pred, ext_zeros), 2)
            g = T.sigmoid(F.linear(T.cat([self.dropout(emb_x + emb_p_w), h, atts], -1), self.v, self.bv))
            pred = (g * pred).scatter_add(2, xids, (1 - g) * dists)
        


        return pred
