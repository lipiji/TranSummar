# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import numpy as np
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable
import copy

from utils_pg import *
from encoder import *
from decoder import *

from transformer.layers import Embeddings, PositionEmbeddings

class Model(nn.Module):
    def __init__(self, modules, consts, options):
        super(Model, self).__init__()  
        
        self.is_predicting = options["is_predicting"]
        self.beam_decoding = options["beam_decoding"]
        self.device = options["device"]
        self.copy = options["copy"]
        self.coverage = options["coverage"]
        self.avg_nll = options["avg_nll"]

        self.dim_x = consts["dim_x"]
        self.dim_y = consts["dim_y"]
        self.len_x = consts["len_x"]
        self.len_y = consts["len_y"]
        self.hidden_size = consts["hidden_size"]
        self.d_model = self.hidden_size
        self.d_ff = consts["d_ff"]
        self.num_heads = consts["num_heads"]
        self.dropout = consts["dropout"]
        self.num_layers = consts["num_layers"]
        self.dict_size = consts["dict_size"] 
        self.pad_token_idx = consts["pad_token_idx"] 
        self.word_pos_size = consts["word_pos_size"]
        #self.sent_pos_size = consts["sent_pos_size"]

        self.word_emb = Embeddings(self.dict_size, self.dim_x, self.pad_token_idx)
        self.pos_emb_w = PositionEmbeddings(self.dim_x, self.word_pos_size)
        self.pos_emb_s = PositionEmbeddings(self.dim_x, self.word_pos_size)

        self.encoder = LocalEncoder(self.word_emb, self.pos_emb_w, self.pos_emb_s, \
                                    self.d_model, self.d_ff, self.num_heads,\
                                    self.dropout, self.num_layers)
        self.decoder = LocalDecoder(self.word_emb, self.pos_emb_w, self.pos_emb_s, self.dict_size, \
                                    self.d_model, self.d_ff, self.num_heads,\
                                    self.dropout, self.num_layers)

        self.init_weights()

    def init_weights(self):
        for p in self.word_emb.parameters():
            nn.init.xavier_uniform_(p)
            
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -T.log(T.gather(y_pred, -1, y.unsqueeze(-1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = T.sum(cost * y_mask, 1) / T.sum(y_mask, -1)
        else:
            cost = T.sum(cost * y_mask, 1)
        cost = cost.view((y.size(0), -1))
        return T.mean(cost) 

    def encode(self, x, p, ps, mask_x):
        return self.encoder(x, p, ps, mask_x)

    def decode(self, x, p, ps, m, mask_x, mask_y):
        return self.decoder(x, p, ps, m, mask_x, mask_y)
    
    def forward(self, x, px, pxs, mask_x, y, py, pys, mask_y_tri, y_tgt, mask_y):
        hs = self.encode(x, px, pxs, mask_x)
        pred = self.decode(y, py, pys, hs, mask_x, mask_y_tri)
        loss = self.nll_loss(pred, y_tgt, mask_y, self.avg_nll)
        ppl = T.exp(loss)
        return pred, ppl, None
    

