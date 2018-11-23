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
        self.pos_emb = PositionEmbeddings(self.dim_x, self.word_pos_size)

        self.encoder = LocalEncoder(self.word_emb, self.pos_emb,\
                                    self.d_model, self.d_ff, self.num_heads,\
                                    self.dropout, self.num_layers)
        self.decoder = LocalDecoder(self.word_emb, self.pos_emb,\
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
        cost = -T.log(T.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = T.sum(cost * y_mask, 0) / T.sum(y_mask, 0)
        else:
            cost = T.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return T.mean(cost) 

    def encode(self, x, p, mask_x):
        return self.encoder(x, p, mask_x)

    def decode(self, x, p, m, mask_x, mask_y):
        return self.decoder(x, p, m, mask_x, mask_y)

    def decode_once(self, y, hs, dec_init_state, mask_x, x=None, max_ext_len=None, acc_att=None):
        batch_size = hs.size(1)
        if T.sum(y) < 0:
            y_emb = Variable(T.zeros((1, batch_size, self.dim_y))).to(self.device)
        else:
            y_emb = self.w_rawdata_emb(y)
        mask_y = Variable(T.ones((1, batch_size, 1))).to(self.device)

        if self.copy and self.coverage:
            hcs, dec_status, atted_context, att_dist, xids, C = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, x, acc_att)
        elif self.copy:
            hcs, dec_status, atted_context, att_dist, xids = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, xid=x)
        elif self.coverage:
            hcs, dec_status, atted_context, att_dist, C = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, init_coverage=acc_att)
        else:
            hcs, dec_status, atted_context = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y)
        
        if self.copy:
            y_pred = self.word_prob(dec_status, atted_context, y_emb, att_dist, xids, max_ext_len)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_emb)

        if self.coverage:
            return y_pred, hcs, C
        else:
            return y_pred, hcs

    def forward(self, x, p, mask_x, mask_y):
        hs = self.encode(x, p, mask_x)
        print hs.shape
        out = self.decode(x, p, hs, mask_x, mask_y)
        return out
    

