import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from utils_pg import *

class LSTMAttentionDecoder01(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_size, dict_size, device, copy, coverage, is_predicting):
        super(LSTMAttentionDecoder01, self).__init__()
        self.input_size = input_size
        self.dim_y = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.is_predicting = is_predicting
        self.dict_size = dict_size
        self.device = device
        self.copy = copy
        self.coverage = coverage 
        
        self.lstm_1 = nn.LSTMCell(self.input_size, self.hidden_size)

        self.Wx = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.ctx_size))
        self.Ux = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.hidden_size))
        self.bx = nn.Parameter(torch.Tensor(4 * self.hidden_size))

        self.Wc_att = nn.Parameter(torch.Tensor(self.ctx_size, self.ctx_size))
        self.b_att = nn.Parameter(torch.Tensor(self.ctx_size))

        self.W_comb_att = nn.Parameter(torch.Tensor(self.ctx_size, 2*self.hidden_size))
        self.U_att = nn.Parameter(torch.Tensor(1, self.ctx_size))

        self.w_ds = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size + self.ctx_size + self.dim_y))
        self.b_ds = nn.Parameter(torch.Tensor(self.hidden_size)) 
        self.w_logit = nn.Parameter(torch.Tensor(self.dict_size, self.hidden_size))
        self.b_logit = nn.Parameter(torch.Tensor(self.dict_size)) 
        if self.copy:
            self.v = nn.Parameter(torch.Tensor(1, self.hidden_size + self.ctx_size + self.dim_y))
            self.bv = nn.Parameter(torch.Tensor(1))

        if self.coverage:
            self.W_coverage= nn.Parameter(torch.Tensor(self.ctx_size, 1))

        self.init_weights()

    def init_weights(self):
        init_lstm_weight(self.lstm_1)
        init_ortho_weight(self.Wx)
        init_ortho_weight(self.Ux)
        init_bias(self.bx)
        init_ortho_weight(self.Wc_att)
        init_bias(self.b_att)
        init_ortho_weight(self.W_comb_att)
        init_ortho_weight(self.U_att)
        if self.coverage:
            init_ortho_weight(self.W_coverage)
        
        init_xavier_weight(self.w_ds)
        init_bias(self.b_ds)
        init_xavier_weight(self.w_logit)
        init_bias(self.b_logit)
        if self.copy:
            init_xavier_weight(self.v)
            init_bias(self.bv)

    def forward(self, y_emb, context, init_state, x_mask, y_mask, xid, init_coverage, max_ext_len):
        def _get_word_atten(pctx, h1, x_mask, acc_att=None): #acc_att: B * len(x)
            if acc_att is not None:
                h = F.linear(h1, self.W_comb_att) + F.linear(T.transpose(acc_att, 0, 1).unsqueeze(2), self.W_coverage) # len(x) * B * ?
            else:
                h = F.linear(h1, self.W_comb_att)
            unreg_att = T.tanh(pctx + h) * x_mask
            unreg_att = F.linear(unreg_att, self.U_att)

            word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim = True)[0]) * x_mask
            sum_word_atten = T.sum(word_atten, 0, keepdim = True)
            word_atten =  word_atten / sum_word_atten
            return word_atten

        def recurrence(x, y_mask, hidden, pctx, context, x_mask, xids, max_ext_len, acc_att):
            pre_h, pre_c = hidden

            h1, c1 = self.lstm_1(x, hidden)  
            h1 = y_mask * h1 + (1.0 - y_mask) * pre_h
            c1 = y_mask * c1 + (1.0 - y_mask) * pre_c
            
            # len(x) * batch_size * 1
            s = T.cat((h1.view(-1, self.hidden_size), c1.view(-1, self.hidden_size)), 1)
            if self.coverage:
                word_atten = _get_word_atten(pctx, s, x_mask, acc_att)
            else:
                word_atten = _get_word_atten(pctx, s, x_mask)
            atted_ctx = T.sum(word_atten * context, 0)

            ifoc_preact = F.linear(h1, self.Ux) + F.linear(atted_ctx, self.Wx, self.bx)
            x4i, x4f, x4o, x4c = ifoc_preact.chunk(4, 1)
            i = torch.sigmoid(x4i)
            f = torch.sigmoid(x4f)
            o = torch.sigmoid(x4o)
            c2 = f * c1 + i * torch.tanh(x4c)
            h2 = o * torch.tanh(c2)
            c2 = y_mask * c2 + (1.0 - y_mask) * c1
            h2 = y_mask * h2 + (1.0 - y_mask) * h1

            word_atten_ = T.transpose(word_atten.view(x_mask.size(0), -1), 0, 1)

            h = T.cat((h2, atted_ctx, x), 1)
            logit = T.tanh(F.linear(h, self.w_ds, self.b_ds))
            logit = F.linear(logit, self.w_logit, self.b_logit)
            y_dec = T.softmax(logit, dim = 1)
        
            if self.copy:
                ext_zeros = Variable(torch.zeros(y_dec.size(0), max_ext_len)).to(self.device)
                y_dec = T.cat((y_dec, ext_zeros), 1)
                g = T.sigmoid(F.linear(h, self.v, self.bv))
                y_dec = (g * y_dec).scatter_add(1, xids, (1 - g) * word_atten_)
        
    
            acc_att += word_atten_
            return (h2, c2), h2, atted_ctx, word_atten_, acc_att, logit, y_dec

        hs = []
        cs = []
        ss = []
        atts = []
        dists = [] 
        Cs = []
        logits = []
        ydecs = []

        hidden = init_state
        acc_att = init_coverage
        if self.copy: 
            xid = T.transpose(xid, 0, 1) # B * len(x)

        pctx = F.linear(context, self.Wc_att, self.b_att)
        x = y_emb
        
        steps = range(y_emb.size(0))
        for i in steps:
            Cs.append(acc_att)
            hidden, s, att, att_dist, acc_atti, logit, y_dec = \
                recurrence(x[i], y_mask[i], hidden, pctx, context, x_mask, xid, max_ext_len, acc_att)
            hs.append(hidden[0])
            cs.append(hidden[1])
            ss.append(s)
            atts.append(att)
            dists.append(att_dist)
            logits.append(logit)
            ydecs.append(y_dec)
        
        if self.coverage:
            if self.is_predicting :
                Cs.append(acc_att)
                Cs = Cs[1:]
            Cs = T.cat(Cs, 0).view(y_emb.size(0), *Cs[0].size())
        

        hs = T.cat(hs, 0).view(y_emb.size(0), *hs[0].size())
        cs = T.cat(cs, 0).view(y_emb.size(0), *cs[0].size())
        ss = T.cat(ss, 0).view(y_emb.size(0), *ss[0].size())
        atts = T.cat(atts, 0).view(y_emb.size(0), *atts[0].size())
        dists = T.cat(dists, 0).view(y_emb.size(0), *dists[0].size())
        logits = T.cat(logits, 0).view(y_emb.size(0), *logits[0].size())
        ydecs = T.cat(ydecs, 0).view(y_emb.size(0), *ydecs[0].size())

        return (hs, cs), ss, atts, dists, Cs, ydecs

