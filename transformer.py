import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class TransformerLayer(nn.Module):
    
    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, with_external=False, weights_dropout = True):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.attn_layer_norm = LayerNorm(embed_dim)
        self.ff_layer_norm = LayerNorm(embed_dim)
        self.with_external = with_external
        self.dropout = dropout
        if self.with_external:
            self.external_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
            self.external_layer_norm = LayerNorm(embed_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, kv = None,
                self_padding_mask = None, self_attn_mask = None,
                external_memories = None, external_padding_mask=None,
                need_weights = False):
        # x: seq_len x bsz x embed_dim
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights = need_weights)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights = need_weights)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        if self.with_external:
            residual = x
            x, external_attn = self.external_attn(query=x, key=external_memories, value=external_memories, key_padding_mask=external_padding_mask, need_weights = need_weights)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn = None

        residual = x
        x = gelu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)

        return x, self_attn, external_attn
    
class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False):
        """ Input shape: Time x Batch x Channel
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q*self.scaling
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        # k,v: bsz*heads x src_len x dim
        # q: bsz*heads x tgt_len x dim 

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        
        if attn_mask is not None:
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0),
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.transpose(0, 1).unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            
            #attn_weights, _ = attn_weights.max(dim=1)
            attn_weights = attn_weights[:, 0, :, :]
            #attn_weights = attn_weights.mean(dim=1)
            attn_weights = attn_weights.transpose(0, 1)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf*x

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.eps = eps
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, std=0.02)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class SelfAttentionMask(nn.Module):
    def __init__(self, init_size = 100, device = 0):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)
        self.device = device
    
    @staticmethod
    def get_mask(size):
        weights = torch.triu(torch.ones((size, size), dtype = torch.bool), 1)
        return weights

    def forward(self, size):
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        res = self.weights[:size,:size].cuda(self.device).detach()
        return res

class LearnedPositionalEmbedding(nn.Module):
    """This module produces LearnedPositionalEmbedding.
    """
    def __init__(self, embedding_dim, init_size=1024, device=0):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weights = nn.Embedding(init_size, embedding_dim)
        self.device= device
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weights.weight, std=0.02)

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        seq_len, bsz = input.size()
        positions = (offset + torch.arange(seq_len)).cuda(self.device)
        res = self.weights(positions).unsqueeze(1).expand(-1, bsz, -1)
        return res

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    """
    def __init__(self, embedding_dim, init_size=1024, device=0):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim
        )
        self.device= device

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        seq_len, bsz = input.size()
        mx_position = seq_len + offset
        if self.weights is None or mx_position > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                mx_position,
                self.embedding_dim,
            )

        positions = offset + torch.arange(seq_len)
        res = self.weights.index_select(0, positions).unsqueeze(1).expand(-1, bsz, -1).cuda(self.device).detach()
        return res
