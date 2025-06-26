import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .transformer_components import MLP

class ICLAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_k = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        if config.icl_use_wv:
            self.W_v = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            self.W_o = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        else:
            self.W_o = nn.Linear(config.n_heads * config.hidden_dim, config.hidden_dim, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.hidden_dim)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.hidden_dim // config.n_heads, max_seq_len=config.max_seq_len + 1)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
        # Mask diagonal (ignoring the first to prevent softmax issues) 
        # cached_training_mask = torch.triu(torch.ones(1, 1, config.max_seq_len + 1, config.max_seq_len + 1), diagonal=0).bool()
        # cached_training_mask[0, 0] = False
        
        # self.register_buffer("cached_training_mask", cached_training_mask, persistent=False)

    def forward(self, q, k, v):
        
        B, S, E = q.shape
        device = q.device
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        
        if self.config.icl_use_wv:
            v = self.W_v(v).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        else:
            v = v.unsqueeze(2).expand(B, S, self.config.n_heads, self.config.hidden_dim).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        # if S == self.config.max_seq_len + 1:
        #     causal_mask = self.cached_training_mask
        # else:
        causal_mask = torch.triu(torch.ones(1, 1, S, S), diagonal=0).bool().to(device)
        causal_mask[0, 0] = False
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        if self.config.icl_use_wv:
            attn_output = attn_output.view(B, S, self.config.hidden_dim)
        else:
            attn_output = attn_output.view(B, S, self.config.n_heads * self.config.hidden_dim)
            
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output

class ICLBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.mlp = MLP(config)
        
        if config.icl_use_ln_mlp:
            self.ln_mlp = nn.LayerNorm(config.hidden_dim)
        
        self.attention = ICLAttention(config)
        
        if config.icl_use_ln_v:
            self.ln_v = nn.LayerNorm(config.hidden_dim)
        if config.icl_use_ln_qk:
            self.ln_qk = nn.LayerNorm(config.hidden_dim)
        
    def _calculate_ex(self, functional_update):
        
        if self.config.icl_use_ln_mlp:
            ex_term = self.mlp(self.ln_mlp(functional_update))
        else:
            ex_term = self.mlp(functional_update)
        
        if self.config.icl_use_skip_mlp:
            ex_term += functional_update
        
        return ex_term
        
    def forward(self, covariates, targets, functional_update):
        v = targets - self._calculate_ex(functional_update)
        
        if self.config.icl_use_ln_v:
            v = self.ln_v(v)
        
        if self.config.icl_use_ln_qk:
            q = k = self.ln_qk(covariates)
        else:
            q = k = covariates
            
        functional_update = functional_update + self.attention(q, k, v)
        
        return covariates, targets, functional_update