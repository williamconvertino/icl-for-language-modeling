import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class ICLAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.d_component, config.d_attn_icl, bias=False)
        self.W_k = nn.Linear(config.d_component, config.d_attn_icl, bias=False)
        
        if config.use_W_v:
            self.W_v = nn.Linear(config.d_component, config.d_attn_icl, bias=False)
            self.W_o = nn.Linear(config.d_attn_icl, config.d_component, bias=False)
        else:
            self.W_o = nn.Linear(config.n_heads_icl * config.d_component, config.d_component, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.d_attn_icl)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_attn_icl // config.n_heads_icl)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        
        B, S, E = q.shape
        
        q = self.W_q(q).view(B, S, self.config.n_heads_icl, self.config.d_attn_icl // self.config.n_heads_icl).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads_icl, self.config.d_attn_icl // self.config.n_heads_icl).transpose(1, 2)
        
        if self.config.use_W_v:
            v = self.W_v(v).view(B, S, self.config.n_heads_icl, self.config.d_attn_icl // self.config.n_heads_icl).transpose(1, 2)
        else:
            v = v.unsqueeze(2).expand(B, S, self.config.n_heads_icl, self.config.d_component).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=0).bool().to(q.device)
        causal_mask[0, 0] = False
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        if self.config.use_W_v:
            attn_output = self.W_o(attn_output.view(B, S, self.config.d_attn_icl))
        else:
            attn_output = self.W_o(attn_output.view(B, S, self.config.n_heads_icl * self.config.d_component))
        
        attn_output = self.drop_resid(attn_output)
        
        return attn_output
    
class CovariateAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.d_component, config.d_attn_covariate, bias=False)
        self.W_k = nn.Linear(config.d_component, config.d_attn_covariate, bias=False)
        self.W_v = nn.Linear(config.d_component, config.d_attn_covariate, bias=False)
        self.W_o = nn.Linear(config.d_attn_covariate, config.d_component, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.d_attn_covariate)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.d_attn_covariate // config.n_heads_covariate)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        
        k = v = q
        
        B, S, E = q.shape
        
        q = self.W_q(q).view(B, S, self.config.n_heads_covariate, self.config.d_attn_covariate // self.config.n_heads_covariate).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads_covariate, self.config.d_attn_covariate // self.config.n_heads_covariate).transpose(1, 2)
        v = self.W_v(v).view(B, S, self.config.n_heads_covariate, self.config.d_attn_covariate // self.config.n_heads_covariate).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.d_attn_covariate)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_1 = nn.Linear(config.d_component, 4 * config.d_mlp)
        self.fc_2 = nn.Linear(4 * config.d_mlp, config.d_component)
        
        self.activation = nn.GELU()    
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x
    
class ICLBlock(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        
        self.config = config
        self.layer_index = layer_index
        
        total_heads = config.n_heads_icl + config.n_heads_covariate
        d_attn_per_head = config.d_embed // total_heads
        
        config.d_attn_covariate = d_attn_per_head * config.n_heads_covariate
        config.d_attn_icl = d_attn_per_head * config.n_heads_icl
        
        # Either share the MLP layer or learn 2 smaller MLPs
        if config.share_mlp:
            config.d_mlp = config.d_embed
            self.mlp_covariate = self.mlp_icl = MLP(config)
        else:
            config.d_mlp = config.d_embed // 2
            self.mlp_covariate = MLP(config)
            self.mlp_icl = MLP(config)
        
        if self.layer_index < self.config.n_blocks - 1:
            self.attn_covariate = CovariateAttention(config)
        
        self.attn_icl = ICLAttention(config)
        
        self.ln_covariate_mlp = nn.LayerNorm(config.d_component)
        self.ln_icl_mlp = nn.LayerNorm(config.d_component)
        
        self.ln_covariate_attn = nn.LayerNorm(config.d_component)
        self.ln_icl_attn = nn.LayerNorm(config.d_component)
        
    def forward(self, covariates, targets, functional_update):
        
        if self.config.start_with_mlp or self.layer_index != 0:
            covariates = covariates + self.mlp_icl(self.ln_covariate_mlp(covariates))

            q = k = self.ln_icl_attn(covariates)

            if self.config.update_targets:
                targets = targets + self.mlp_icl(self.ln_icl_mlp(functional_update))
                v = targets
            else:
                v = targets + self.mlp_icl(self.ln_icl_mlp(functional_update))
        else:
            q = k = covariates
            v = targets
        
        functional_update = functional_update + self.attn_icl(q, k, v)
        
        if self.layer_index < self.config.n_blocks - 1:
            v = covariates
            covariates = covariates + self.attn_covariate(q, k, v)
            
        return covariates, targets, functional_update