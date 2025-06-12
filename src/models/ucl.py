import torch
import torch.nn as nn
from src.models.model_components import UCLBlock, initialize_weights

class UCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.w_s = nn.Parameter(torch.randn(1, 1, config.d_embed)) # (B, S, E)
        
        self.blocks = nn.ModuleList([UCLBlock(config, self.embedding) for _ in range(config.n_icl_blocks)])
                
        self.ln_out = nn.LayerNorm(config.d_embed)
        
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(initialize_weights)
    
    def forward(self, x, inference_mode=False):
        
        embeddings = self.embedding(x) # (B, S, E)
        
        B, S, E = embeddings.shape
        device = embeddings.device
        
        w_s = self.w_s.expand(B, -1, -1) # (B, 1, E)
        y_NP1 = torch.zeros(B, 1, E, device=device) # (B, 1, E)
        
        icl_covariates = torch.cat([w_s, embeddings], dim=1) # (B, S+1, E)
        icl_targets = torch.cat([embeddings, y_NP1], dim=1) # (B, S+1, E)
        icl_functional_update = torch.zeros(B, S+1, E, device=device) # (B, S+1, E)
        
        for block in self.blocks:
            icl_covariates, icl_targets, icl_functional_update = block(icl_covariates, icl_targets, icl_functional_update)

        x = icl_functional_update[:, 1:, :] # (B, S, E)
    
        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)
        
        return logits