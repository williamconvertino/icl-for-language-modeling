import torch
import torch.nn as nn
from src.models.model_components import TransformerBlock, ICLBlock, initialize_weights

class ICL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.w_s = nn.Parameter(torch.randn(1, 1, config.d_embed)) # (B, S, E)
        
        if config.use_icl_for_features:
            self.feature_blocks = nn.ModuleList(self.get_icl_blocks(config.n_feature_blocks))
        else:
            self.feature_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_feature_blocks)])
        
        self.icl_blocks = nn.ModuleList(self.get_icl_blocks(config.n_icl_blocks))
        
        # Prevents transformer blocks from training
        if config.random_feature_blocks:
            for block in self.feature_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            
        if config.random_icl_blocks:
            for block in self.icl_blocks:
                for param in block.parameters():
                    param.requires_grad = False
                
        self.ln_out = nn.LayerNorm(config.d_embed)
        
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(initialize_weights)
    
    def get_icl_blocks(self, num_blocks):
        if self.config.share_heads_for_icl or self.config.share_projection_for_icl:
            base_icl_block = ICLBlock(self.config, self.embedding)
            icl_blocks = [base_icl_block] + [ICLBlock(self.config, self.embedding, base_icl_block.attn) for _ in range(num_blocks - 1)] # Ensures we have access to shared parameters
        else:
            icl_blocks = [ICLBlock(self.config, self.embedding) for _ in range(num_blocks)]
        return icl_blocks
    
    def forward(self, x, inference_mode=False):
        
        embeddings = self.embedding(x) # (B, S, E)
        
        B, S, E = embeddings.shape
        device = embeddings.device
        
        w_s = self.w_s.expand(B, -1, -1) # (B, 1, E)
        y_NP1 = torch.zeros(B, 1, E, device=device) # (B, 1, E)
        
        icl_covariates = torch.cat([w_s, embeddings], dim=1) # (B, S+1, E)
        icl_targets = torch.cat([embeddings, y_NP1], dim=1) # (B, S+1, E)
        icl_functional_update = torch.zeros(B, S+1, E, device=device) # (B, S+1, E)
        
        if self.config.use_icl_for_features:
            for block in self.feature_blocks:
                icl_covariates, icl_targets, icl_functional_update = block(icl_covariates, icl_targets, icl_functional_update)
            icl_covariates = icl_functional_update # Use output of first ICL blocks as covariates
            icl_functional_update = torch.zeros(B, S+1, E, device=device)    
        else:
            for block in self.feature_blocks:
                icl_covariates = block(icl_covariates)
        
        for block in self.icl_blocks:
            icl_covariates, icl_targets, icl_functional_update = block(icl_covariates, icl_targets, icl_functional_update)

        x = icl_functional_update[:, 1:, :] # (B, S, E)
    
        x = self.ln_out(x)
        
        logits = self.lm_head(x) # (B, S, V)
        
        return logits