import torch
import torch.nn as nn
from src.models.model_components import ICLBlock, MLP, initialize_weights

class ICL2(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_component)
        
        if config.end_with_mlp:
            config.d_mlp = config.d_embed
            self.end_mlp = MLP(config)
        
        self.icl_blocks = nn.ModuleList([ICLBlock(config, i) for i in range(config.n_blocks)])
                
        self.ln_out = nn.LayerNorm(config.d_component)
        
        self.lm_head = nn.Linear(config.d_component, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(initialize_weights)
    
    def forward(self, x):
        
        embeddings = self.embedding(x)
        
        B, S, E = embeddings.shape
        device = x.device
        
        covariates = embeddings
        
        targets = embeddings[:, 1:, :]
        targets = torch.cat([targets, torch.zeros(B, 1, E, device=device)], dim=1)
        
        functional_update = torch.zeros_like(embeddings)
        
        for block in self.icl_blocks:
            covariates, targets, functional_update = block(covariates, targets, functional_update)
            
        if self.config.end_with_mlp:
            functional_update = functional_update + self.end_mlp(functional_update)
            
        functional_update = self.ln_out(functional_update)
        logits = self.lm_head(functional_update)
        
        return logits