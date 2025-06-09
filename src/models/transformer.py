import torch.nn as nn
from src.models.model_components import TransformerBlock, initialize_weights

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        
        # Prevents transformer blocks from training
        if config.random_blocks:
            for block in self.transformer_blocks:
                for param in block.parameters():
                    param.requires_grad = False
                
        self.ln_out = nn.LayerNorm(config.d_embed)
        
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(initialize_weights)
    
    def forward(self, x, inference_mode=False):
        
        x = self.embedding(x)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.ln_out(x)
        x = self.lm_head(x)
        
        return x