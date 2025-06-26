import torch
import torch.nn as nn
from src.models.model_components import ICLBlock, TransformerBlock, MLP, initialize_weights

class ICL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        
        if config.block_order is None:
            config.block_order = ['t', 'i'] * (config.n_blocks // 2)
            if config.n_blocks % 2 != 0:
                config.block_order = ['t'] + config.block_order
        
        self.blocks = nn.ModuleList([TransformerBlock(config) if sym.lower() == "t" else ICLBlock(config) for sym in config.block_order])
        
        if self.config.share_covariate_attn:
            self.share_covariate_attn()
        if self.config.share_covariate_mlp:
            self.share_covariate_mlp()
        if self.config.share_icl_attn:
            self.share_icl_attn()
        if self.config.share_icl_mlp:
            self.share_icl_mlp()
        
        if config.use_output_mlp:
            self.output_mlp = MLP(config)
        
        self.ln_out = nn.LayerNorm(config.d_embed)
        
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(initialize_weights)
    
    def share_covariate_attn(self):
        shared_W_q = shared_W_k = shared_W_v = None

        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() == "t":
                attn = block.attention

                if shared_W_q is None:
                    shared_W_q = attn.W_q
                    shared_W_k = attn.W_k
                    shared_W_v = attn.W_v
                else:
                    attn._modules.pop('W_q', None)
                    attn._modules.pop('W_k', None)
                    attn._modules.pop('W_v', None)

                    attn.W_q = shared_W_q
                    attn.W_k = shared_W_k
                    attn.W_v = shared_W_v

                    attn.add_module('W_q', shared_W_q)
                    attn.add_module('W_k', shared_W_k)
                    attn.add_module('W_v', shared_W_v)

    def share_icl_attn(self):
        shared_W_q = shared_W_k = shared_W_v = None

        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() != "t":
                attn = block.attention

                if shared_W_q is None:
                    shared_W_q = attn.W_q
                    shared_W_k = attn.W_k
                    if self.config.icl_use_wv:
                        shared_W_v = attn.W_v
                else:
                    attn._modules.pop('W_q', None)
                    attn._modules.pop('W_k', None)
                    attn.W_q = shared_W_q
                    attn.W_k = shared_W_k
                    attn.add_module('W_q', shared_W_q)
                    attn.add_module('W_k', shared_W_k)

                    if self.config.icl_use_wv:
                        attn._modules.pop('W_v', None)
                        attn.W_v = shared_W_v
                        attn.add_module('W_v', shared_W_v)

    def share_covariate_mlp(self):
        shared_fc_1 = shared_fc_2 = None

        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() == "t":
                mlp = block.mlp

                if shared_fc_1 is None:
                    shared_fc_1 = mlp.fc_1
                    shared_fc_2 = mlp.fc_2
                else:
                    mlp._modules.pop('fc_1', None)
                    mlp._modules.pop('fc_2', None)

                    mlp.fc_1 = shared_fc_1
                    mlp.fc_2 = shared_fc_2

                    mlp.add_module('fc_1', shared_fc_1)
                    mlp.add_module('fc_2', shared_fc_2)

    def share_icl_mlp(self):
        shared_fc_1 = shared_fc_2 = None

        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() != "t":
                mlp = block.mlp

                if shared_fc_1 is None:
                    shared_fc_1 = mlp.fc_1
                    shared_fc_2 = mlp.fc_2
                else:
                    mlp._modules.pop('fc_1', None)
                    mlp._modules.pop('fc_2', None)

                    mlp.fc_1 = shared_fc_1
                    mlp.fc_2 = shared_fc_2

                    mlp.add_module('fc_1', shared_fc_1)
                    mlp.add_module('fc_2', shared_fc_2)
        
    def forward(self, x):
        
        embeddings = self.embedding(x)
        
        B, S, E = embeddings.shape
        device = x.device
        
        covariates = embeddings
        
        targets = embeddings[:, 1:, :]
        targets = torch.cat([targets, torch.zeros(B, 1, E, device=device)], dim=1)
        
        functional_update = torch.zeros_like(embeddings)
        
        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() == "t":
                covariates = block(covariates)
            else:
                covariates, targets, functional_update = block(covariates, targets, functional_update)
            
        if self.config.use_output_mlp:
            functional_update = functional_update + self.output_mlp(functional_update)
            
        functional_update = self.ln_out(functional_update)
        logits = self.lm_head(functional_update)
        
        return logits