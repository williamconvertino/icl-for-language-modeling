import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_components import TransformerBlock, Attention, MLP
from .icl_components import ICLBlock

class UCLBlock(nn.Module):
    def __init__(self, config, embedding, base_icl_attn=None):
        super().__init__()
        
        self.config = config
        
        if self.config.use_icl_for_features:
            self.feature_block = ICLBlock(config, embedding, base_icl_attn)
        elif self.config.uc_update_mode == "func_attn" or self.config.uc_update_mode == "x_attn":
            self.feature_block = Attention(config)
        elif self.config.uc_update_mode == "func_mlp" or self.config.uc_update_mode == "x_mlp":
            self.feature_block = MLP(config)
        else:
            self.feature_block = TransformerBlock(config)
        
        self.icl_block = ICLBlock(config, embedding, base_icl_attn)
                    
    def forward(self, covariates, targets, functional_update):
        
        if self.config.use_icl_for_features:
            _, _, covariate_update = self.feature_block(functional_update, targets, covariates, skip_update=True)
            covariates = covariates + covariate_update
        elif self.config.uc_update_mode == "func_trans" or self.config.uc_update_mode == "func_attn":
            covariates = self.feature_block(covariates, k=functional_update, v=functional_update)
        else:
            covariates = self.feature_block(covariates)
            
        covariates, targets, functional_update = self.icl_block(covariates, targets, functional_update)

        return covariates, targets, functional_update