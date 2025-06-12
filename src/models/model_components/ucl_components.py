import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_components import TransformerBlock
from .icl_components import ICLBlock

class UCLBlock(nn.Module):
    def __init__(self, config, embedding, base_icl_attn=None):
        super().__init__()
        
        if self.config.use_icl_for_features:
            self.feature_block = ICLBlock(config, embedding, base_icl_attn)
        else:
            self.feature_block = TransformerBlock(config)
        
        self.icl_block = ICLBlock(config, embedding, base_icl_attn)
                    
    def forward(self, covariates, targets, functional_update):
        
        if self.config.use_icl_for_features:
            _, _, covariate_update = self.feature_block(functional_update, targets, covariates, skip_update=True)
            covariates = covariates + covariate_update
        else:
            covariates = self.feature_block(covariates)
            
        covariates, targets, functional_update = self.icl_block(covariates, targets, functional_update)

        return covariates, targets, functional_update