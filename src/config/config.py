import os
import yaml
from types import SimpleNamespace
from src.data import Tokenizer
from src.data import DiskDataset

class Config:
    
    # Shared Fields
    model_type = "transformer"
    d_embed = 512
    max_seq_len = DiskDataset.MAX_SEQ_LEN
    n_heads = 8
    vocab_size = Tokenizer.VOCAB_SIZE
    
    # Transformer Model
    n_blocks = 1
    random_blocks = False
    
    # ICL Model
    n_feature_blocks = 1
    n_icl_blocks = 1
    
    random_feature_blocks = False
    random_icl_blocks = False
    
    share_heads_for_icl = True
    share_projection_for_icl = False
    use_wv_for_icl = False
    use_rotary_for_icl = False
    use_mlp_for_icl = False
    
    update_covariates = False
    
    use_icl_for_features = False
    
    # Training Details
    dataset_name = None
    
    def __init__(self, preset_name=None, config_override=None, dataset_name=None):
        
        self.dataset_name = dataset_name
        
        if preset_name is not None:
            self._load_from_yml(preset_name)
            
        if config_override is not None:
            self._override_values(config_override)
        
    def _load_from_yml(self, preset_name):
        path = f"../config/presets/{preset_name}.yml"
        
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Preset '{preset_name}' not found at {path}")
        
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config field '{key}' in {preset_name}.yml - ignored.")
    
    def _override_values(self, config_override):
        config_override = config_override.split(",")
        for override in config_override:
            kv = override.split("=")
            key = kv[0]
            value = kv[1]
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config field '{key}' in override values - ignored.")

    def get_name(self):
        
        name = f"{self.model_type}_{self.d_embed}D_{self.max_seq_len}S_{self.n_heads}H"
        
        if self.model_type == "transformer":
            name += f"_{self.n_blocks}L"
            
            if self.random_blocks:
                name += f"_rand"
        
        elif self.model_type == "icl":
            name += f"_{self.n_feature_blocks}F_{self.n_icl_blocks}ICL"
            
            if self.random_feature_blocks:
                name += f"_randF"
            
            if self.random_icl_blocks:
                name += f"_randICL"
            
            if self.share_heads_for_icl:
                name += f"_shareHeadsICL"
            
            if self.share_projection_for_icl:
                name += f"_sharedProjICL"
            
            if self.use_wv_for_icl:
                name += f"_wvICL"
            
            if self.use_rotary_for_icl:
                name += f"_rotaryICL"
                
            if self.use_mlp_for_icl:
                name += f"_mlpICL"
    
            if self.update_covariates:
                name += f"_updatedCovariates"

            if self.use_icl_for_features:
                name += f"_iclFeatures"
        
        if self.dataset_name is not None:
            name += f"_ds={self.dataset_name}"
        
        return name