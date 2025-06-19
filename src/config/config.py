import os
import yaml
from src.data import TOKENIZER
import copy

class Config:
    
    # Common Fields
    model_type = "transformer"
    d_embed = 512
    max_seq_len = 512
    n_heads = 8
    vocab_size = len(TOKENIZER)
    
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
    use_no_icl_exp = False
    
    update_covariates = False
    
    use_icl_for_features = False
    
    # UCL Model
    uc_update_mode = "x_trans"
    
    # Training Details
    dataset_name = None
    
    def __init__(self, preset_name=None, config_override=None, dataset_name=None):
        
        self.dataset_name = dataset_name
        
        if preset_name is not None:
            self._load_from_yml(preset_name)
            
        if config_override is not None:
            self._override_values(config_override)
        
    def _load_from_yml(self, preset_name):
        
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "presets", f"{preset_name}.yml"))
        
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
        def parse_value(val):
            # Try to convert to int
            try:
                return int(val)
            except ValueError:
                pass
            # Try to convert to float
            try:
                return float(val)
            except ValueError:
                pass
            # Try to convert to bool
            lowered = val.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            # Fallback: keep as string
            return val

        config_override = config_override.split(",")
        for override in config_override:
            kv = override.split("=")
            if len(kv) != 2:
                print(f"Warning: Invalid override format '{override}' - ignored.")
                continue
            key, value = kv
            if hasattr(self, key):
                parsed_value = parse_value(value)
                setattr(self, key, parsed_value)
            else:
                print(f"Warning: Unknown config field '{key}' in override values - ignored.")

    def clone(self):
        new_config = Config()
        new_config.__dict__ = copy.deepcopy(self.__dict__)
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                if attr not in new_config.__dict__:
                    setattr(new_config, attr, copy.deepcopy(getattr(self, attr)))
        return new_config

    def get_name(self):
        
        name = f"{self.model_type}_{self.d_embed}D_{self.max_seq_len}S_{self.n_heads}H"
        
        if self.model_type == "transformer":
            name += f"_{self.n_blocks}L"
            
            if self.random_blocks:
                name += f"_rand"
        
        elif self.model_type == "icl" or self.model_type == "ucl":
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

            if self.use_no_icl_exp:
                name += f"_noICLEXP"
        
        if self.model_type == "ucl":
            name += f"_ucUpdate={self.uc_update_mode}"
        
        if self.dataset_name is not None:
            name += f"_ds={self.dataset_name}"
        
        return name