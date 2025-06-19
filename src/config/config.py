import os
import yaml
from src.data import TOKENIZER
import copy

class Config:
    
    # Model Info
    model_type = "transformer"
    d_embed = 512
    max_seq_len = 512
    n_heads = 8
    vocab_size = len(TOKENIZER)
    n_blocks = 8
    
    # ICL Specific
    share_mlp = False
    reduced_vectors = False
    start_with_mlp = False
    end_with_mlp = False
    
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
            
            try:
                return int(val)
            except ValueError:
                pass
            
            try:
                return float(val)
            except ValueError:
                pass
            
            lowered = val.lower()
            
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            
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
        
        name = f"{self.model_type}_{self.d_embed}D_{self.max_seq_len}S_{self.n_heads}H_{self.n_blocks}L"
        
        if self.model_type == "icl":
          
            if self.share_mlp:
                name += f"_shareMLP"
            
            if self.start_with_mlp:
                name += f"_mlpStart"
            
            if self.end_with_mlp:
                name += f"_mlpEnd"
                
            if self.reduced_vectors:
                name += f"_reducedVectors"
        
        if self.dataset_name is not None:
            name += f"_ds={self.dataset_name}"
        
        return name