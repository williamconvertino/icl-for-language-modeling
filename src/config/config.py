import os
import yaml
from src.data import Tokenizer
import copy

class Config:
    
    # Model Info
    model_type = "transformer"
    d_embed = 512
    max_seq_len = 512
    n_heads = 8
    vocab_size = len(Tokenizer())
    n_blocks = 8
    
    # ICL Specific
    
    block_order = None
    
    icl_use_wv = False
    icl_use_ln_mlp = False
    icl_use_skip_mlp = False
    icl_use_ln_v = False
    icl_use_ln_qk = False

    share_covariate_attn = False
    share_covariate_mlp = False
    share_icl_attn = False
    share_icl_mlp = False
    
    use_output_mlp = False
        
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
        parts = [f"{self.model_type}", f"{self.d_embed}D", f"{self.max_seq_len}S", f"{self.n_blocks}L", f"{self.n_heads}H"]

        if self.model_type.startswith("icl"):
            parts.append("ICL")

            if getattr(self, "start_with_mlp", False):
                parts.append("mlpStart")
            if getattr(self, "end_with_mlp", False):
                parts.append("mlpEnd")
            if getattr(self, "update_targets", False):
                parts.append("updateTargets")
            if getattr(self, "icl_use_wv", False):
                parts.append("useWV")

        # Optional shared component flags
        if self.share_covariate_attn:
            parts.append("shareCovAttn")
        if self.share_covariate_mlp:
            parts.append("shareCovMLP")
        if self.share_icl_attn:
            parts.append("shareICLAttn")
        if self.share_icl_mlp:
            parts.append("shareICLMLP")
        if self.use_output_mlp:
            parts.append("outputMLP")

        # Normalization flags
        if self.icl_use_ln_mlp:
            parts.append("lnMLP")
        if self.icl_use_ln_v:
            parts.append("lnV")
        if self.icl_use_ln_qk:
            parts.append("lnQK")
        if self.icl_use_skip_mlp:
            parts.append("skipMLP")

        if self.dataset_name is not None:
            parts.append(f"ds={self.dataset_name}")

        return "_".join(parts)
