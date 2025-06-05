import os
from argparse import ArgumentParser

from icl.config import Config
from icl.models import Transformer, ICL
from icl.data import Tokenizer, TinyStoriesDataset, SlimPajamaDataset

def parse_experiment_args():
    
    parser = ArgumentParser()
    
    # Model
    parser.add_argument("--preset", type=str, required=True) # transformer or icl
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="tinystories") # tinystories or slimpajama
    
    # Training
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_check_interval", type=float, default=0.2)
    parser.add_argument("--strategy", type=str, default="fsdp") # ddp for small models, fsdp for large
    parser.add_argument("--batch_size", type=int, default=16)
    # parser.add_argument("--devices", type=int, default=None) # Not useable when doing torchrun
    
    # Eval
    parser.add_argument("--checkpoint", type=str) # last, best, epoch_x
    
    args, config_options = parser.parse_known_args()
    config_override_dict = parse_config_args(config_options)
    
    config = Config(preset_name=args.preset, config_override_dict=config_override_dict, dataset=args.dataset)
    
    if config.model_type == "transformer":
        model = Transformer(config)
    elif config.model_type == "icl":
        model = ICL(config)
    else:
        raise ValueError(f"Invalid model type: '{config.model_type}'.")

    tokenizer = Tokenizer()
    
    if config.dataset == "tinystories":
        splits = TinyStoriesDataset.get_splits(tokenizer, args.batch_size)
    elif config.dataset == "slimpajama":
        splits = SlimPajamaDataset.get_splits(tokenizer, args.batch_size)
    else:
        raise ValueError(f"Dataset '{config.dataset}' is not recognized.")
    
    return model, tokenizer, splits, args
    
def parse_config_args(config_args):
    config_override_dict = {}
    for i in range(0, len(config_args), 2):
        key = config_args[i].lstrip('--')
        value = config_args[i+1]
        
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        elif value.lower() in {"true", "false"}:
            value = value.lower() == "true"
            
        config_override_dict[key] = value
    return config_override_dict

def resolve_checkpoint_path(model_name, checkpoint_name):
    checkpoint_dir = f"../checkpoints/{model_name}"

    if checkpoint_name == "last":
        ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")

    elif checkpoint_name == "best":
        candidates = sorted(
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("best-model") and f.endswith(".ckpt")
        )
        if not candidates:
            raise FileNotFoundError("No best checkpoint found.")
        ckpt_path = os.path.join(checkpoint_dir, candidates[0])

    elif checkpoint_name.startswith("epoch_"):
        try:
            epoch_num = int(checkpoint_name.split("_")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid format for epoch checkpoint: {checkpoint_name}. Expected format: epoch_<int>")

        pattern = f"epoch-{epoch_num:02d}"
        candidates = sorted(
            f for f in os.listdir(checkpoint_dir)
            if f.startswith(pattern) and f.endswith(".ckpt")
        )
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found for epoch {epoch_num}.")
        ckpt_path = os.path.join(checkpoint_dir, candidates[0])

    else:
        raise ValueError(f"Unsupported checkpoint name: {checkpoint_name}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    return ckpt_path
