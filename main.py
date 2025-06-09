from argparse import ArgumentParser

from src.config import Config
from src.models import Transformer, ICL
from src.data import Tokenizer, TinyStoriesDataset, SlimPajamaDataset

from src.util import train_model

def get_args():
    parser = ArgumentParser()

    # General Info
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval", "generate"], help="Execution mode.")
    parser.add_argument("--preset", type=str, default="transformer", choices=["transformer", "icl"], help="Model preset to use.")
    parser.add_argument("--override", type=str, default=None, help="Override config options, e.g., 'n_blocks=5,n_heads=4'.")
    parser.add_argument("--dataset", type=str, default="slimpajama", choices=["tinystories", "slimpajama"], help="Dataset to use for training.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load: 'last', 'best', or 'epoch_x'.")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per GPU.")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument("--val_check_interval", type=float, default=0.2, help="Fraction of an epoch between validation checks.")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "fsdp"], help="Training strategy.")
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["16-mixed", "bf16-mixed", "32-true", "64-true"], help="Floating point precision: '16-mixed', 'bf16-mixed', '32-true', or '64-true'.")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "tpu"], help="Device to run on.")

    # Generation
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt string for text generation.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of dataset samples to generate from.")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum number of tokens to generate.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p value.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more random).")

    args = parser.parse_args()
        
    return args

def main():
    
    # Initial Setup
    
    args = get_args()
    
    config = Config(preset_name=args.preset, config_override=args.override, dataset=args.dataset)
    
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
    
    # Training
    if args.mode == "train":
        train_model(model, args, splits, tokenizer)
    
    # Evaluation
    if args.mode == "eval":
        raise NotImplementedError("Evaluation mode is not yet implemented.")

    # Generation
    if args.mode == "generate":
        raise NotImplementedError("Evaluation mode is not yet implemented.")
    
if __name__ == "__main__":
    main()