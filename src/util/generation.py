import os
import glob
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
from .lightning import LightningWrapper

def generate_from_model(model, args, splits, tokenizer):
    model_name = model.config.get_name()
    
    log_dir = os.path.join(args.output_dir, "logs")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints", model_name)
    os.makedirs(log_dir, exist_ok=True)

    ckpt_pattern = os.path.join(checkpoint_dir, "best-model-*.ckpt")
    ckpt_files = glob.glob(ckpt_pattern)
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint matching '{ckpt_pattern}' found.")
    best_ckpt_path = sorted(ckpt_files)[-1]  # take latest/best by name
    print(f"Loading best checkpoint from: {best_ckpt_path}")

    lightning_model = LightningWrapper.load_from_checkpoint(
        checkpoint_path=best_ckpt_path,
        model=model,
        tokenizer=tokenizer,
        args=args,
        lr=args.lr
    )

    logger = TensorBoardLogger(save_dir=log_dir, name=model_name)

    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision=args.precision,
        logger=logger
    )

    test_dataset = splits["test"]
    single_batch = Subset(test_dataset, range(args.batch_size))
    test_dataset = DataLoader(single_batch, batch_size=args.batch_size, num_workers=args.num_workers)

    outputs = trainer.predict(model=lightning_model, dataloaders=test_dataset)

    # Concatenate and optionally decode
    all_outputs = torch.cat(outputs, dim=0)
    if getattr(args, "decode", True):
        decoded = tokenizer.batch_decode(all_outputs, skip_special_tokens=True)
        return decoded
    else:
        return all_outputs
