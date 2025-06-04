from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from icl.config import Config
from icl.util import parse_config_args, LightningWrapper
from icl.models import Transformer
from icl.data import TinyStoriesDataset, Tokenizer

def main():
    parser = ArgumentParser()
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--strategy", type=str, default="fsdp")
    parser.add_argument("--batch_size", type=int, default=16)
    
    args, config_options = parser.parse_known_args()
    config_override_dict = parse_config_args(config_options)
    
    config = Config(preset_name=args.preset, config_override_dict=config_override_dict)
    
    if config.model_type == "transformer":
        model = Transformer(config)
    elif config.model_type == "icl":
        pass
    else:
        ValueError(f"Invalid model type: '{config.model_type}'.")

    tokenizer = Tokenizer()
    splits = TinyStoriesDataset.get_splits(tokenizer, args.batch_size)

    train_model(
        model,
        tokenizer,
        splits["train"],
        splits["val"],
        max_epochs=args.max_epochs, 
        lr=args.lr, 
        devices=args.devices,
        strategy=args.strategy
        )

def train_model(
    model,
    tokenizer,
    train_dataloader,
    val_dataloader=None,
    max_epochs=10,
    lr=1e-4,
    precision="16-mixed",
    accelerator="gpu",
    devices=-1,
    strategy="fsdp" # Use ddp for smaller models for a speedup
):
    lightning_model = LightningWrapper(model=model, tokenizer=tokenizer, lr=lr)
    
    model_name = model.config.get_name()

    log_dir=f"../logs/{model_name}"
    checkpoint_dir=f"../checkpoints/{model_name}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    logger = TensorBoardLogger(save_dir=log_dir, name=model_name)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )

    trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    main()