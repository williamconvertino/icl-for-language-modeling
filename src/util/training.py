import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from .lightning import LightningWrapper

def train_model(model, args, datamodule, tokenizer):

    lr=args.lr
    strategy=args.strategy
    val_check_interval=args.val_check_interval
    max_epochs=args.max_epochs
    precision=args.precision
    accelerator=args.accelerator

    model_name = model.config.get_name()
    
    log_dir = os.path.join(args.output_dir, "logs")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints", model_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    lightning_model = LightningWrapper(model=model, tokenizer=tokenizer, lr=lr)

    best_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-model-{epoch:02d}-{step:06d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    epoch_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}-{val_loss:.2f}",
        every_n_epochs=1,
        save_top_k=-1,
        save_on_train_epoch_end=True
    )

    callbacks = [best_ckpt_callback, epoch_ckpt_callback]

    last_ckpt_path = f"{checkpoint_dir}/last.ckpt"
    
    if os.path.exists(last_ckpt_path):
        print(f"Training from checkpoint: {last_ckpt_path}")
    else:
        print(f"No checkpoint found, training from scratch")
        last_ckpt_path = None

    logger = TensorBoardLogger(save_dir=log_dir, name=model_name)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        strategy=strategy,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=val_check_interval
    )

    trainer.fit(
        lightning_model, 
        datamodule=datamodule,
        ckpt_path=last_ckpt_path
        )
    
    print(f"Training complete ({max_epochs} epochs)")