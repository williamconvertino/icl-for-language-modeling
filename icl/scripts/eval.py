from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from icl.util import parse_experiment_args, resolve_checkpoint_path, LightningWrapper

def main():
    
    model, tokenizer, splits, args = parse_experiment_args()
    
    test_dataloader = splits["test"]

    strategy=args.strategy
    
    precision = "16-mixed"
    accelerator="gpu"

    checkpoint_name = args.checkpoint
    model_name = model.config.get_name()
    ckpt_path = resolve_checkpoint_path(model_name, checkpoint_name)

    print(f"Evaluating checkpoint: {ckpt_path}")

    lightning_model = LightningWrapper.load_from_checkpoint(
        ckpt_path=ckpt_path,
        model=model,
        tokenizer=tokenizer,
    )

    logger = TensorBoardLogger(save_dir=f"../logs/{model_name}/eval", name=checkpoint_name)

    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        precision=precision,
        logger=logger
    )

    results = trainer.test(lightning_model, dataloaders=test_dataloader, verbose=True)
    print(f"Test Results: {results}")

if __name__ == "__main__":
    main()