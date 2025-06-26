import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

class LightningWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        args,
        lr=1e-4
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def forward(self, x):
        return self.model(x)  # (B, S, V)

    def _compute_loss(self, logits, targets):
        logits = logits.reshape(-1, logits.size(-1)) # (B*S, V)
        targets = targets.reshape(-1) # (B*S,)
        return self.loss_fn(logits, targets)

    def training_step(self, batch, batch_idx):
        x = batch[:, :-1]
        y = batch[:, 1:]
        x[:, 0] = self.tokenizer.pad_token_id
        y[:, 0] = self.tokenizer.pad_token_id # Necessary for faster ICL training (otherwise we need to append a global start token)
        
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, :-1]
        y = batch[:, 1:]
        x[:, 0] = self.tokenizer.pad_token_id
        y[:, 0] = self.tokenizer.pad_token_id 
        
        logits = self(x)
        loss = self._compute_loss(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            betas=(0.9, 0.95), 
            weight_decay=0.1
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
