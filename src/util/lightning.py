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
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = self(x)
        loss = self._compute_loss(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        
        max_new_tokens = batch.shape[1] + self.model.config.max_seq_len
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        
        top_p = self.args.top_p
        temperature = self.args.temperature

        input_ids = batch
        B = input_ids.size(0)

        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_new_tokens):
            logits = self.model(input_ids)
            next_token_logits = logits[:, -1, :]

            probs = F.softmax(next_token_logits / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)

            mask = cumulative_probs <= top_p
            mask[:, 0] = 1

            probs_filtered = torch.where(mask, sorted_probs, torch.tensor(0.0, device=self.device))
            probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

            sampled = torch.multinomial(probs_filtered, num_samples=1).squeeze(-1)
            next_tokens = sorted_indices.gather(dim=1, index=sampled.unsqueeze(-1)).squeeze(-1)

            next_tokens = torch.where(finished, pad_token_id, next_tokens)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)

            finished = finished | (next_tokens == eos_token_id)

            if finished.all():
                break

        return input_ids
        
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
