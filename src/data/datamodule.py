import os
import json
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import torch
import pytorch_lightning as pl

class HFDataset(IterableDataset):
    def __init__(self, hf_path, split, tokenizer, seq_len, rank, world_size):
        self.hf_path = hf_path
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        self.dataset = load_dataset(hf_path, split=split, streaming=False, cache_dir=self.cache_dir)
        
        self._estimate_length()
        
    def _estimate_length(self):
        cache_file = os.path.join(
            self.cache_dir, f"{self.hf_path.replace('/', '_')}_{self.split}_{self.seq_len}_length.json"
        )

        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached = json.load(f)
            self._length = cached["length"] // self.world_size
            print(f"Loaded cached length: {self._length * self.world_size} total chunks")
        else:
            print(f"Estimating dataset length for split '{self.split}'...")
            total_chars = 0
            for example in tqdm(self.dataset, desc=f"Processing {self.split}"):
                total_chars += len(example["text"])
            total_chunks = total_chars // self.seq_len
            self._length = total_chunks // self.world_size

            with open(cache_file, "w") as f:
                json.dump({"length": total_chunks}, f)
            print(f"Saved cached length: {total_chunks} total chunks")

    def __len__(self):
        return self._length

    def __iter__(self):
        iterable = iter(self.dataset.shard(num_shards=self.world_size, index=self.rank))

        buffer = ""
        for item in iterable:
            buffer += item["text"]
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(
                    self.tokenizer(chunk, truncation=True, padding="max_length", max_length=self.seq_len)
                )

class PLDataModule(pl.LightningDataModule):
    def __init__(self, hf_path, tokenizer, seq_len, batch_size, num_workers):
        super().__init__()
        self.hf_path = hf_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        rank = self.trainer.local_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1
        
        if stage == "fit" or stage is None:
            
            self.train_dataset = HFDataset(
                hf_path=self.hf_path,
                split="train",
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                rank=rank,
                world_size=world_size
            )
            
            self.val_dataset = HFDataset(
                hf_path=self.hf_path,
                split="validation",
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                rank=rank,
                world_size=world_size
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = HFDataset(
                hf_path=self.hf_path,
                split="test",
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                rank=rank,
                world_size=world_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        
        
        