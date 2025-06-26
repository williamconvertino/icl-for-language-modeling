import os
import math
import torch
import numpy as np
from tqdm import tqdm
import re

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

class DiskDataset(torch.utils.data.Dataset):
    replacements = {
        "�": "", "â": "", "€": "", "œ": "", "™": "",
        "``": '"', "''": '"', "“": '"', "”": '"', "‘": "'", "’": "'",
        "…": "..."
    }
    
    re_replace = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))
    stride_multiplier = 0.5

    def __init__(self, file_path, tokenizer, max_seq_len, allow_overlap=True):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = int(max_seq_len * self.stride_multiplier)
        self.allow_overlap = allow_overlap

        self.data = np.memmap(file_path, dtype="int32", mode="r")
        self.file_size = len(self.data)

        # Precompute valid start indices
        self.start_indices = list(range(0, self.file_size - max_seq_len + 1, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        seq = self.data[start : start + self.max_seq_len].copy()
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        if not self.allow_overlap:
            indices = np.where(seq == eos_token_id)[0]
            if indices.size > 0:
                first_eos = indices[0]
                seq[first_eos + 1:] = pad_token_id

        return torch.tensor(seq).long()

    @staticmethod
    def preprocess(examples, tokenizer, column, separate_lines=True):
        texts = [DiskDataset.re_replace.sub(lambda m: DiskDataset.replacements[m.group()], text) for text in examples[column]]
        examples["input_ids"] = tokenizer.encode(texts, eos=separate_lines, bos=separate_lines)
        return examples

    @staticmethod
    def generate_data_file(dataset, file_path, tokenizer, separate_lines=True, column="text"):
        dataset = dataset.map(lambda x: DiskDataset.preprocess(x, tokenizer, column, separate_lines), batched=True, remove_columns=[column])
        file_size = sum(len(example) for example in dataset["input_ids"])

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        memmap_array = np.memmap(file_path, dtype="int32", mode="w+", shape=(file_size,))

        buffer = []
        write_pointer = 0
        buffer_size = 1024

        for sequence in tqdm(dataset["input_ids"], desc="Generating dataset files"):
            buffer.extend(sequence)
            if len(buffer) >= buffer_size:
                memmap_array[write_pointer: write_pointer + len(buffer)] = buffer
                write_pointer += len(buffer)
                buffer = []

        if buffer:
            memmap_array[write_pointer: write_pointer + len(buffer)] = buffer

        memmap_array.flush()
        return memmap_array
