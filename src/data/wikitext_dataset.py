import os
from datasets import load_dataset, concatenate_datasets
from .disk_dataset import DiskDataset, DATASET_DIR

HUGGINGFACE_PATH = "mindchain/wikitext2"

class WikiTextDataset(DiskDataset):
    
    def __init__(self, split, tokenizer, context_size, allow_overlap=True):
    
        file_path = f"{DATASET_DIR}/wikitext/{split}.bin"    
        
        if not os.path.exists(file_path):

            print(f"Creating WikiText [{split}] dataset files...")
            
            dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"{DATASET_DIR}/raw", trust_remote_code=True)
            dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
            
            train_test_splits = dataset.train_test_split(test_size=10000, shuffle=True, seed=42)

            train_dataset = train_test_splits["train"]
            test_dataset = train_test_splits["test"]
            
            train_val_split = train_dataset.train_test_split(test_size=10000, shuffle=True, seed=42)
            train_dataset = train_val_split["train"]
            val_dataset = train_val_split["test"]

            DiskDataset.generate_data_file(train_dataset, f"{DATASET_DIR}/wikitext/train.bin", tokenizer)
            DiskDataset.generate_data_file(test_dataset, f"{DATASET_DIR}/wikitext/test.bin", tokenizer)
            DiskDataset.generate_data_file(val_dataset, f"{DATASET_DIR}/wikitext/val.bin", tokenizer)
    
        super().__init__(file_path, tokenizer, context_size, allow_overlap=allow_overlap)

    def get_splits(tokenizer, max_seq_len):
        return {
            "train": WikiTextDataset("train", tokenizer, max_seq_len, allow_overlap=True),
            "val": WikiTextDataset("val", tokenizer, max_seq_len, allow_overlap=False),
            "test": WikiTextDataset("test", tokenizer, max_seq_len, allow_overlap=False)
        }