import os
from datasets import load_dataset, concatenate_datasets
from .disk_dataset import DiskDataset, DATASET_DIR

HUGGINGFACE_PATH = "DKYoon/SlimPajama-6B"

class SlimPajamaDataset(DiskDataset):
    
    def __init__(self, split, tokenizer, batch_size, do_shuffle=False, allow_overlap=True):
    
        file_path = f"{DATASET_DIR}/slimpajama/{split}.bin"    
        
        if not os.path.exists(file_path):

            print(f"Creating SlimPajama [{split}] dataset files...")
            
            dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"{DATASET_DIR}/raw")
            
            train_dataset = dataset["train"]
            test_dataset = dataset["test"]
            val_dataset = dataset["validation"]
            
            DiskDataset.generate_data_file(train_dataset, f"{DATASET_DIR}/slimpajama/train.bin", tokenizer)
            DiskDataset.generate_data_file(test_dataset, f"{DATASET_DIR}/slimpajama/test.bin", tokenizer)
            DiskDataset.generate_data_file(val_dataset, f"{DATASET_DIR}/slimpajama/val.bin", tokenizer)
    
        super().__init__(file_path, tokenizer, batch_size, do_shuffle=do_shuffle, allow_overlap=allow_overlap)

    def get_splits(tokenizer, batch_size):
        return {
            "train": SlimPajamaDataset("train", tokenizer, batch_size, do_shuffle=False, allow_overlap=True),
            "val": SlimPajamaDataset("val", tokenizer, batch_size, do_shuffle=False, allow_overlap=False),
            "test": SlimPajamaDataset("test", tokenizer, batch_size, do_shuffle=False, allow_overlap=False)
        }