import os
from datasets import load_dataset, concatenate_datasets
from .disk_dataset import DiskDataset, DATASET_DIR

HUGGINGFACE_PATH = "euirim/goodwiki"

class GoodWikiDataset(DiskDataset):
    
    def __init__(self, split, tokenizer, context_size, allow_overlap=True):
    
        file_path = f"{DATASET_DIR}/goodwiki/{split}.bin"    
        
        if not os.path.exists(file_path):

            print(f"Creating GoodWiki [{split}] dataset files...")
            
            dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"{DATASET_DIR}/raw", split="train")
            
            val_data = dataset.select(range(2_000)) # 2k val
            test_data = dataset.select(range(2_000, 4_000)) # 2k test
            train_data = dataset.select(range(4_000, len(dataset))) # 40k train     

            DiskDataset.generate_data_file(train_data, f"{DATASET_DIR}/goodwiki/train.bin", tokenizer, column="markdown")
            DiskDataset.generate_data_file(test_data, f"{DATASET_DIR}/goodwiki/test.bin", tokenizer, column="markdown")
            DiskDataset.generate_data_file(val_data, f"{DATASET_DIR}/goodwiki/val.bin", tokenizer, column="markdown")
    
        super().__init__(file_path, tokenizer, context_size, allow_overlap=allow_overlap)

    def get_splits(tokenizer, max_seq_len):
        return {
            "train": GoodWikiDataset("train", tokenizer, max_seq_len, allow_overlap=True),
            "val": GoodWikiDataset("val", tokenizer, max_seq_len, allow_overlap=False),
            "test": GoodWikiDataset("test", tokenizer, max_seq_len, allow_overlap=False)
        }