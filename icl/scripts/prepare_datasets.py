import os
from icl.data import TinyStoriesDataset, SlimPajamaDataset, Tokenizer

def main():
    tokenizer = Tokenizer()
    
    # if not os.path.exists("../datasets/slimpajama"):
    #     SlimPajamaDataset.get_splits(tokenizer=tokenizer, batch_size=16)
    #     print("Generated SlimPajama Dataset")
    # else:
    #     print("SlimPajama found, skipping...")
    
    if not os.path.exists("../datasets/tinystories"):
        TinyStoriesDataset.get_splits(tokenizer=tokenizer, batch_size=16)
        print("Generated TinyStories Dataset")
    else:
        print("TinyStories found, skipping...")

if __name__ == "__main__":
    main()