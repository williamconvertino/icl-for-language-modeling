from .tokenizer import Tokenizer
from .datamodule import PLDataModule

TOKENIZER = Tokenizer()

def SLIMPAJAMA_DM(seq_len, batch_size, num_workers):
    return PLDataModule("DKYoon/SlimPajama-6B", TOKENIZER, seq_len, batch_size, num_workers)

def TINYSTORIES_DM(seq_len, batch_size, num_workers):
    return PLDataModule("roneneldan/TinyStories", TOKENIZER, seq_len, batch_size, num_workers)

def WIKITEXT_DM(seq_len, batch_size, num_workers):
    return PLDataModule("mindchain/wikitext2", TOKENIZER, seq_len, batch_size, num_workers)