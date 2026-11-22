
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent

TOKENIZER_PATH = ROOT / "vocab.json"

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))


def encode(text):
    return tokenizer.encode(text).ids


def decode(ids):
    return tokenizer.decode(ids)


def tokenize(text):
    return tokenizer.encode(text).tokens


def vocab_size():
    return tokenizer.get_vocab_size()
