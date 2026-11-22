
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, trainers
from tokenizers import pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import text_loader

# CONFIGURAZIONE
VOCAB_SIZE = 32000

OUTPUT = ROOT / "tokenizer" / "vocab.json"

# CARICAMENTO DATASET DA TEXT_LOADER
text = text_loader.get_data()

if not text or not text.strip():
    raise ValueError("Il dataset è vuoto!")

print("Dati caricati:", len(text), "caratteri")

# CREAZIONE TOKENIZER
tokenizer = Tokenizer(
    models.BPE(unk_token="<unk>")
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = decoders.ByteLevel()

# TRAINER
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>"
    ],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

# ADDESTRAMENTO
print("Training tokenizer...")

tokenizer.train_from_iterator(
    text.splitlines(),
    trainer=trainer
)

# SALVATAGGIO
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(OUTPUT))

print("Tokenizer addestrato!")
print("Vocabolario:", tokenizer.get_vocab_size())
print("Salvato in:", OUTPUT)

# TEST
test_text = "Ciao, sono Trio!"
encoded = tokenizer.encode(test_text)

print("Token:", encoded.tokens)
print("ID:", encoded.ids)
print("Decodificato:", tokenizer.decode(encoded.ids))
