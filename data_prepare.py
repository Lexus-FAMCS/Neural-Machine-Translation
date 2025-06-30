import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece

def split_data(data_dir, output_dir, train_ratio, val_ratio):
    assert 0 < train_ratio < 1, "Train ratio must be between 0 and 1"
    assert 0 < val_ratio < 1, "Validation ratio must be between 0 and 1"
    assert train_ratio + val_ratio < 1, "Train and validation ratios must sum to less than 1"

    en_sentences, ru_sentences = [], []
    with open(data_dir + "/data.en", "r") as file:
        for sent in file:
            en_sentences.append(sent.strip())
    with open(data_dir + "/data.ru", "r") as file:
        for sent in file:
            ru_sentences.append(sent.strip())

    assert len(ru_sentences) == len(en_sentences), "Data mismatch between English and Russian sentences"
    
    indices = np.arange(len(ru_sentences))
    np.random.shuffle(indices)

    train_split = int(train_ratio * len(indices))
    val_split = int((train_ratio + val_ratio) * len(indices))

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    train_ru = [ru_sentences[i] for i in train_indices]
    train_en = [en_sentences[i] for i in train_indices]

    val_ru = [ru_sentences[i] for i in val_indices]
    val_en = [en_sentences[i] for i in val_indices]

    test_ru = [ru_sentences[i] for i in test_indices]
    test_en = [en_sentences[i] for i in test_indices]

    with open(output_dir + "/train.en", "w") as file:
        for sent in train_en:
            file.write(sent + "\n")
    with open(output_dir + "/train.ru", "w") as file:
        for sent in train_ru:
            file.write(sent + "\n")
    with open(output_dir + "/val.en", "w") as file:
        for sent in val_en:
            file.write(sent + "\n")
    with open(output_dir + "/val.ru", "w") as file:
        for sent in val_ru:
            file.write(sent + "\n")
    with open(output_dir + "/test.en", "w") as file:
        for sent in test_en:
            file.write(sent + "\n")
    with open(output_dir + "/test.ru", "w") as file:
        for sent in test_ru:
            file.write(sent + "\n")
    

def load_data(data_dir):
    train_en, train_ru = [], []
    val_en, val_ru = [], []
    test_en, test_ru = [], []

    with open(data_dir + "/train.en", "r") as file:
        for sent in file:
            train_en.append(sent.strip())
    with open(data_dir + "/train.ru", "r") as file:
        for sent in file:
            train_ru.append(sent.strip())

    with open(data_dir + "/val.en", "r") as file:
        for sent in file:
            val_en.append(sent.strip())
    with open(data_dir + "/val.ru", "r") as file:
        for sent in file:
            val_ru.append(sent.strip())

    with open(data_dir + "/test.en", "r") as file:
        for sent in file:
            test_en.append(sent.strip())
    with open(data_dir + "/test.ru", "r") as file:
        for sent in file:
            test_ru.append(sent.strip())

    assert len(train_en) == len(train_ru), "Train data mismatch"
    assert len(val_en) == len(val_ru), "Validation data mismatch"
    assert len(test_en) == len(test_ru), "Test data mismatch"

    return {
        "train": {"en": train_en, "ru": train_ru},
        "val": {"en": val_en, "ru": val_ru},
        "test": {"en": test_en, "ru": test_ru},
    }


def create_tokenizers(train_en, train_ru, args):
    en_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    en_tokenizer.pre_tokenizer = Whitespace()
    en_tokenizer.decoder = WordPiece(prefix="##")
    en_bpe_trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
        continuing_subword_prefix="##",
    )

    ru_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    ru_tokenizer.pre_tokenizer = Whitespace()
    ru_tokenizer.decoder = WordPiece(prefix="##")

    ru_bpe_trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
        continuing_subword_prefix="##"
    )

    en_tokenizer.train_from_iterator(train_en, en_bpe_trainer)
    ru_tokenizer.train_from_iterator(train_ru, ru_bpe_trainer)

    return en_tokenizer, ru_tokenizer

def wrap_sentence(sentence, tokenizer, max_len):
    ids = tokenizer.encode(sentence).ids[:max_len-2]

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    ids = [bos_id] + ids + [eos_id]

    pad_id = tokenizer.token_to_id('<pad>')
    ids = ids + [pad_id] * (max_len - len(ids))

    return torch.tensor(ids, dtype=torch.long)

def create_dataloaders(data, en_tokenizer, ru_tokenizer, batch_size, max_len):
    train_en, train_ru = data["train"]["en"], data["train"]["ru"]
    val_en, val_ru = data["val"]["en"], data["val"]["ru"]
    test_en, test_ru = data["test"]["en"], data["test"]["ru"]

    class EnRuDataset(torch.utils.data.Dataset):
        def __init__(self, en_sentences, ru_sentences, mode='train'):
            super().__init__()
            print(f'Creating dataset for {mode}...')
            self.en_tokens = [ wrap_sentence(s, en_tokenizer, max_len) 
                            for s in tqdm(en_sentences, desc=f'Wrapping english sentences') ]
            self.ru_tokens = [ wrap_sentence(s, ru_tokenizer, max_len) 
                            for s in tqdm(ru_sentences, desc=f'Wrapping russian sentences') ]
            
            # self.en_tokens = []
            # for i, s in enumerate(tqdm(en_sentences, desc=f'Wrapping english sentences')):
            #     if i > 1000:
            #         break
            #     self.en_tokens.append(wrap_sentence(s, en_tokenizer, max_len))
            # self.ru_tokens = []
            # for i, s in enumerate(tqdm(ru_sentences, desc=f'Wrapping russian sentences')):
            #     if i > 1000:
            #         break
            #     self.ru_tokens.append(wrap_sentence(s, ru_tokenizer, max_len))

        def __len__(self):
            return len(self.en_tokens)

        def __getitem__(self, idx):
            return self.en_tokens[idx], self.ru_tokens[idx]

    train_dataset = EnRuDataset(train_en, train_ru, mode='train')
    val_dataset = EnRuDataset(val_en, val_ru, mode='val')
    test_dataset = EnRuDataset(test_en, test_ru, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Neural Machine Translation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the raw data")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save processed data")
    parser.add_argument("--train_ratio", type=float, default=0.82, help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.03, help="Ratio of validation data")

    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size for tokenizers")
    parser.add_argument("--min_frequency", type=int, default=5, help="Minimum frequency for tokens in tokenizers")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=False)
    split_data(args.data_dir, args.output_dir, args.train_ratio, args.val_ratio)
    data = load_data(args.output_dir)
    en_tokenizer, ru_tokenizer = create_tokenizers(
        data["train"]["en"], data["train"]["ru"], args
    )
    en_tokenizer.save(args.output_dir + "/en_tokenizer.json")
    ru_tokenizer.save(args.output_dir + "/ru_tokenizer.json")
    
