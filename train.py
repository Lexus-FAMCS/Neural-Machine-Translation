import data_prepare
import transformer

import argparse
import os
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.steps = {'train': 0, 'val': 0}

    def log(self, fn_name, tag, data, mode, increment_step):
        assert mode in self.steps, f"Mode must be one of {list(self.steps.keys())}"
        step = self.steps[mode]
        fn = getattr(self.writer, fn_name)
        fn(tag, data, step)
        if increment_step:
            self.steps[mode] += 1

    def close(self):
        self.writer.close()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def val_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    num = 0
    with torch.no_grad():
        for en_batch, ru_batch in tqdm(val_loader, desc='Validating', leave=False):
            en_batch, ru_batch = en_batch.to(device), ru_batch.to(device)

            output = model(en_batch, ru_batch[:, :-1]).transpose(1, 2)
            reference = ru_batch[:, 1:]
            loss = criterion(output, reference)
            val_loss += loss.item()
            num += (reference != model.ru_pad).sum().item()

    val_loss /= num
    return val_loss


def generate_samples(model, test_loader, num_samples, device):
    with torch.no_grad():
        for en_batch, ru_batch in test_loader:
            en_batch, ru_batch = en_batch.to(device), ru_batch.to(device)
            for j, (en_tokens, ru_tokens) in enumerate(zip(en_batch, ru_batch)):
                if j > num_samples:
                    break
                en_sentence = model.en_tokenizer.decode(en_tokens.tolist())
                ru_sentence = model.ru_tokenizer.decode(ru_tokens.tolist())
                gen = model.generate(en_sentence)

                text = (
                    f"**English:** {en_sentence}\n\n"
                    f"**Original Russian:** {ru_sentence}\n\n"
                    f"**Generated Russian:** {gen}"
                )
                model.logger.log('add_text', f'Sample {j}', text, mode='val', increment_step=False)
            break


def train_epoch(model, 
                train_loader, val_loader, test_loader,
                opt, criterion, scheduler, device,
                train_log_interval, eval_log_interval, num_samples):
    model.train()   
    train_losses = []
    val_losses = []
    train_loss = 0.0
    num = 0
    for i, (en_batch, ru_batch) in enumerate(tqdm(train_loader, desc='Training'), 1):   
        if i % train_log_interval == 0:
            train_loss /= num
            train_losses.append(train_loss)
            model.logger.log('add_scalar', 'Loss/train', train_loss, mode='train', increment_step=True)
            train_loss = 0.0
            num = 0

            grads = defaultdict(int)
            for name, p in model.named_parameters():
                tmp = name.split('.')
                if p.grad is None:
                    continue
                if len(tmp) > 1 and tmp[1].isdecimal():
                    grads[''.join(tmp[:2])] += torch.norm(p.grad)
                else:
                    grads[tmp[0]] += torch.norm(p.grad)

            for name, grad in grads.items():
                model.logger.log('add_scalar', f'Grads/{name}', grad, mode='train', increment_step=False)
            
        if i % eval_log_interval == 0:
            val_loss = val_epoch(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            if scheduler is not None:
                try:
                    scheduler.step()
                except:
                    scheduler.step(val_losses[-1])
            model.logger.log('add_scalar', 'Loss/val', val_loss, mode='val', increment_step=True)
            model.logger.log('add_scalar', 'Learning Rate', opt.param_groups[0]['lr'], mode='val', increment_step=False)
            generate_samples(model, test_loader, num_samples, device)
            model.train()
        
        en_batch, ru_batch = en_batch.to(device), ru_batch.to(device)

        output = model(en_batch, ru_batch[:, :-1]).transpose(1, 2)
        reference = ru_batch[:, 1:]
        loss = criterion(output, reference)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()
        num += (reference != model.ru_pad).sum().item() 

    return train_losses, val_losses

def train(args):
    set_seed(args.seed)

    data = data_prepare.load_data(args.data_dir)
    en_tokenizer = Tokenizer.from_file(args.data_dir + '/en_tokenizer.json')
    ru_tokenizer = Tokenizer.from_file(args.data_dir + '/ru_tokenizer.json')
    train_loader, val_loader, test_loader = data_prepare.create_dataloaders(
        data, en_tokenizer, ru_tokenizer, 
        args.batch_size, args.wrap_max_len
    )

    translator = transformer.Transformer(
        en_tokenizer, ru_tokenizer,
        d_model=args.d_model, num_heads=args.num_heads,
        d_hid=args.d_hid, dropout=args.dropout,
        num_layers=args.num_layers, max_len=args.model_max_len,
        device=args.device, logger=Logger(f'runs/{args.output_dir}')
    ).to(args.device)
    opt = torch.optim.AdamW(translator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=translator.ru_pad, reduction='sum')
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=0, min_lr=1e-6)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")

        _, val_losses = train_epoch(
            translator,
            train_loader, val_loader, test_loader,
            opt, criterion, scheduler,
            args.device, args.train_log_interval,
            args.eval_log_interval, args.num_samples
        )
        print(f'Epoch {epoch}, validation losses: {val_losses}')
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            print(f"Saving model on {epoch} epoch...")
            torch.save(translator.state_dict(), f'runs/{args.output_dir}/best_model.pth')
            print("Model saved successfully!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Neural Machine Translation")
    
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the training, validation, and test data files and tokenizers.")
    
    parser.add_argument("--wrap_max_len", type=int, default=32,
                        help="Maximum length of the input sequences after wrapping. Sequences longer than this will be truncated.")
    parser.add_argument("--model_max_len", type=int, default=64,
                        help="Maximum length of the model input sequences.")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of layers in the Transformer model")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Dimension of the embedding")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads in the Transformer model")
    parser.add_argument("--d_hid", type=int, default=2048,
                        help="Hidden dimension of the feedforward network in the Transformer model")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for the Transformer model")

    parser.add_argument("--train_log_interval", type=int, default=150, 
                        help="Interval for logging training progress")
    parser.add_argument("--eval_log_interval", type=int, default=500, 
                        help="Interval for logging validation progress")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate during validation")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save the model and TensorBoard logs")
    
    args = parser.parse_args()
    train(args)    