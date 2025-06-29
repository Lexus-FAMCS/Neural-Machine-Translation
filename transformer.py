import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # (Batch, Seq_len, d_model) -> (Batch, num_heads, Seq_len, d_head)
        return x.view(x.size(0), x.size(1), self.num_heads, self.d_head).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attn_scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_weights, V).transpose(1, 2)
        output = self.W_o(output.reshape(output.size(0), output.size(1), self.d_model))

        return output
        

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_in)
        )

    def forward(self, x):
        return self.seq(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(device)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_hid, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_hid)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, src_mask=None):
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_hid, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_hid)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
        
# en_pad = en_tokenizer.token_to_id('<pad>')
# ru_pad = ru_tokenizer.token_to_id('<pad>')
class Transformer(nn.Module):
    def __init__(self, en_tokenizer, ru_tokenizer,
                    d_model, num_heads, d_hid,
                    dropout, num_layers, max_len,
                    device, logger
                ):
        super().__init__()
        
        self.device = device
        self.logger = logger
        self.en_tokenizer = en_tokenizer
        self.ru_tokenizer = ru_tokenizer
        src_vocab_size = en_tokenizer.get_vocab_size()
        tgt_vocab_size = ru_tokenizer.get_vocab_size()
        self.en_pad = en_tokenizer.token_to_id('<pad>')
        self.ru_pad = ru_tokenizer.token_to_id('<pad>')
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_hid, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_hid, dropout) for _ in range(num_layers)])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        enc_output = self.positional_encoding(self.encoder_embedding(src))
        dec_output = self.positional_encoding(self.decoder_embedding(tgt))

        src_mask = (src == self.en_pad).unsqueeze(1).unsqueeze(2).to(self.device)
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool().to(self.device)
        tgt_mask = tgt_mask | (tgt == self.ru_pad).unsqueeze(1).unsqueeze(2).to(self.device)
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.fc(dec_output)

    def generate(self, en_sentence, temperature, top_p, max_len):
        with torch.no_grad():
            input_tokens = self.en_tokenizer.encode(en_sentence).ids
            if max_len is None:
                max_len = 2 * len(input_tokens)
            
            input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            enc_output = self.encoder_embedding(input_tensor)
            enc_output = self.positional_encoding(enc_output)

            src_mask = (input_tensor == self.en_pad).unsqueeze(1).unsqueeze(2).to(self.device)
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)

            ru_bos = self.ru_tokenizer.token_to_id("<bos>")
            ru_eos = self.ru_tokenizer.token_to_id("<eos>")
            output_tokens = [ru_bos]
            output_embedding = torch.tensor([]).to(self.device)
            while output_tokens[-1] != ru_eos and len(output_tokens) <= max_len:
                last_tensor = torch.tensor(output_tokens[-1]).to(self.device)
                last_embedding = self.decoder_embedding(last_tensor).unsqueeze(0).unsqueeze(0)
                output_embedding = torch.cat([output_embedding, last_embedding], dim=1).to(self.device)
                dec_output = self.positional_encoding(output_embedding)
                tgt_mask = torch.triu(torch.ones(dec_output.size(1), dec_output.size(1)), diagonal=1).bool().to(self.device)
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
                logits = self.fc(dec_output[:, -1, :]).squeeze()
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                indices_to_keep = sorted_indices[cumulative_probs <= top_p]
                if len(indices_to_keep) == 0:
                    next_token = sorted_indices[0].item()
                else:
                    next_token = random.choice(indices_to_keep).item()
                output_tokens.append(next_token)
        return self.ru_tokenizer.decode(output_tokens)

