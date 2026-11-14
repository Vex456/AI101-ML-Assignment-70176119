"""
Task C1: English-to-Urdu Neural Machine Translation
AI-101 Machine Learning Assignment, Fall 2026
Department of Intelligent Systems, The University of Lahore
Instructor: Dr. Muhammad Awais
"""

import os
import re
import json
import random
import unicodedata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import sacrebleu
from transformers import MarianMTModel, MarianTokenizer

# Global seeds & device
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs("C1_English_Urdu_NMT/figures", exist_ok=True)
os.makedirs("C1_English_Urdu_NMT/data", exist_ok=True)

# Special Tokens
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

print("=" * 60, flush=True)
print("1. INGESTING & FILTERING OPUS-100 (EN-UR)", flush=True)
print("=" * 60, flush=True)

dataset = load_dataset("Helsinki-NLP/opus-100", "en-ur")
raw_train = dataset['train']
raw_val = dataset['validation']
raw_test = dataset['test']

print(f"Raw counts -> Train: {len(raw_train)}, Val: {len(raw_val)}, Test: {len(raw_test)}", flush=True)

def has_urdu(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def clean_pairs(hf_split, max_len=30, max_ratio=3.0):
    pairs = []
    seen = set()
    initial_count = len(hf_split)
    
    empty_removed = 0
    dup_removed = 0
    ratio_removed = 0
    length_removed = 0
    no_urdu_removed = 0
    
    for item in hf_split:
        en = item['translation']['en'].strip()
        ur = item['translation']['ur'].strip()
        
        # 1. Empty filter
        if not en or not ur:
            empty_removed += 1
            continue
            
        # 2. Duplicate filter
        pair_key = (en, ur)
        if pair_key in seen:
            dup_removed += 1
            continue
        seen.add(pair_key)
        
        en_tokens = en.split()
        ur_tokens = ur.split()
        
        # 3. Urdu script verification
        if not has_urdu(ur):
            no_urdu_removed += 1
            continue
            
        # 4. Length check
        if len(en_tokens) > max_len or len(ur_tokens) > max_len:
            length_removed += 1
            continue
            
        # 5. Length ratio check
        len_en, len_ur = max(1, len(en_tokens)), max(1, len(ur_tokens))
        ratio = max(len_en, len_ur) / min(len_en, len_ur)
        if ratio > max_ratio:
            ratio_removed += 1
            continue
            
        pairs.append((en, ur))
        
    stats = {
        'initial': initial_count,
        'empty_removed': empty_removed,
        'duplicate_removed': dup_removed,
        'no_urdu_removed': no_urdu_removed,
        'length_exceeded_removed': length_removed,
        'ratio_exceeded_removed': ratio_removed,
        'retained': len(pairs)
    }
    return pairs, stats

train_pairs, train_filter_stats = clean_pairs(raw_train)
val_pairs, _ = clean_pairs(raw_val)
test_pairs, _ = clean_pairs(raw_test)

print(f"Train filter statistics: {train_filter_stats}", flush=True)
print(f"Retained -> Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}", flush=True)

# Select working training subset (20,000 pairs for rapid, stable convergence on CPU)
SUBSET_SIZE = 20000
train_subset = train_pairs[:SUBSET_SIZE]
print(f"Working training subset: {len(train_subset)} pairs", flush=True)

# ==============================================================================
# 2. VOCABULARY & TOKENIZATION
# ==============================================================================
print("\n" + "=" * 60, flush=True)
print("2. BUILDING VOCABULARY & TOKENIZER", flush=True)
print("=" * 60, flush=True)

class SimpleTokenizer:
    def __init__(self, max_vocab=10000):
        self.max_vocab = max_vocab
        self.word2idx = {'<pad>': PAD_IDX, '<sos>': SOS_IDX, '<eos>': EOS_IDX, '<unk>': UNK_IDX}
        self.idx2word = {PAD_IDX: '<pad>', SOS_IDX: '<sos>', EOS_IDX: '<eos>', UNK_IDX: '<unk>'}
        self.word_counts = {}

    def build_vocab(self, sentences):
        for s in sentences:
            tokens = s.strip().lower().split()
            for t in tokens:
                self.word_counts[t] = self.word_counts.get(t, 0) + 1
        sorted_tokens = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for t, _ in sorted_tokens[:self.max_vocab - 4]:
            idx = len(self.word2idx)
            self.word2idx[t] = idx
            self.idx2word[idx] = t

    def encode(self, sentence, add_special=True):
        tokens = sentence.strip().lower().split()
        idxs = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        if add_special:
            return [SOS_IDX] + idxs + [EOS_IDX]
        return idxs

    def decode(self, idxs):
        words = []
        for i in idxs:
            if i == EOS_IDX:
                break
            if i not in [PAD_IDX, SOS_IDX]:
                words.append(self.idx2word.get(i, '<unk>'))
        return ' '.join(words)

src_tokenizer = SimpleTokenizer(max_vocab=8000)
tgt_tokenizer = SimpleTokenizer(max_vocab=8000)

src_tokenizer.build_vocab([p[0] for p in train_subset])
tgt_tokenizer.build_vocab([p[1] for p in train_subset])

print(f"Source (EN) Vocab size: {len(src_tokenizer.word2idx)}", flush=True)
print(f"Target (UR) Vocab size: {len(tgt_tokenizer.word2idx)}", flush=True)

# Compute Test OOV Rate
test_en_tokens = [w for p in test_pairs for w in p[0].strip().lower().split()]
test_ur_tokens = [w for p in test_pairs for w in p[1].strip().lower().split()]

en_oov = sum(1 for w in test_en_tokens if w not in src_tokenizer.word2idx) / len(test_en_tokens) * 100
ur_oov = sum(1 for w in test_ur_tokens if w not in tgt_tokenizer.word2idx) / len(test_ur_tokens) * 100

print(f"English Test OOV Rate: {en_oov:.2f}% | Urdu Test OOV Rate: {ur_oov:.2f}%", flush=True)

# ==============================================================================
# 3. PYTORCH SEQ2SEQ MODEL WITH BAHDANAU ATTENTION
# ==============================================================================
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_tok, tgt_tok):
        self.data = [(src_tok.encode(p[0]), tgt_tok.encode(p[1])) for p in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]
    
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    
    src_padded = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_padded = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    
    for i, (s, t) in enumerate(batch):
        src_padded[i, :len(s)] = torch.tensor(s)
        tgt_padded[i, :len(t)] = torch.tensor(t)
        
    return src_padded, tgt_padded, torch.tensor(src_lens)

train_loader = DataLoader(TranslationDataset(train_subset, src_tokenizer, tgt_tokenizer), batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(TranslationDataset(val_pairs[:1000], src_tokenizer, tgt_tokenizer), batch_size=64, shuffle=False, collate_fn=collate_fn)

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(embedded)
        # Combine forward & backward hidden states
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

# Bahdanau Additive Attention
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.U_a = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [B, H], encoder_outputs: [B, S, 2H]
        src_len = encoder_outputs.shape[1]
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W_a(hidden_expanded) + self.U_a(encoder_outputs))
        attention = torch.softmax(self.v_a(energy).squeeze(2), dim=1)
        return attention

# Attention Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, attention, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU((hidden_dim * 2) + emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear((hidden_dim * 2) + hidden_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, encoder_outputs):
        input_tok = input_tok.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_tok))
        a = self.attention(hidden, encoder_outputs) # [B, S]
        
        # Weighted context vector
        context = torch.bmm(a.unsqueeze(1), encoder_outputs) # [B, 1, 2H]
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))
        
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden.squeeze(0), a

# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        input_tok = tgt[:, 0]
        for t in range(1, tgt_len):
            prediction, hidden, _ = self.decoder(input_tok, hidden, encoder_outputs)
            outputs[:, t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1
        return outputs

EMB_DIM = 128
HIDDEN_DIM = 256

attn = BahdanauAttention(HIDDEN_DIM)
enc = Encoder(len(src_tokenizer.word2idx), EMB_DIM, HIDDEN_DIM)
dec = Decoder(len(tgt_tokenizer.word2idx), EMB_DIM, HIDDEN_DIM, attn)
model = Seq2Seq(enc, dec, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
EPOCHS = 4
train_losses = []
val_losses = []

print("\n" + "=" * 60, flush=True)
print("3. TRAINING PYTORCH ATTENTION SEQ2SEQ (MODEL 1)", flush=True)
print("=" * 60, flush=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for src, tgt, _ in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt, _ in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

# Save Model Checkpoint
torch.save(model.state_dict(), "C1_English_Urdu_NMT/model1_attention.pt")

# Plot Loss Curves (Figure 1)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, EPOCHS+1), train_losses, 'b-o', label='Training Loss', lw=2)
ax.plot(range(1, EPOCHS+1), val_losses, 'r--s', label='Validation Loss', lw=2)
ax.set_title("Training & Validation Loss Curves (Model 1 Attention Seq2Seq)", fontsize=11, fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross Entropy Loss")
ax.legend(frameon=True)
plt.tight_layout()
fig.savefig("C1_English_Urdu_NMT/figures/loss_curves.png", dpi=300)
plt.close(fig)

# ==============================================================================
# 4. DECODING COMPARISON: GREEDY VS BEAM SEARCH (K=4)
# ==============================================================================
def translate_greedy(sentence, max_len=30):
    model.eval()
    tokens = src_tokenizer.encode(sentence)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    
    input_tok = torch.tensor([SOS_IDX]).to(device)
    decoded_indices = []
    attentions = []
    
    for _ in range(max_len):
        with torch.no_grad():
            prediction, hidden, attention = model.decoder(input_tok, hidden, encoder_outputs)
        top1 = prediction.argmax(1).item()
        if top1 == EOS_IDX:
            break
        decoded_indices.append(top1)
        attentions.append(attention.squeeze(0).cpu().numpy())
        input_tok = torch.tensor([top1]).to(device)
        
    return tgt_tokenizer.decode(decoded_indices), attentions

def translate_beam(sentence, beam_width=4, max_len=30):
    model.eval()
    tokens = src_tokenizer.encode(sentence)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        
    # (score, [token_indices], hidden_state)
    hypotheses = [(0.0, [SOS_IDX], hidden)]
    completed = []
    
    for _ in range(max_len):
        new_hypotheses = []
        for score, seq, h in hypotheses:
            if seq[-1] == EOS_IDX:
                completed.append((score, seq))
                continue
            input_tok = torch.tensor([seq[-1]]).to(device)
            with torch.no_grad():
                pred, next_h, _ = model.decoder(input_tok, h, encoder_outputs)
            log_probs = torch.log_softmax(pred, dim=1).squeeze(0)
            topk_probs, topk_indices = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                new_score = score + topk_probs[k].item()
                new_seq = seq + [topk_indices[k].item()]
                new_hypotheses.append((new_score, new_seq, next_h))
                
        hypotheses = sorted(new_hypotheses, key=lambda x: x[0] / len(x[1]), reverse=True)[:beam_width]
        if len(completed) >= beam_width:
            break
            
    best_seq = (completed + hypotheses)[0][1]
    return tgt_tokenizer.decode(best_seq)

# ==============================================================================
# 5. PRETRAINED BENCHMARK: HELSINKI-NLP/OPUS-MT-EN-UR (MARIANMT)
# ==============================================================================
print("\n" + "=" * 60, flush=True)
print("5. EVALUATING PRETRAINED MARIANMT (MODEL 2)", flush=True)
print("=" * 60, flush=True)

marian_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
marian_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ur").to(device)
marian_model.eval()

def translate_marian(sentences, batch_size=32):
    translated = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = marian_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = marian_model.generate(**inputs, num_beams=4, max_length=128)
        decoded = marian_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(decoded)
    return translated

# Evaluate on 2,000 Test Pairs
EVAL_TEST_COUNT = 2000
test_subset = test_pairs[:EVAL_TEST_COUNT]
test_en = [p[0] for p in test_subset]
test_ur_refs = [[p[1] for p in test_subset]] # Sacrebleu expects list of reference lists

print(f"Translating {len(test_en)} test sentences with Model 1 (Greedy)...", flush=True)
model1_greedy_preds = [translate_greedy(s)[0] for s in test_en]

print(f"Translating {len(test_en)} test sentences with Model 1 (Beam Search k=4)...", flush=True)
model1_beam_preds = [translate_beam(s, beam_width=4) for s in test_en]

print(f"Translating {len(test_en)} test sentences with Model 2 (MarianMT)...", flush=True)
model2_marian_preds = translate_marian(test_en, batch_size=64)

# Compute BLEU and chrF
bleu_m1_greedy = sacrebleu.corpus_bleu(model1_greedy_preds, test_ur_refs)
chrf_m1_greedy = sacrebleu.corpus_chrf(model1_greedy_preds, test_ur_refs)

bleu_m1_beam = sacrebleu.corpus_bleu(model1_beam_preds, test_ur_refs)
chrf_m1_beam = sacrebleu.corpus_chrf(model1_beam_preds, test_ur_refs)

bleu_m2 = sacrebleu.corpus_bleu(model2_marian_preds, test_ur_refs)
chrf_m2 = sacrebleu.corpus_chrf(model2_marian_preds, test_ur_refs)

print("\n" + "=" * 60, flush=True)
print("TEST EVALUATION RESULTS (2,000 Test Pairs):", flush=True)
print(f"Model 1 (Greedy):     BLEU = {bleu_m1_greedy.score:5.2f} | chrF = {chrf_m1_greedy.score:5.2f}", flush=True)
print(f"Model 1 (Beam k=4):   BLEU = {bleu_m1_beam.score:5.2f} | chrF = {chrf_m1_beam.score:5.2f}", flush=True)
print(f"Model 2 (MarianMT):   BLEU = {bleu_m2.score:5.2f} | chrF = {chrf_m2.score:5.2f}", flush=True)
print("=" * 60, flush=True)

# BLEU by source-length bucket (1-10, 11-20, 21-30 tokens)
length_buckets = {'1-10': ([], []), '11-20': ([], []), '21-30': ([], [])}
for i, s in enumerate(test_en):
    l = len(s.split())
    ref = test_ur_refs[0][i]
    pred = model1_beam_preds[i]
    if l <= 10:
        length_buckets['1-10'][0].append(pred)
        length_buckets['1-10'][1].append(ref)
    elif l <= 20:
        length_buckets['11-20'][0].append(pred)
        length_buckets['11-20'][1].append(ref)
    else:
        length_buckets['21-30'][0].append(pred)
        length_buckets['21-30'][1].append(ref)

bucket_bleu = {}
for b_name, (preds, refs) in length_buckets.items():
    if len(preds) > 0:
        bucket_bleu[b_name] = round(sacrebleu.corpus_bleu(preds, [refs]).score, 2)
print("BLEU by sentence length bucket:", bucket_bleu, flush=True)

# Figure 2: BLEU by source length plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=list(bucket_bleu.keys()), y=list(bucket_bleu.values()), palette='Purples_r', ax=ax, edgecolor='black')
ax.set_title("BLEU Score by Source Sentence Length (Tokens)", fontsize=11, fontweight='bold')
ax.set_ylabel("sacreBLEU Score")
ax.set_xlabel("Source Sentence Token Length")
for i, (k, v) in enumerate(bucket_bleu.items()):
    ax.text(i, v + 0.4, f"{v:.1f}", ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
fig.savefig("C1_English_Urdu_NMT/figures/bleu_by_length.png", dpi=300)
plt.close(fig)

# Figure 3: Attention Heatmap for Sample Sentence
sample_en = "the student reads the book"
sample_pred, sample_attns = translate_greedy(sample_en)
if len(sample_attns) > 0:
    attn_matrix = np.array(sample_attns)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    src_words = sample_en.split() + ['<eos>']
    pred_words = sample_pred.split()
    sns.heatmap(attn_matrix[:len(pred_words), :len(src_words)], annot=True, fmt=".2f", cmap='Purples',
                xticklabels=src_words, yticklabels=pred_words, ax=ax)
    ax.set_title("Bahdanau Attention Alignment (English to Urdu)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Source English Words")
    ax.set_ylabel("Generated Urdu Words")
    plt.tight_layout()
    fig.savefig("C1_English_Urdu_NMT/figures/attention_heatmap.png", dpi=300)
    plt.close(fig)

# Save 10 Translation Examples
examples = []
for i in range(10):
    examples.append({
        'Source (English)': test_en[i],
        'Reference (Urdu)': test_ur_refs[0][i],
        'Model 1 (Greedy)': model1_greedy_preds[i],
        'Model 1 (Beam k=4)': model1_beam_preds[i],
        'Model 2 (MarianMT)': model2_marian_preds[i]
    })

results_summary = {
    'metrics': {
        'Model 1 (Greedy)': {'BLEU': round(bleu_m1_greedy.score, 2), 'chrF': round(chrf_m1_greedy.score, 2)},
        'Model 1 (Beam k=4)': {'BLEU': round(bleu_m1_beam.score, 2), 'chrF': round(chrf_m1_beam.score, 2)},
        'Model 2 (MarianMT)': {'BLEU': round(bleu_m2.score, 2), 'chrF': round(chrf_m2.score, 2)}
    },
    'length_buckets': bucket_bleu,
    'examples': examples
}

with open("C1_English_Urdu_NMT/results_summary.json", "w", encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print("\nTask C1 execution completed successfully! Summary saved to C1_English_Urdu_NMT/results_summary.json", flush=True)
