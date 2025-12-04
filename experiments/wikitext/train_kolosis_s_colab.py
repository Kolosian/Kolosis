"""
Kolosis-S Training on WikiText-103 (Colab-Ready) - CORRECTED VERSION
Streamlined multi-stream architecture with shared backbone.

MATCHES LOCAL IMPLEMENTATION EXACTLY:
- 4 streams (Symbol, Temporal, Semantic, Concept)
- Hierarchical embeddings in ConceptAdapter
- Correct causal padding in SemanticAdapter
- Full FusionGate with entropy loss

Usage in Colab:
1. Upload this file to Colab
2. Run: !python train_kolosis_s_colab.py
3. Training will save results automatically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
from tqdm import tqdm

# ============================================================================
# KOLOSIS-S MODEL DEFINITION (EXACT MATCH TO LOCAL)
# ============================================================================

class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, n_embd, n_out, block_size, dropout=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        
        scale = 1.0 / (C ** 0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        return self.proj(out)

class TemporalAdapter(nn.Module):
    def __init__(self, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.attn = MultiScaleTemporalAttention(n_embd, n_embd, block_size, dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SemanticAdapter(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.relation_encoder = nn.Linear(n_embd * 2, n_embd)
        self.norm = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        B, T, C = x.shape
        
        if T > 1:
            pairs = torch.cat([x[:, :-1], x[:, 1:]], dim=-1)
            relation_features = self.relation_encoder(pairs)
            
            # CRITICAL FIX: Pad at START to ensure causality
            relation_features = F.pad(relation_features, (0, 0, 1, 0))
            x = x + relation_features
            
        x = x + self.ffn(self.norm(x))
        return x

class ConceptAdapter(nn.Module):
    def __init__(self, vocab_size, n_embd, dropout=0.1):
        super().__init__()
        # Hierarchical embeddings
        self.concept_emb = nn.Embedding(vocab_size, n_embd)
        self.law_emb = nn.Embedding(vocab_size, n_embd)
        
        # Learnable weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.3))
        
        self.norm = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, idx):
        concept = self.concept_emb(idx)
        law = self.law_emb(idx)
        x = x + self.alpha * concept + self.beta * law
        x = x + self.ffn(self.norm(x))
        return x

class FusionGate(nn.Module):
    def __init__(self, n_streams, n_embd):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(n_streams * n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_streams),
            nn.Softmax(dim=-1)
        )
        self.entropy_weight = 0.01
        
    def forward(self, stream_features):
        combined = torch.cat(stream_features, dim=-1)
        gate_weights = self.gate_network(combined)
        
        fused = torch.zeros_like(stream_features[0])
        for i, features in enumerate(stream_features):
            fused += gate_weights[:, :, i:i+1] * features
            
        return fused, gate_weights
    
    def compute_entropy_loss(self, gate_weights):
        avg_weights = gate_weights.mean(dim=[0, 1])
        entropy = -(avg_weights * torch.log(avg_weights + 1e-8)).sum()
        return -self.entropy_weight * entropy

class KolosisS(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        self.backbone = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=4, dim_feedforward=4*n_embd,
                dropout=dropout, batch_first=True
            ) for _ in range(n_layer)
        ])
        
        self.temporal_adapter = TemporalAdapter(n_embd, block_size, dropout)
        self.semantic_adapter = SemanticAdapter(n_embd, dropout)
        self.concept_adapter = ConceptAdapter(vocab_size, n_embd, dropout)
        
        self.fusion_gate = FusionGate(n_streams=4, n_embd=n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None, return_stream_outputs=False):
        B, T = idx.shape
        
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        for layer in self.backbone:
            x = layer(x, src_mask=mask)
            
        # 4 STREAMS (Symbol, Temporal, Semantic, Concept)
        symbol_feat = x
        temporal_feat = self.temporal_adapter(x)
        semantic_feat = self.semantic_adapter(x)
        concept_feat = self.concept_adapter(x, idx)
        
        fused, gate_weights = self.fusion_gate([symbol_feat, temporal_feat, semantic_feat, concept_feat])
        logits = self.head(fused)
        
        loss = None
        entropy_loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            entropy_loss = self.fusion_gate.compute_entropy_loss(gate_weights)
            loss = loss + entropy_loss
            
        return logits, loss

# ============================================================================
# DATASET
# ============================================================================

class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.examples = []
        for text in tqdm(texts, desc="Tokenizing"):
            if len(text.strip()) == 0:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False, max_length=2048, truncation=True)
            for i in range(0, len(tokens) - block_size, block_size):
                chunk = tokens[i:i + block_size + 1]
                if len(chunk) == block_size + 1:
                    self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)  # loss now includes entropy regularization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    return total_loss / len(loader), torch.exp(torch.tensor(total_loss / len(loader))).item()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("WIKITEXT-103: KOLOSIS-S (STREAMLINED)")
    print("="*60)
    
    config = {
        'vocab_size': 50257,
        'n_embd': 128,
        'block_size': 128,
        'n_layer': 2,
        'dropout': 0.1,
        'batch_size': 32,
        'epochs': 10,
        'lr': 0.0003
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nLoading WikiText-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    train_data = WikiTextDataset(dataset['train']['text'], tokenizer, config['block_size'])
    val_data = WikiTextDataset(dataset['validation']['text'], tokenizer, config['block_size'])
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    print(f"\nCreating Kolosis-S...")
    model = KolosisS(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    results = {
        'config': config,
        'train_losses': [],
        'val_losses': [],
        'perplexities': []
    }
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch+1)
        val_loss, perplexity = evaluate(model, val_loader, device)
        
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['perplexities'].append(perplexity)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PPL: {perplexity:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'kolosis_s_best.pt')
            print("✅ Saved best model")
    
    with open('kolosis_s_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best perplexity: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")
    
    return results

if __name__ == "__main__":
    results = main()
