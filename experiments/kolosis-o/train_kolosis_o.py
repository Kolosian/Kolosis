import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neural_networks.kolosis.kolosis_o import KolosisO

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

def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        logits, loss, stats = model(x, y, return_stats=True)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            # Monitor gate weights from first block, first token
            # Stats shape: [n_layer, B, T, 3]
            gate_weights = stats['gate_weights'][0, 0, 0].tolist() 
            gate_str = "/".join([f"{w:.2f}" for w in gate_weights])
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gates': gate_str,
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()

def main():
    config = {
        'pretrained_tokenizer_name': "Xenova/gpt-4", # GPT-4 tokenizer (cl100k_base)
        'vocab_size': 100277, # Size for cl100k_base
        'n_embd': 256,
        'n_head': 8,
        'n_kv_head': 2,
        'n_layer': 6,
        'block_size': 128,
        'dropout': 0.1,
        'batch_size': 32,
        'epochs': 5,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'warmup_steps': 100
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    print(f"Loading '{config['pretrained_tokenizer_name']}' tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config['pretrained_tokenizer_name'])
    except Exception as e:
        print(f"Failed to load primary tokenizer: {e}")
        print("Fallback: Using GPT2Tokenizer (Warning: Mismatch in vocab size possible)")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading WikiText-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    train_data = WikiTextDataset(dataset['train']['text'], tokenizer, config['block_size'])
    val_data = WikiTextDataset(dataset['validation']['text'], tokenizer, config['block_size'])
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    print("Creating Kolosis-O...")
    model = KolosisO(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_kv_head=config['n_kv_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['lr'], 
        total_steps=total_steps,
        pct_start=0.1
    )
    
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1)
        val_loss, ppl = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = 'kolosis_o_best.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config
            }, save_path)
            print(f"✅ Saved best model to {save_path}")

if __name__ == "__main__":
    main()
