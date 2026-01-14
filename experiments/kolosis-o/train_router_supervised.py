import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import json
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neural_networks.kolosis.kolosis_o import KolosisO

class GoldenRoutingDataset(Dataset):
    """
    Loads processed_golden_data.jsonl for supervised router training.
    """
    def __init__(self, file_path, tokenizer, block_size=128):
        self.examples = []
        # Mapping from Agent names to RLGate indices:
        # [Semantic, Temporal, Conceptual, Causal]
        self.key_map = {
            "Semantic Agent": 0,
            "Temporal Agent": 1,
            "Conceptual Agent": 2,
            "Causal Agent": 3
        }
        
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                weights = data.get("routing_weights")
                if not weights:
                    continue
                
                # Convert weights dict to tensor
                target_weights = torch.zeros(4)
                for key, idx in self.key_map.items():
                    target_weights[idx] = weights.get(key, 0.25)
                
                # Tokenize prompt
                tokens = tokenizer.encode(data["prompt"], add_special_tokens=False, max_length=block_size, truncation=True)
                if len(tokens) < 2: continue
                
                self.examples.append({
                    "ids": torch.tensor(tokens, dtype=torch.long),
                    "targets": target_weights
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def train_router_imitation():
    # 1. Config
    config = {
        "n_embd": 256,
        "n_head": 8,
        "n_kv_head": 2,
        "n_layer": 6,
        "block_size": 128,
        "lr": 1e-3, # Higher LR since we only train the gate
        "epochs": 20,
        "batch_size": 16,
        "data_path": "processed_golden_data.jsonl"
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 2. Data
    if not os.path.exists(config["data_path"]):
        print(f"❌ Error: {config['data_path']} not found. Run Kolosis-A collector/processor first.")
        return

    dataset = GoldenRoutingDataset(config["data_path"], tokenizer, config["block_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # 3. Model
    model = KolosisO(
        vocab_size=50257,
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_kv_head=config["n_kv_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"]
    ).to(device)
    
    # ❄️ FREEZE BACKBONE: Only train RLGate in each block
    for name, param in model.named_parameters():
        if "rl_gate" not in name:
            param.requires_grad = False
        else:
            print(f"🔥 Training parameter: {name}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    
    # 4. Training Loop
    print("\n🚀 Starting Supervised Router Imitation (Phase 1)...")
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            ids = batch["ids"].to(device)
            target_weights = batch["targets"].to(device) # (B, 4)
            
            optimizer.zero_grad()
            
            # Forward pass
            # logits, loss, stats (where stats['gate_weights'] is [n_layer, B, T, 4])
            _, _, stats = model(ids, return_stats=True)
            
            # We want the LAST token's routing weights to match the Golden Data
            # (Assuming the prompt-level reasoning matches the final state)
            # shape: [n_layer, B, 4]
            pred_weights = stats["gate_weights"][:, :, -1, :] 
            
            # Loss: KL Divergence between predicted weights and agentic weights
            # Avg across all layers
            loss = 0
            for layer_idx in range(pred_weights.shape[0]):
                # F.kl_div expects log_probabilities
                log_pred = torch.log(pred_weights[layer_idx] + 1e-10)
                loss += F.kl_div(log_pred, target_weights, reduction='batchmean')
            
            loss = loss / pred_weights.shape[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")

    # 5. Save ONLY the router weights (or full model with frozen backbone)
    save_path = "kolosis_o_imitator.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "mode": "supervised_imitation"
    }, save_path)
    print(f"\n✅ Imitation Phase Complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train_router_imitation()
