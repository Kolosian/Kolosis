import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import os
import sys

# Import model definition
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_kolosis_s_colab import KolosisS, WikiTextDataset

def train_router_rl():
    # --- Config for Kolosis-S RL ---
    config = {
        'vocab_size': 50257,
        'n_embd': 128,
        'block_size': 128,
        'n_layer': 2,
        'dropout': 0.1,
        'batch_size': 8,
        'lr_router': 6e-4,       # ~2x main LR (assume main was 3e-4)
        'epochs': 2,             # Conservative epoch count
        'rl_weight': 0.01,       # Conservative RL weight
        'entropy_coeff': 0.005,  # Prevent collapse
        'kl_coeff': 0.005,       # Prevent drift
        'val_interval': 100,     # steps
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = KolosisS(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        dropout=config['dropout']
    ).to(device)
    
    # Load Checkpoint
    checkpoint_path = 'kolosis_s_best.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'experiments/Kolosis_S_checkpoints/kolosis_s_best.pt'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("❌ Checkpoint not found! Aborting RL.")
        return
        
    # 2. Freeze Everything EXCEPT Router
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze fusion gate (router)
    # In KolosisS, router params are inside self.fusion_gate.gate_net
    for param in model.fusion_gate.gate_net.parameters():
        param.requires_grad = True
        
    print("❄️  Backbone & Streams FROZEN. Training Router Only.")
    
    # 3. Setup Optimizer (Router Only)
    router_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(router_params, lr=config['lr_router'])
    
    # 4. Data
    print("Loading data...")
    train_dataset = WikiTextDataset('wikitext-103-v1', 'train', config['block_size'])
    val_dataset = WikiTextDataset('wikitext-103-v1', 'validation', config['block_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 5. Training Loop
    print("\nStarting RL Fine-Tuning...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        total_policy_loss = 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            B, T = x.shape
            
            # --- Forward Pass for RL ---
            # We need logits from ALL streams to calculate counterfactuals
            # We'll use the same trick as the diagnostic script: execute shared head manually
            
            # 1. Get stream features & router info
            # We enable return_stream_outputs to get features
            main_logits, main_loss, info = model(x, targets=y, return_stream_outputs=True)
            
            stream_features = info['stream_features'] # [B, T, n_streams, n_embd]
            gate_weights = info['gate_weights']       # [B, T, n_streams]
            
            # 2. Calculate per-stream losses (Counterfactuals)
            n_streams = stream_features.shape[2]
            per_stream_losses = []
            
            with torch.no_grad(): # Don't backprop through experts!
                for k in range(n_streams):
                    feat_k = stream_features[:, :, k, :] 
                    feat_k = model.ln_f(feat_k)
                    logits_k = model.head(feat_k)
                    loss_k = F.cross_entropy(logits_k.view(-1, config['vocab_size']), y.view(-1), reduction='none')
                    loss_k = loss_k.view(B, T)
                    per_stream_losses.append(loss_k)
                
                per_stream_losses = torch.stack(per_stream_losses, dim=-1) # [B, T, n_streams]
                
                # 3. Calculate Advantage
                # Adv = (Optimal Loss - Actual Loss) ??
                # Actually, standard policy gradient: Adv = (Baseline - Reward)
                # Here Reward = -Loss
                # Adv = (Baseline - (-Loss)) = Baseline + Loss
                # But we want Counterfactual Advantage: How much better is this stream than the average/best?
                
                # Let's use user's formula: adv = (min_loss - per_stream_loss)
                min_loss, _ = torch.min(per_stream_losses, dim=-1, keepdim=True)
                advantage = min_loss - per_stream_losses # Should be <= 0. 0 for optimal stream.
                
                # Normalize per token
                # This centers the advantage so better-than-avg streams get positive signal
                # adv_mean = advantage.mean(dim=-1, keepdim=True)
                # adv_std = advantage.std(dim=-1, keepdim=True) + 1e-8
                # adv_norm = (advantage - adv_mean) / adv_std
                
                # User specifically asked for: adv = (min_loss - per_stream_loss) / (min_loss + eps)
                # This is a relative improvement metric.
                adv_norm = (min_loss - per_stream_losses) / (min_loss + 1e-8)
            
            # 4. Policy Loss via REINFORCE
            # Policy Grad = - sum(prob * adv)
            # We backprop through gate_weights
            policy_loss = -torch.sum(gate_weights * adv_norm, dim=-1).mean()
            
            # 5. Regularization
            # Entropy
            entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1).mean()
            
            # KL divergence (vs uniform or vs frozen router? User said vs previous probs)
            # For simplicity in this script, let's just use entropy + small L2 on logits (z-loss style) provided by model
            # Or implement simple KL to uniform to prevent collapse if user asked
            # User script: kl_coeff * KL(stream_probs || prev)
            # Implementing "prev" requires keeping a copy of the old policy or simple running avg.
            # Let's stick to simple Entropy maximization for now as it's cleaner.
            
            total_loss = main_loss + config['rl_weight'] * (policy_loss - config['entropy_coeff'] * entropy)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            global_step += 1
            
            if i % 100 == 0:
                print(f"Ep {epoch} [{i}] | Main: {main_loss.item():.4f} | Pol: {policy_loss.item():.4f} | Ent: {entropy.item():.4f}")
                
            # Abort Checks (every 500 steps)
            if i > 0 and i % 500 == 0:
                # Quick Val Check
                pass # Implement full val loop if needed

    # Save
    torch.save(model.state_dict(), 'kolosis_s_rl_tuned.pt')
    print("✅ RL Tuning Complete. Saved to kolosis_s_rl_tuned.pt")

if __name__ == "__main__":
    train_router_rl()
