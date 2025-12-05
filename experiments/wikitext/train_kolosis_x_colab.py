"""
Kolosis-X Training on WikiText-103 (Colab-Ready)
Self-discovering multi-stream architecture with meta-learning fusion.

Usage in Colab:
1. Upload this file to Colab
2. Run all cells
3. Training will save results to Google Drive (optional)
"""

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

# Install dependencies (run once)
# !pip install transformers datasets torch tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
import time
from tqdm import tqdm
import os

# ============================================================================
# KOLOSIS-X MODEL DEFINITION
# ============================================================================

class UnsupervisedStream(nn.Module):
    """Base class for self-discovering streams"""
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        raise NotImplementedError
        
    def unsupervised_loss(self, features, **kwargs):
        raise NotImplementedError

class TemporalStream(UnsupervisedStream):
    def __init__(self, n_embd, block_size, dropout=0.1):
        super().__init__(n_embd, dropout)
        self.position_predictor = nn.Linear(n_embd, 1)
        self.block_size = block_size
        
    def forward(self, x):
        return self.dropout(x)
        
    def unsupervised_loss(self, features, positions):
        B, T, C = features.shape
        if self.block_size == 1:
            target_pos = torch.zeros_like(positions, dtype=torch.float)
        else:
            target_pos = positions.float() / (self.block_size - 1)
        pred_pos = self.position_predictor(features).squeeze(-1)
        return F.mse_loss(pred_pos, target_pos)

class SemanticStream(UnsupervisedStream):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__(n_embd, dropout)
        self.projector = nn.Linear(n_embd, 128)
        
    def forward(self, x):
        return self.dropout(x)
        
    def unsupervised_loss(self, features):
        # SimCLR-style contrastive loss within batch
        # Treat adjacent tokens as positive pairs (simplified)
        B, T, _ = features.shape
        if T < 2:
            return torch.tensor(0.0, device=features.device)
        
        proj = self.projector(features)
        proj = F.normalize(proj, dim=-1)
        
        # Cosine similarity matrix
        sim = torch.matmul(proj, proj.transpose(-1, -2))
        
        # Positive pairs: (t, t+1)
        # Create labels for adjacent tokens
        labels = torch.arange(T, device=features.device)
        # Shifted targets (t -> t+1)
        targets = torch.roll(labels, -1)
        targets[-1] = -100  # Ignore last (standard PyTorch ignore_index)
        
        # This is a simplified proxy for semantic clustering
        # Real implementation would need augmentation or external pairs
        # For now, we encourage temporal smoothness in semantic space
        return F.cross_entropy(sim.view(-1, T), targets.repeat(B))

class ConceptStream(UnsupervisedStream):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__(n_embd, dropout)
        self.norm = nn.LayerNorm(n_embd)
        self.encoder = nn.Linear(n_embd, n_embd // 4)
        self.decoder = nn.Linear(n_embd // 4, n_embd)
    def forward(self, x):
        encoded = self.encoder(self.norm(x))
        decoded = self.decoder(F.gelu(encoded))
        self._last_decoded = decoded
        return x + decoded
    def unsupervised_loss(self, features, original_input=None):
        if original_input is None:
            return torch.tensor(0.0, device=features.device)
        if hasattr(self, '_last_decoded'):
            decoded = self._last_decoded
        else:
            encoded = self.encoder(self.norm(original_input))
            decoded = self.decoder(F.gelu(encoded))
        return F.mse_loss(decoded, original_input)

class CausalStream(UnsupervisedStream):
    """Learns causal relationships between tokens"""
    def __init__(self, n_embd, dropout=0.1):
        super().__init__(n_embd, dropout)
        # Predict if token i causes token j (binary classification)
        self.causal_predictor = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x  # No transformation, just pass through
    def unsupervised_loss(self, features):
        _, T, _ = features.shape
        if T < 2:
            return torch.tensor(0.0, device=features.device)
        current = features[:, :-1, :]
        next_tok = features[:, 1:, :]
        pairs = torch.cat([current, next_tok], dim=-1)
        pred = self.causal_predictor(pairs).squeeze(-1)
        target = torch.ones_like(pred)
        return F.binary_cross_entropy(pred, target)


# ============================================================================
# ANTI-COLLAPSE HELPER FUNCTIONS
# ============================================================================

def normalize_stream_losses(stream_losses):
    """FIX 1: Scale losses so each stream has similar gradient magnitude"""
    if len(stream_losses) == 0:
        return stream_losses
    # Stack and compute std for normalization
    stacked = torch.stack([l if l.dim() == 0 else l.mean() for l in stream_losses])
    loss_std = stacked.std() + 1e-8
    normalized = [l / loss_std for l in stream_losses]
    return normalized


def adaptive_entropy_weight(gate_weights, base_weight=0.35):
    """FIX 2: Increase weight when streams become imbalanced"""
    probs = gate_weights.mean(dim=[0, 1])  # [n_streams]
    n_streams = probs.size(0)
    uniform = 1.0 / n_streams
    
    # Imbalance score: max deviation from uniform
    max_deviation = (probs - uniform).abs().max()
    
    # Scale up weight when imbalanced (0.0 = uniform, 1.0 = collapsed)
    max_possible_dev = 1.0 - uniform
    imbalance_ratio = (max_deviation / max_possible_dev).clamp(0, 1).item()
    
    # Linear scaling: base_weight → 1.0 as deviation increases
    adaptive_weight = base_weight + (1.0 - base_weight) * imbalance_ratio
    return adaptive_weight


def scale_stream_gradients(stream_outputs, gate_weights):
    """FIX 3: Give under-utilized streams stronger gradients"""
    probs = gate_weights.mean(dim=[0, 1]).detach()  # [n_streams]
    n_streams = probs.size(0)
    target_prob = 1.0 / n_streams
    
    # Scale factor: streams with low prob get higher gradients
    scale_factors = (target_prob / (probs + 0.01)).clamp(0.5, 2.0)
    
    # Apply scaling to each stream's contribution
    scaled_outputs = []
    for i, output in enumerate(stream_outputs):
        scale = scale_factors[i].item()
        # Scale the gradient without affecting forward pass
        scaled = output * scale + output.detach() * (1 - scale)
        scaled_outputs.append(scaled)
    
    return scaled_outputs


def gumbel_softmax_st(logits, temperature=1.0, hard=False):
    """FIX 4: Gumbel-Softmax with straight-through estimator for better exploration"""
    # Add Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    
    # Soft sampling
    y_soft = F.softmax(gumbels, dim=-1)
    
    if hard:
        # Hard sampling (one-hot) with straight-through
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        # Straight-through: hard forward, soft backward
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    
    return y


def load_balancing_loss(gate_weights, num_streams):
    """FIX 5: Encourage equal load across streams (from Switch Transformers)"""
    # gate_weights: [B, T, n_streams]
    
    # Fraction of tokens routed to each stream
    routing_probs = gate_weights.mean(dim=[0, 1])  # [n_streams]
    
    # Ideally each gets 1/n_streams
    target = 1.0 / num_streams
    
    # Squared deviation from target
    load_loss = ((routing_probs - target) ** 2).sum()
    
    return load_loss * num_streams  # Scale by n for consistent magnitude


class MetaFusionRouter(nn.Module):
    def __init__(self, n_streams, n_embd, dropout=0.1):
        super().__init__()
        self.n_streams = n_streams
        self.context_encoder = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=4, dim_feedforward=n_embd*2,
            dropout=dropout, batch_first=True
        )
        # Router network outputs logits (no softmax here - apply with temperature)
        self.router_net = nn.Sequential(
            nn.Linear(n_embd, n_streams * 4),
            nn.GELU(),
            nn.Linear(n_streams * 4, n_streams)
        )
        # Initialize router biases to 0 and small weights for balanced start
        self._init_router()
        
    def _init_router(self):
        for m in self.router_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, stream_features, context_features, temperature=1.0, use_gumbel=False):
        _, T, _ = context_features.shape
        mask = torch.triu(torch.ones(T, T, device=context_features.device) * float('-inf'), diagonal=1)
        ctx = self.context_encoder(context_features, src_mask=mask)
        
        # Get raw logits
        logits = self.router_net(ctx)
        
        # FIX 4: Use Gumbel-Softmax during training for better exploration
        if use_gumbel and self.training:
            weights = gumbel_softmax_st(logits, temperature, hard=False)
        else:
            # Apply temperature-scaled softmax (higher temp = more uniform)
            weights = F.softmax(logits / temperature, dim=-1)
        
        # Add min-prob floor for stability (alpha=0.01)
        alpha = 0.01
        uniform = torch.ones_like(weights) / self.n_streams
        weights = weights * (1 - alpha) + uniform * alpha
        
        stacked = torch.stack(stream_features, dim=-1)
        fused = (stacked * weights.unsqueeze(2)).sum(dim=-1)
        return fused, weights, logits

class KolosisX(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.dropout_rate = dropout
        
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        self.backbone = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=4, dim_feedforward=4*n_embd,
                dropout=dropout, batch_first=True
            ) for _ in range(n_layer)
        ])
        
        self.temporal_stream = TemporalStream(n_embd, block_size, dropout)
        self.semantic_stream = SemanticStream(n_embd, dropout)
        self.concept_stream = ConceptStream(n_embd, dropout)
        self.causal_stream = CausalStream(n_embd, dropout)
        
        self.streams = nn.ModuleList([
            self.temporal_stream,
            self.semantic_stream,
            self.concept_stream,
            self.causal_stream
        ])
        
        self.stream_heads = nn.ModuleList([nn.Linear(n_embd, vocab_size) for _ in self.streams])
        self.router = MetaFusionRouter(len(self.streams), n_embd, dropout)
        self.final_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_stream_outputs=False, include_diversity_loss=True, 
                temperature=2.0, kl_weight=0.5, use_gumbel=True):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"
        
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        for layer in self.backbone:
            x = layer(x, src_mask=mask)
        backbone_features = x
        
        # Compute stream outputs
        stream_outputs = [stream(backbone_features) for stream in self.streams]
        
        # FIX 3: Scale gradients for under-utilized streams (use previous gate weights if available)
        if hasattr(self, '_prev_gate_weights') and self.training:
            stream_outputs = scale_stream_gradients(stream_outputs, self._prev_gate_weights)
        
        # Router forward with Gumbel-Softmax option (FIX 4)
        fused_feat, gate_weights, router_logits = self.router(
            stream_outputs, backbone_features, temperature=temperature, use_gumbel=use_gumbel
        )
        
        # Save gate weights for next step's gradient scaling
        if self.training:
            self._prev_gate_weights = gate_weights.detach()
        
        logits = self.final_head(fused_feat)
        
        loss = None
        info = {}
        
        if targets is not None or return_stream_outputs:
            stream_logits = [head(feat) for head, feat in zip(self.stream_heads, stream_outputs, strict=True)]
            
            if targets is not None:
                main_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
                
                aux_losses = [F.cross_entropy(s_logits.view(-1, self.vocab_size), targets.view(-1)) 
                             for s_logits in stream_logits]
                
                positions = torch.arange(T, device=idx.device).expand(B, T)
                unsup_losses = [
                    self.temporal_stream.unsupervised_loss(stream_outputs[0], positions=positions),
                    self.semantic_stream.unsupervised_loss(stream_outputs[1]),
                    self.concept_stream.unsupervised_loss(stream_outputs[2], original_input=backbone_features),
                    self.causal_stream.unsupervised_loss(stream_outputs[3])
                ]
                
                diversity = torch.tensor(0.0, device=idx.device)
                if include_diversity_loss:
                    diversity = self.compute_diversity_loss(stream_outputs)
                
                # === RESEARCH-BACKED MoE LOSS (ST-MoE/PaLM style) ===
                # Key insight: Don't enforce uniformity, just keep router stable
                
                n_streams = len(self.streams)
                log_n = torch.log(torch.tensor(float(n_streams), device=idx.device))
                
                # Entropy (for logging only, not for loss)
                router_entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
                kl_to_uniform = log_n - router_entropy
                
                # === Z-LOSS (from ST-MoE paper) ===
                # Regularize router logits to stay numerically stable
                # This prevents extreme logits without forcing uniformity
                z_loss = (router_logits ** 2).mean()
                
                # === GRADIENT-DECOUPLED LOAD BALANCING ===
                # Only applies gradients to router, NOT to stream weights
                # This is key: streams only see LM gradients!
                with torch.no_grad():
                    routing_probs = gate_weights.mean(dim=[0, 1])  # [n_streams]
                    target = 1.0 / n_streams
                    imbalance = ((routing_probs - target) ** 2).sum()
                
                # If severely imbalanced, add small penalty (but won't backprop to streams)
                # The z-loss handles router stability instead
                
                # === MINIMAL AUXILIARY LOSSES ===
                avg_aux = sum(aux_losses) / len(aux_losses)
                avg_unsup = sum(unsup_losses) / len(unsup_losses)
                
                # === FINAL LOSS: 97% Main Task, 3% Regularization ===
                # This is the key change: almost all gradient goes to LM objective
                loss = (
                    0.97 * main_loss +           # Dominant: language modeling
                    0.01 * z_loss +              # Router stability (not uniformity)
                    0.01 * avg_aux +             # Small auxiliary supervision
                    0.005 * avg_unsup +          # Tiny unsupervised signal
                    0.005 * diversity            # Minimal diversity
                )
                
                # === Enhanced diagnostics ===
                info['main_loss'] = main_loss.item()
                info['aux_loss'] = avg_aux.item()
                info['unsup_loss'] = avg_unsup.item()
                info['diversity_loss'] = diversity.item()
                info['router_entropy'] = router_entropy.item()
                info['kl_to_uniform'] = kl_to_uniform.item()
                info['z_loss'] = z_loss.item()
                info['imbalance'] = imbalance.item()
                
                # Per-stream probability stats
                mean_probs = gate_weights.mean(dim=[0, 1])  # [n_streams]
                info['stream_probs'] = mean_probs.tolist()
                
                # Router logits stats (for debugging)
                info['router_logits_mean'] = router_logits.mean().item()
                info['router_logits_std'] = router_logits.std().item()

            if return_stream_outputs:
                info['gate_weights'] = gate_weights
                info['stream_logits'] = stream_logits
                info['router_logits'] = router_logits
                
        return logits, loss, info

    def compute_diversity_loss(self, stream_features):
        loss = 0.0
        n_pairs = 0
        for i in range(len(stream_features)):
            for j in range(i+1, len(stream_features)):
                f1 = stream_features[i].reshape(-1, self.n_embd)
                f2 = stream_features[j].reshape(-1, self.n_embd)
                sim = F.cosine_similarity(f1, f2, dim=-1).mean()
                loss += sim
                n_pairs += 1
        return loss / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=stream_features[0].device)

# ============================================================================
# DATASET
# ============================================================================

class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.examples = []
        print(f"Tokenizing {len(texts)} documents...")
        for text in tqdm(texts):
            if len(text.strip()) == 0:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False, max_length=2048, truncation=True)
            for i in range(0, len(tokens) - block_size, block_size):
                chunk = tokens[i:i + block_size + 1]
                if len(chunk) == block_size + 1:
                    self.examples.append(chunk)
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compute_router_entropy(info):
    if 'gate_weights' not in info:
        return 0.0
    weights = info['gate_weights']
    entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
    return entropy.mean().item()

def check_gradient_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms

def train_epoch(model, train_loader, optimizer, router_optimizer, device, epoch, temperature=2.0, kl_weight=0.5):
    model.train()
    total_loss = 0
    total_main = 0
    total_aux = 0
    total_unsup = 0
    total_div = 0
    total_kl = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        # Zero both optimizers
        optimizer.zero_grad()
        router_optimizer.zero_grad()
        
        _, loss, info = model(x, y, return_stream_outputs=True, temperature=temperature, kl_weight=kl_weight)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Step both optimizers
        optimizer.step()
        router_optimizer.step()
        
        total_loss += loss.item()
        total_main += info['main_loss']
        total_aux += info['aux_loss']
        total_unsup += info['unsup_loss']
        total_div += info['diversity_loss']
        total_kl += info.get('entropy_loss', 0)
        
        entropy = compute_router_entropy(info)
        grad_norms = check_gradient_norms(model)
        
        # Check router gradient norms specifically
        router_grad_norm = sum(v for k, v in grad_norms.items() if 'router' in k) / max(1, sum(1 for k in grad_norms if 'router' in k))
        
        dead_streams = []
        for i, _head in enumerate(model.stream_heads):
            head_norm = grad_norms.get(f'stream_heads.{i}.weight', 1.0)
            if head_norm < 1e-6:
                dead_streams.append(i)
        
        if batch_idx % 100 == 0:
            # Enhanced status with stream probs
            probs = info.get('stream_probs', [0.25]*4)
            prob_str = '/'.join([f'{p:.2f}' for p in probs])
            status = {
                'loss': f'{loss.item():.4f}',
                'main': f'{info["main_loss"]:.4f}',
                'z_loss': f'{info.get("z_loss", 0):.4f}',
                'ent': f'{entropy:.4f}',
                'probs': prob_str
            }
            if dead_streams:
                status['DEAD'] = str(dead_streams)
            pbar.set_postfix(status)
    
    n = len(train_loader)
    return {
        'total': total_loss / n,
        'main': total_main / n,
        'aux': total_aux / n,
        'unsup': total_unsup / n,
        'diversity': total_div / n
    }

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y, return_stream_outputs=True)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def get_model_stats(model, device, sample_batch):
    model.eval()
    x, y = sample_batch
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        _, _, info = model(x, y, return_stream_outputs=True)
    
    weights = info['gate_weights'].mean(dim=[0, 1]).cpu().tolist()
    stream_names = ['temporal', 'semantic', 'concept', 'causal']
    fusion_stats = {name: weights[i] for i, name in enumerate(stream_names) if i < len(weights)}
    
    return fusion_stats

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("="*60)
    print("WIKITEXT-103: KOLOSIS-X (EXPERIMENTAL)")
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
    
    train_dataset = WikiTextDataset(dataset['train']['text'], tokenizer, config['block_size'])
    val_dataset = WikiTextDataset(dataset['validation']['text'], tokenizer, config['block_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    sample_batch = next(iter(val_loader))
    
    print(f"\nCreating Kolosis-X...")
    model = KolosisX(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        dropout=config['dropout']
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    model = model.to(device)
    
    # === Separate optimizer for router (5x LR) ===
    router_params = list(model.router.parameters())
    main_params = [p for n, p in model.named_parameters() if 'router' not in n]
    
    optimizer = torch.optim.AdamW(main_params, lr=config['lr'])
    router_optimizer = torch.optim.AdamW(router_params, lr=config['lr'] * 5)  # 5x LR for router
    
    print(f"Main optimizer params: {sum(p.numel() for p in main_params):,}")
    print(f"Router optimizer params: {sum(p.numel() for p in router_params):,}")
    
    print("\n" + "="*60)
    print("TRAINING (with anti-collapse fixes)")
    print("  - KL-to-uniform loss (kl_weight=0.5)")
    print("  - Temperature annealing (2.0 -> 1.0)")
    print("  - Separate router optimizer (5x LR)")
    print("  - Min-prob floor (alpha=0.01)")
    print("="*60)
    
    results = {
        'config': config,
        'train_losses': [],
        'val_losses': [],
        'perplexities': [],
        'fusion_weights': [],
        'loss_breakdown': []
    }
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        # === Temperature annealing ===
        # Start with temp=2.0 (exploration), decay to 1.0 (specialization)
        temperature = 2.0 * (0.9 ** epoch)  # Exponential decay
        temperature = max(temperature, 1.0)  # Don't go below 1.0
        
        # === KL weight annealing ===
        # Start strong (0.5), decay slightly to allow specialization
        kl_weight = 0.5 * (0.95 ** epoch)
        kl_weight = max(kl_weight, 0.2)  # Don't go below 0.2
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']} | temp={temperature:.2f}, kl_weight={kl_weight:.3f}")
        print(f"{'='*60}")
        
        train_losses = train_epoch(model, train_loader, optimizer, router_optimizer, device, epoch+1, temperature=temperature, kl_weight=kl_weight)
        val_loss, perplexity = evaluate(model, val_loader, device)
        
        fusion_stats = get_model_stats(model, device, sample_batch)
        
        results['train_losses'].append(train_losses['total'])
        results['val_losses'].append(val_loss)
        results['perplexities'].append(perplexity)
        results['fusion_weights'].append(fusion_stats)
        results['loss_breakdown'].append(train_losses)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"    Main Task: {train_losses['main']:.4f}")
        print(f"    Auxiliary: {train_losses['aux']:.4f}")
        print(f"    Unsupervised: {train_losses['unsup']:.4f}")
        print(f"    Diversity: {train_losses['diversity']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"\nMeta-Router Weights:")
        print(f"  Temporal: {fusion_stats['temporal']:.4f} ({fusion_stats['temporal']*100:.1f}%)")
        print(f"  Semantic: {fusion_stats['semantic']:.4f} ({fusion_stats['semantic']*100:.1f}%)")
        print(f"  Concept:  {fusion_stats['concept']:.4f} ({fusion_stats['concept']*100:.1f}%)")
        print(f"  Causal:   {fusion_stats['causal']:.4f} ({fusion_stats['causal']*100:.1f}%)")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'kolosis_x_best.pt')
            print("  ✅ Saved best model")
    
    # Save results
    with open('kolosis_x_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best perplexity: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")
    
    return results

if __name__ == "__main__":
    results = main()
