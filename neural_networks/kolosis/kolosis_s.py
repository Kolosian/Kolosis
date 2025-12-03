"""
Kolosis-S (Streamlined): Optimized Multi-Stream Architecture
Features:
- Shared Embeddings (reduces params by 75%)
- Shared Backbone (learns general features)
- Specialized Adapters (Temporal, Semantic, Concept)
- Shared Prediction Head (forces latent alignment)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_attention import MultiScaleTemporalAttention

class TemporalAdapter(nn.Module):
    """Adds temporal awareness to features"""
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
        # Attention block
        x = x + self.attn(self.norm1(x))
        # FFN block
        x = x + self.ffn(self.norm2(x))
        return x

class SemanticAdapter(nn.Module):
    """Adds relationship awareness to features"""
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
        
        # Pairwise relationship encoding
        if T > 1:
            pairs = torch.cat([x[:, :-1], x[:, 1:]], dim=-1)
            relation_features = self.relation_encoder(pairs)
            
            # CRITICAL FIX: Pad at START to ensure causality
            # (padding_left, padding_right, padding_top, padding_bottom)
            relation_features = F.pad(relation_features, (0, 0, 1, 0))
            
            # Add to input
            x = x + relation_features
            
        # FFN block
        x = x + self.ffn(self.norm(x))
        return x

class ConceptAdapter(nn.Module):
    """Adds hierarchical abstraction to features"""
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
        # Create hierarchical features
        concept = self.concept_emb(idx)
        law = self.law_emb(idx)
        
        # Add to input (weighted)
        x = x + self.alpha * concept + self.beta * law
        
        # FFN block
        x = x + self.ffn(self.norm(x))
        return x

class FusionGate(nn.Module):
    """Learn to combine outputs from multiple streams"""
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
        # Concatenate all stream features
        combined = torch.cat(stream_features, dim=-1)
        
        # Compute gate weights
        gate_weights = self.gate_network(combined)
        
        # Weighted combination
        fused = torch.zeros_like(stream_features[0])
        for i, features in enumerate(stream_features):
            fused += gate_weights[:, :, i:i+1] * features
            
        return fused, gate_weights
    
    def compute_entropy_loss(self, gate_weights):
        avg_weights = gate_weights.mean(dim=[0, 1])
        entropy = -(avg_weights * torch.log(avg_weights + 1e-8)).sum()
        return -self.entropy_weight * entropy

class KolosisS(nn.Module):
    """
    Kolosis-S: Streamlined Multi-Stream Architecture
    """
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        
        # 1. SHARED EMBEDDINGS (Huge parameter saving)
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # 2. SHARED BACKBONE (Learns general features)
        self.backbone = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=4,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layer)
        ])
        
        # 3. SPECIALIZED ADAPTERS
        self.temporal_adapter = TemporalAdapter(n_embd, block_size, dropout)
        self.semantic_adapter = SemanticAdapter(n_embd, dropout)
        self.concept_adapter = ConceptAdapter(vocab_size, n_embd, dropout)
        
        # 4. FUSION
        self.fusion_gate = FusionGate(n_streams=4, n_embd=n_embd)
        
        # 5. SHARED PREDICTION HEAD (Forces alignment)
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
        
        # 1. Shared Embeddings
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        # 2. Shared Backbone
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        for layer in self.backbone:
            x = layer(x, src_mask=mask)
            
        # 3. Split into Streams (Adapters)
        # Stream 1: Symbol (Identity - just the backbone features)
        symbol_feat = x
        
        # Stream 2: Temporal (Adds temporal bias)
        temporal_feat = self.temporal_adapter(x)
        
        # Stream 3: Semantic (Adds relation features)
        semantic_feat = self.semantic_adapter(x)
        
        # Stream 4: Concept (Adds hierarchical embeddings)
        concept_feat = self.concept_adapter(x, idx)
        
        # 4. Fusion
        fused_feat, gate_weights = self.fusion_gate([
            symbol_feat, temporal_feat, semantic_feat, concept_feat
        ])
        
        # 5. Prediction (Shared Head)
        fusion_logits = self.head(fused_feat)
        
        # Compute loss
        loss = None
        stream_losses = None
        if targets is not None:
            # Auxiliary losses for each stream using SHARED HEAD
            symbol_logits = self.head(symbol_feat)
            temporal_logits = self.head(temporal_feat)
            semantic_logits = self.head(semantic_feat)
            concept_logits = self.head(concept_feat)
            
            stream_logits_list = [symbol_logits, temporal_logits, semantic_logits, concept_logits]
            
            stream_losses = []
            for logits in stream_logits_list:
                stream_loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
                stream_losses.append(stream_loss)
                
            fusion_loss = F.cross_entropy(fusion_logits.view(B*T, -1), targets.view(B*T))
            entropy_loss = self.fusion_gate.compute_entropy_loss(gate_weights)
            
            # Combined loss
            total_loss = (
                0.5 * fusion_loss +
                0.5 * sum(stream_losses) / len(stream_losses) +
                entropy_loss
            )
            loss = total_loss
            
        if return_stream_outputs:
            # Generate logits for return if needed
            symbol_logits = self.head(symbol_feat)
            temporal_logits = self.head(temporal_feat)
            semantic_logits = self.head(semantic_feat)
            concept_logits = self.head(concept_feat)
            
            stream_info = {
                'symbol_logits': symbol_logits,
                'temporal_logits': temporal_logits,
                'semantic_logits': semantic_logits,
                'concept_logits': concept_logits,
                'gate_weights': gate_weights,
                'stream_losses': stream_losses
            }
            return fusion_logits, loss, stream_info
            
        return fusion_logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    print("Testing Kolosis-S (Streamlined)")
    
    config = {
        'vocab_size': 100,
        'n_embd': 64,
        'block_size': 32,
        'n_layer': 2,
        'dropout': 0.1
    }
    
    model = KolosisS(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    
    logits, loss, info = model(x, y, return_stream_outputs=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gate weights: {info['gate_weights'].mean(dim=[0,1])}")
