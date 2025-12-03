"""
Kolosis V2 Minimal + Temporal (Single Head)
Combines hierarchical embeddings, semantic stream, and temporal attention.
Single prediction head for fair comparison with baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_attention import MultiScaleTemporalAttention

class KolosisV2MinimalTemporalSingleHead(nn.Module):
    """
    Kolosis V2 with:
    - Hierarchical embeddings
    - Concept stream
    - Semantic stream  
    - Temporal stream (multi-scale attention)
    - Single ensemble prediction head (fair comparison)
    """
    
    def __init__(self, vocab_size, n_embd, block_size, n_layer=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Hierarchical embeddings
        self.symbol_emb = nn.Embedding(vocab_size, n_embd)
        self.concept_emb = nn.Embedding(vocab_size, n_embd)
        self.law_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Learnable hierarchy weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.3))
        
        # Concept stream
        self.concept_stream = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=4,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layer)
        ])
        
        # Semantic stream
        self.semantic_stream = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=4,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layer)
        ])
        
        # Temporal stream (multi-scale attention)
        self.temporal_layers = nn.ModuleList([
            MultiScaleTemporalAttention(
                head_size=n_embd,
                n_embd=n_embd,
                block_size=block_size,
                dropout=dropout
            )
            for _ in range(n_layer)
        ])
        self.temporal_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4*n_embd),
                nn.GELU(),
                nn.Linear(4*n_embd, n_embd),
                nn.Dropout(dropout)
            )
            for _ in range(n_layer)
        ])
        
        # Relationship encoder
        self.relation_encoder = nn.Linear(n_embd * 2, n_embd)
        
        # Layer norms
        self.concept_norm = nn.LayerNorm(n_embd)
        self.semantic_norm = nn.LayerNorm(n_embd)
        self.temporal_norm = nn.LayerNorm(n_embd)
        
        # Three-way fusion (learned weights)
        self.fusion_logits = nn.Parameter(torch.zeros(3))  # concept, semantic, temporal
        
        # Single ensemble head
        self.ensemble_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_hierarchical_embedding(self, idx):
        """Create hierarchical embeddings"""
        B, T = idx.shape
        
        symbol = self.symbol_emb(idx)
        concept = self.concept_emb(idx)
        law = self.law_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        
        return symbol + self.alpha * concept + self.beta * law + pos
    
    def create_semantic_embedding(self, idx):
        """Create relationship-aware embeddings"""
        B, T = idx.shape
        
        base_emb = self.create_hierarchical_embedding(idx)
        
        if T > 1:
            pairs = torch.cat([base_emb[:, :-1], base_emb[:, 1:]], dim=-1)
            relation_features = self.relation_encoder(pairs)
            # Pad at START to ensure causality
            relation_features = F.pad(relation_features, (0, 0, 1, 0))
            return base_emb + relation_features
        else:
            return base_emb
    
    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices (optional)
            
        Returns:
            logits: (B, T, vocab_size) predictions
            loss: Scalar loss (if targets provided)
        """
        B, T = idx.shape
        
        # Create embeddings
        concept_emb = self.create_hierarchical_embedding(idx)
        semantic_emb = self.create_semantic_embedding(idx)
        temporal_emb = self.create_hierarchical_embedding(idx)
        
        # Create causal mask for transformer layers
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        # Process concept stream
        concept_feat = concept_emb
        for layer in self.concept_stream:
            concept_feat = layer(concept_feat, src_mask=mask)
        concept_feat = self.concept_norm(concept_feat)
        
        # Process semantic stream
        semantic_feat = semantic_emb
        for layer in self.semantic_stream:
            semantic_feat = layer(semantic_feat, src_mask=mask)
        semantic_feat = self.semantic_norm(semantic_feat)
        
        # Process temporal stream
        temporal_feat = temporal_emb
        for attn, ffn in zip(self.temporal_layers, self.temporal_ffn):
            # Multi-scale temporal attention
            temporal_feat = temporal_feat + attn(temporal_feat)
            # Feed-forward
            temporal_feat = temporal_feat + ffn(temporal_feat)
        temporal_feat = self.temporal_norm(temporal_feat)
        
        # Three-way fusion
        fusion_weights = F.softmax(self.fusion_logits, dim=0)
        fused_feat = (
            fusion_weights[0] * concept_feat +
            fusion_weights[1] * semantic_feat +
            fusion_weights[2] * temporal_feat
        )
        
        # Single ensemble prediction
        ensemble_logits = self.ensemble_head(fused_feat)
        
        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                ensemble_logits.view(B*T, self.vocab_size),
                targets.view(B*T)
            )
        
        return ensemble_logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens"""
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
    
    def get_fusion_weights(self):
        """Get current fusion weights"""
        weights = F.softmax(self.fusion_logits, dim=0).detach().cpu()
        return {
            'concept': weights[0].item(),
            'semantic': weights[1].item(),
            'temporal': weights[2].item()
        }
    
    def get_temporal_stats(self):
        """Get temporal attention statistics"""
        stats = []
        for layer in self.temporal_layers:
            gamma_fast = torch.sigmoid(layer.gamma_fast_logit).item()
            gamma_medium = torch.sigmoid(layer.gamma_medium_logit).item()
            gamma_slow = torch.sigmoid(layer.gamma_slow_logit).item()
            
            alphas = F.softmax(layer.alpha_logits, dim=0).detach().cpu()
            
            stats.append({
                'gamma_fast': gamma_fast,
                'gamma_medium': gamma_medium,
                'gamma_slow': gamma_slow,
                'alpha_fast': alphas[0].item(),
                'alpha_medium': alphas[1].item(),
                'alpha_slow': alphas[2].item()
            })
        return stats
