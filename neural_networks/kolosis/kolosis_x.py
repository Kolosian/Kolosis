"""
Kolosis-X: Experimental Self-Discovering Multi-Stream Architecture
Features:
- Self-organizing streams with unsupervised pre-training objectives
- Meta-learning fusion router (per-token)
- Diversity regularization
- Modular growth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UnsupervisedStream(nn.Module):
    """Base class for streams that can learn without labels"""
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        return x + self.adapter(self.norm(x))
        
    def unsupervised_loss(self, features, **kwargs):
        raise NotImplementedError

class TemporalStream(UnsupervisedStream):
    """Learns to predict future token positions (rhythm/structure)"""
    def __init__(self, n_embd, block_size, dropout=0.1):
        super().__init__(n_embd, dropout)
        self.position_predictor = nn.Linear(n_embd, 1)
        self.block_size = block_size
        
    def unsupervised_loss(self, features, positions=None):
        if positions is None:
            return torch.tensor(0.0, device=features.device)
            
        # Predict relative position in sequence
        pred_pos = self.position_predictor(features).squeeze(-1)
        
        # Normalize positions to [0, 1]
        target_pos = positions.float() / self.block_size
        
        return F.mse_loss(pred_pos, target_pos)

class SemanticStream(UnsupervisedStream):
    """Learns to cluster semantically similar tokens (contrastive)"""
    def __init__(self, n_embd, dropout=0.1):
        super().__init__(n_embd, dropout)
        self.projector = nn.Linear(n_embd, 128) # Low-dim projection
        
    def unsupervised_loss(self, features, **kwargs):
        # SimCLR-style contrastive loss within batch
        # Treat adjacent tokens as positive pairs (simplified)
        B, T, C = features.shape
        if T < 2: return torch.tensor(0.0, device=features.device)
        
        proj = self.projector(features)
        proj = F.normalize(proj, dim=-1)
        
        # Cosine similarity matrix
        sim = torch.matmul(proj, proj.transpose(-1, -2))
        
        # Positive pairs: (t, t+1)
        # Create labels for adjacent tokens
        labels = torch.arange(T, device=features.device)
        # Shifted targets (t -> t+1)
        targets = torch.roll(labels, -1)
        targets[-1] = -100 # Ignore last (standard PyTorch ignore_index)
        
        # This is a simplified proxy for semantic clustering
        # Real implementation would need augmentation or external pairs
        # For now, we encourage temporal smoothness in semantic space
        return F.cross_entropy(sim.view(-1, T), targets.repeat(B))

class ConceptStream(UnsupervisedStream):
    """Learns hierarchical compression (Autoencoder)"""
    def __init__(self, n_embd, dropout=0.1):
        super().__init__(n_embd, dropout)
        self.encoder = nn.Linear(n_embd, n_embd // 4)
        self.decoder = nn.Linear(n_embd // 4, n_embd)
        
    def forward(self, x):
        # Bottleneck forward pass
        encoded = self.encoder(self.norm(x))
        decoded = self.decoder(F.gelu(encoded))
        return x + decoded
        
    def unsupervised_loss(self, features, original_input=None):
        if original_input is None: return torch.tensor(0.0, device=features.device)
        
        # Reconstruct original input features
        # Note: 'features' here is the output of forward(), which is x + decoded
        # We want to minimize reconstruction error of the bottleneck
        
        # Re-run bottleneck for loss calculation (inefficient but clear)
        encoded = self.encoder(self.norm(original_input))
        decoded = self.decoder(F.gelu(encoded))
        
        return F.mse_loss(decoded, original_input)

class MetaFusionRouter(nn.Module):
    """Learns to route tokens to streams based on context"""
    def __init__(self, n_streams, n_embd, dropout=0.1):
        super().__init__()
        self.context_encoder = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=4,
            dim_feedforward=4*n_embd,
            dropout=dropout,
            batch_first=True
        )
        
        self.router = nn.Sequential(
            nn.Linear(n_embd, n_streams * 4),
            nn.GELU(),
            nn.Linear(n_streams * 4, n_streams),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, stream_features, context_features):
        # Encode context (with causal mask to prevent leakage)
        B, T, C = context_features.shape
        mask = torch.triu(torch.ones(T, T, device=context_features.device) * float('-inf'), diagonal=1)
        ctx = self.context_encoder(context_features, src_mask=mask)
        
        # Route per-token
        weights = self.router(ctx) # (B, T, n_streams)
        
        # Weighted fusion
        # stream_features is list of (B, T, C)
        # weights is (B, T, n_streams)
        
        fused = torch.zeros_like(stream_features[0])
        for i, feat in enumerate(stream_features):
            w = weights[:, :, i:i+1]
            fused = fused + w * feat
            
        return fused, weights

class KolosisX(nn.Module):
    """
    Kolosis-X: Self-Discovering Architecture
    """
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.dropout_rate = dropout
        
        # 1. Shared Components
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
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
        
        # 2. Self-Discovering Streams
        self.temporal_stream = TemporalStream(n_embd, block_size, dropout)
        self.semantic_stream = SemanticStream(n_embd, dropout)
        self.concept_stream = ConceptStream(n_embd, dropout)
        
        self.streams = nn.ModuleList([
            self.temporal_stream,
            self.semantic_stream,
            self.concept_stream
        ])
        
        # 3. Meta-Router
        self.router = MetaFusionRouter(len(self.streams), n_embd, dropout)
        
        # 4. Task-Specific Heads (for auxiliary supervision)
        # In Kolosis-X, we allow streams to have their own heads to specialize
        self.stream_heads = nn.ModuleList([
            nn.Linear(n_embd, vocab_size) for _ in self.streams
        ])
        
        # 5. Final Prediction Head
        self.final_head = nn.Linear(n_embd, vocab_size)
        
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
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        for layer in self.backbone:
            x = layer(x, src_mask=mask)
            
        backbone_features = x
        
        # 3. Stream Processing
        stream_outputs = []
        for stream in self.streams:
            stream_outputs.append(stream(backbone_features))
            
        # 4. Meta-Routing
        fused_feat, gate_weights = self.router(stream_outputs, backbone_features)
        
        # 5. Final Prediction
        logits = self.final_head(fused_feat)
        
        loss = None
        info = {}
        
        if targets is not None or return_stream_outputs:
            # Calculate stream logits
            stream_logits = [head(feat) for head, feat in zip(self.stream_heads, stream_outputs)]
            
            if targets is not None:
                # Main Task Loss
                main_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
                
                # Auxiliary Task Losses (Supervised)
                aux_losses = []
                for s_logits in stream_logits:
                    aux_losses.append(F.cross_entropy(s_logits.view(-1, self.vocab_size), targets.view(-1)))
                
                # Unsupervised Discovery Losses
                unsup_losses = []
                # Temporal: predict positions
                positions = torch.arange(T, device=idx.device).expand(B, T)
                unsup_losses.append(self.temporal_stream.unsupervised_loss(stream_outputs[0], positions=positions))
                # Semantic: contrastive
                unsup_losses.append(self.semantic_stream.unsupervised_loss(stream_outputs[1]))
                # Concept: reconstruction
                unsup_losses.append(self.concept_stream.unsupervised_loss(stream_outputs[2], original_input=backbone_features))
                
                # Diversity Loss
                diversity = self.compute_diversity_loss(stream_outputs)
                
                # Total Loss
                # 0.4 Main + 0.3 Aux + 0.2 Unsup + 0.1 Diversity
                avg_aux = sum(aux_losses) / len(aux_losses)
                avg_unsup = sum(unsup_losses) / len(unsup_losses)
                
                loss = (0.4 * main_loss + 
                        0.3 * avg_aux + 
                        0.2 * avg_unsup + 
                        0.1 * diversity)
                
                info['main_loss'] = main_loss.item()
                info['aux_loss'] = avg_aux.item()
                info['unsup_loss'] = avg_unsup.item()
                info['diversity_loss'] = diversity.item()

            if return_stream_outputs:
                info['gate_weights'] = gate_weights
                info['stream_logits'] = stream_logits
                
        return logits, loss, info

    def compute_diversity_loss(self, stream_features):
        """Encourage streams to be different (minimize cosine similarity)"""
        loss = 0.0
        n_pairs = 0
        for i in range(len(stream_features)):
            for j in range(i+1, len(stream_features)):
                # Flatten batch and time
                f1 = stream_features[i].view(-1, self.n_embd)
                f2 = stream_features[j].view(-1, self.n_embd)
                
                # Cosine similarity
                sim = F.cosine_similarity(f1, f2, dim=-1).mean()
                loss += sim
                n_pairs += 1
        
        if n_pairs > 0:
            return loss / n_pairs
        return torch.tensor(0.0, device=stream_features[0].device)

    def add_stream(self, new_stream_class, **kwargs):
        """Dynamic stream addition"""
        device = next(self.parameters()).device
        new_stream = new_stream_class(self.n_embd, **kwargs)
        self.streams.append(new_stream.to(device))
        
        new_head = nn.Linear(self.n_embd, self.vocab_size).to(device)
        self._init_weights(new_head)
        self.stream_heads.append(new_head)
        
        # Re-initialize router with new dimension
        # In a real scenario, we'd want to expand it without losing weights
        # For now, we'll just re-init
        self.router = MetaFusionRouter(len(self.streams), self.n_embd, self.dropout_rate).to(device)
        
        return new_stream

if __name__ == "__main__":
    print("Testing Kolosis-X")
    config = {
        'vocab_size': 100,
        'n_embd': 64,
        'block_size': 32,
        'n_layer': 2,
        'dropout': 0.1
    }
    model = KolosisX(**config)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    logits, loss, info = model(x, y)
    print(f"Loss: {loss.item():.4f}")
    print(f"Diversity: {info['diversity_loss']:.4f}")
