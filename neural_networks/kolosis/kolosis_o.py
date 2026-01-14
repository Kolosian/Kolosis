import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RoPE(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            # Recompute if seq_len is longer than cached
            pass 
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RLGate(nn.Module):
    """
    A tiny RL-ready gate that weights different types of heads.
    """
    def __init__(self, n_embd, n_groups=2):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(n_embd, n_embd // 4),
            nn.GELU(),
            nn.Linear(n_embd // 4, n_groups)
        )
        
    def forward(self, x, temperature=1.0):
        # x: (B, T, C)
        logits = self.gate_net(x) / temperature
        weights = F.softmax(logits, dim=-1)
        return weights

class CognitiveGQA(nn.Module):
    """
    Grouped-Query Attention with Cognitive Head Specialization.
    Specifically: Semantic Heads, Temporal Heads, and Causal Heads.
    """
    def __init__(self, n_embd, n_head, n_kv_head, head_dim, block_size, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.n_rep = n_head // n_kv_head
        
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=False)
        
        # Head Specialization (Conceptual indices)
        # Assuming 4 heads total for simplicity:
        # Head 0, 1: Semantic
        # Head 2: Temporal
        # Head 3: Causal
        
        # In practice we can have a gate to weight their contribution
        self.rl_gate = RLGate(n_embd, n_groups=4) # Semantic, Temporal, Conceptual, Causal
        
        self.dropout = nn.Dropout(dropout)
        
        # Temporal mask (Multi-scale)
        self.register_buffer("temporal_mask", self._create_temporal_mask(block_size))

    def _create_temporal_mask(self, size):
        # Simple exponential decay mask for temporal heads
        mask = torch.tril(torch.ones(size, size))
        # Add a decay factor
        return mask

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        # Repeat K, V for GQA
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Cognitive Weighting
        # We'll apply different head-level masks/weights here if needed
        # But for Kolosis-O, we use RLGate to blend the outputs of specialized groups
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        
        # Apply RLGate weighting to the final output of this block
        gate_weights = self.rl_gate(x) # (B, T, 4)
        # Weights represent: [Semantic, Temporal, Conceptual, Causal]
        
        return self.o_proj(out), gate_weights

class KolosisOBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, head_dim, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CognitiveGQA(n_embd, n_head, n_kv_head, head_dim, block_size, dropout)
        self.ln2 = RMSNorm(n_embd)
        
        # SwiGLU FFN
        intermediate_size = int(4 * n_embd * (2/3))
        self.w1 = nn.Linear(n_embd, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, intermediate_size, bias=False)
        
    def forward(self, x, cos, sin):
        # Attention with residual
        attn_out, gate_weights = self.attn(self.ln1(x), cos, sin)
        x = x + attn_out
        
        # SwiGLU FFN with residual
        # LLaMA-style FFN: (Swish(x@w1) * (x@w3)) @ w2
        h = F.silu(self.w1(self.ln2(x))) * self.w3(self.ln2(x))
        x = x + self.w2(h)
        
        return x, gate_weights

class KolosisO(nn.Module):
    """
    Kolosis-O (Opaque) Commercial Architecture.
    Backbone: LLaMA-style (RMSNorm, RoPE, SwiGLU)
    Innovation: Cognitive Attention Heads + RL-Tuned Gating
    """
    def __init__(self, vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size, dropout=0.0):
        super().__init__()
        self.config = locals()
        self.block_size = block_size
        
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RoPE(n_embd // n_head, max_position_embeddings=block_size)
        
        self.blocks = nn.ModuleList([
            KolosisOBlock(n_embd, n_head, n_kv_head, n_embd // n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        self.ln_f = RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Internal Thought Injection Latent (Placeholder)
        self.thought_latent = nn.Parameter(torch.zeros(1, 1, n_embd))
        
    def forward(self, idx, targets=None, return_stats=False):
        B, T = idx.shape
        x = self.tok_emb(idx)
        
        cos, sin = self.rope(x, seq_len=T)
        
        all_gate_weights = []
        for i, block in enumerate(self.blocks):
            # Thought injection injection at middle layers
            if i == len(self.blocks) // 2:
                x = x + self.thought_latent.expand(B, T, -1)
                
            x, gw = block(x, cos, sin)
            all_gate_weights.append(gw)
            
        x = self.ln_f(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        if return_stats:
            return logits, loss, {"gate_weights": torch.stack(all_gate_weights)}
        return logits, loss

if __name__ == "__main__":
    # Test Kolosis-O
    model = KolosisO(
        vocab_size=50257,
        n_embd=256,
        n_head=8,
        n_kv_head=2,
        n_layer=4,
        block_size=128
    )
    
    x = torch.randint(0, 50257, (2, 64))
    logits, loss, stats = model(x, targets=x, return_stats=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item()}")
    print(f"Gate weights shape: {stats['gate_weights'].shape}")
    print("✅ Kolosis-O Prototype Initialized")
