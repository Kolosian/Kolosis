import torch
import json
import numpy as np
from transformers import GPT2Tokenizer
import sys
import os

# Add current dir to path to import KolosisS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_kolosis_s_colab import KolosisS

def validate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Dataset
    dataset_path = 'experiments/wikitext/explainability_test_dataset.json'
    if not os.path.exists(dataset_path):
        # Try looking in current dir if running from experiments/wikitext
        dataset_path = 'explainability_test_dataset.json'
        
    with open(dataset_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test sentences")

    # 2. Load Model (Use training defaults: 128/2)
    model = KolosisS(
        vocab_size=50257,
        n_embd=128,
        block_size=128,
        n_layer=2,
        dropout=0.1
    )
    
    checkpoint_path = 'experiments/Kolosis_S_checkpoints/kolosis_s_best.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle state dict loading
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")

    # 3. Setup Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Run Validation
    categories = ['temporal', 'causal', 'conceptual', 'semantic', 'neutral']
    results = {cat: {'counts': np.zeros(4), 'total': 0} for cat in categories}
    stream_names = ['Symbol', 'Temporal', 'Semantic', 'Concept']

    print("\nRunning inference...")
    for cat_name, cat_data in test_data.items():
        category = cat_name
        sentences = cat_data['sentences']
        
        for sentence in sentences:
            tokens = tokenizer.encode(sentence, return_tensors='pt').to(device)
        
        with torch.no_grad():
            logits, loss, info = model(tokens, return_stream_outputs=True)
            gate_weights = info['gate_weights'] # [B, T, 4]
            
            # Average over time dimension
            avg_weights = gate_weights.mean(dim=[0, 1]).cpu().numpy()
            
            results[category]['counts'] += avg_weights
            results[category]['total'] += 1

    # 5. Print Report
    print("\n" + "="*60)
    print("EXPLAINABILITY VALIDATION REPORT")
    print("="*60)
    
    alignment_score = 0
    total_checks = 0
    
    for cat in categories:
        if results[cat]['total'] == 0: continue
        
        avg_dist = results[cat]['counts'] / results[cat]['total']
        dominant_idx = np.argmax(avg_dist)
        dominant_name = stream_names[dominant_idx]
        
        print(f"\n--- {cat.upper()} ---")
        print(f"Distribution: {', '.join([f'{n}: {p:.1%}' for n, p in zip(stream_names, avg_dist)])}")
        print(f"Dominant: {dominant_name} ({avg_dist[dominant_idx]:.1%})")
        
        # Check alignment
        is_aligned = False
        expected = "None"
        
        if cat == 'temporal':
            expected = 'Temporal'
            if dominant_idx == 1: is_aligned = True
        elif cat == 'causal':
            expected = 'Semantic'
            if dominant_idx == 2: is_aligned = True
        elif cat == 'semantic':
            expected = 'Semantic'
            if dominant_idx == 2: is_aligned = True
        elif cat == 'conceptual':
            expected = 'Concept'
            if dominant_idx == 3: is_aligned = True
        elif cat == 'neutral':
            expected = 'Balanced'
            # Check if max is < 0.4
            if avg_dist[dominant_idx] < 0.4: is_aligned = True
            
        print(f"Expected: {expected} | Match: {'✅' if is_aligned else '❌'}")
        
        if cat != 'neutral':
            total_checks += 1
            if is_aligned: alignment_score += 1

    print("="*60)
    final_score = (alignment_score / total_checks) * 100 if total_checks > 0 else 0
    print(f"FINAL ALIGNMENT SCORE: {final_score:.1f}%")
    print("="*60)

if __name__ == "__main__":
    validate()
