"""
Kolosis Interpretability Analysis
=================================
This script analyzes whether Kolosis streams specialize for different linguistic patterns.
If streams show meaningful specialization, it proves Kolosis has value beyond perplexity.

Key hypothesis: Different sentence types should activate different streams preferentially.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict
import os
import sys

# Add experiments path
sys.path.insert(0, '/home/imsarthakshrma/Projects/RIIK/experiments/wikitext')

# Test sentences grouped by expected dominant stream
TEST_SENTENCES = {
    "temporal": [
        "After the meeting ended, the team went to lunch.",
        "First came the thunder, then the rain started falling.",
        "Before the invention of electricity, people used candles.",
        "The company grew rapidly during the 1990s.",
        "Subsequently, the government passed new legislation.",
        "Meanwhile, scientists were making new discoveries.",
        "Eventually, all the pieces fell into place.",
        "Previously, this technology was considered impossible.",
        "The project started in January and finished in March.",
        "Once upon a time, there lived a wise king.",
    ],
    "causal": [
        "The stock crashed because the CEO resigned suddenly.",
        "Due to heavy rainfall, the roads were flooded.",
        "The experiment failed, therefore we revised our hypothesis.",
        "Pollution caused the river to become contaminated.",
        "As a result of the merger, many employees lost jobs.",
        "The fire was caused by an electrical fault.",
        "Consequently, the company had to file for bankruptcy.",
        "The disease spread rapidly, leading to a pandemic.",
        "Higher interest rates result in reduced spending.",
        "The bridge collapsed owing to structural weaknesses.",
    ],
    "conceptual": [
        "A mammal is a warm-blooded vertebrate animal.",
        "The Constitution defines the fundamental laws of the nation.",
        "Democracy is a system of government by the people.",
        "Photosynthesis is the process by which plants make food.",
        "The category of reptiles includes snakes and lizards.",
        "Philosophy is the study of fundamental nature of knowledge.",
        "Gravity is a force that attracts objects toward each other.",
        "An algorithm is a step-by-step procedure for calculations.",
        "The monarchy consists of a king or queen as head of state.",
        "Biology is the scientific study of living organisms.",
    ],
    "semantic": [
        "The happy children played joyfully in the sunny garden.",
        "She felt melancholy and nostalgic about her childhood.",
        "The ancient ruins revealed mysterious secrets.",
        "The brilliant scientist made a groundbreaking discovery.",
        "The delicious aroma of fresh bread filled the kitchen.",
        "His eloquent speech moved the entire audience to tears.",
        "The majestic mountains towered over the peaceful valley.",
        "The curious cat explored every corner of the room.",
        "The vibrant colors of the sunset painted the sky.",
        "The gentle breeze carried the sweet scent of flowers.",
    ],
    "neutral": [
        "The book is on the table.",
        "She walked to the store.",
        "He opened the door.",
        "They ate dinner together.",
        "The cat sat on the mat.",
        "I like to read books.",
        "The sun was shining.",
        "Birds fly in the sky.",
        "Water flows downhill.",
        "Trees have green leaves.",
    ]
}

def load_kolosis_x():
    """Load trained Kolosis-X model."""
    from train_kolosis_x_colab import KolosisX
    
    model_path = '/home/imsarthakshrma/Projects/RIIK/experiments/wikitext_results/kolosis_x_best.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with same config
    model = KolosisX(
        vocab_size=50257,
        n_embd=256,
        block_size=128,
        n_layer=6,
        n_head=8,
        dropout=0.1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    return model, device

def load_kolosis_s():
    """Load trained Kolosis-S model."""
    from train_kolosis_s_colab import KolosisS
    
    model_path = '/home/imsarthakshrma/Projects/RIIK/experiments/wikitext_results/kolosis_s_best.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with same config
    model = KolosisS(
        vocab_size=50257,
        n_embd=256,
        block_size=128,
        n_layer=6,
        n_head=8,
        dropout=0.1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    return model, device

def get_tokenizer():
    """Load GPT-2 tokenizer."""
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def analyze_routing(model, tokenizer, sentence, device, model_type='x'):
    """Get routing weights for a sentence."""
    tokens = tokenizer.encode(sentence, return_tensors='pt').to(device)
    
    with torch.no_grad():
        if model_type == 'x':
            # For Kolosis-X, we need to extract routing weights
            # Get embeddings
            tok_emb = model.token_embedding(tokens)
            pos_emb = model.position_embedding(torch.arange(tokens.size(1), device=device))
            x = model.drop(tok_emb + pos_emb)
            
            # Pass through transformer
            for block in model.transformer_blocks:
                x = block(x)
            
            # Get stream features
            idx = torch.arange(tokens.size(1), device=device).unsqueeze(0)
            temporal_feat = model.temporal_adapter(x)
            semantic_feat = model.semantic_adapter(x)
            concept_feat = model.concept_adapter(x, idx)
            causal_feat = model.causal_stream(x)
            
            stacked = torch.stack([temporal_feat, semantic_feat, concept_feat, causal_feat], dim=-2)
            
            # Get routing weights from router
            _, gate_weights, _ = model.router(stacked, temperature=1.0, use_gumbel=False)
            
            # Average across sequence
            avg_weights = gate_weights.mean(dim=[0, 1]).cpu().numpy()
            stream_names = ['Temporal', 'Semantic', 'Concept', 'Causal']
            
        else:  # Kolosis-S
            # For Kolosis-S
            tok_emb = model.token_embedding(tokens)
            pos_emb = model.position_embedding(torch.arange(tokens.size(1), device=device))
            x = model.drop(tok_emb + pos_emb)
            
            for block in model.transformer_blocks:
                x = block(x)
            
            idx = torch.arange(tokens.size(1), device=device).unsqueeze(0)
            symbol_feat = model.symbol_adapter(x)
            temporal_feat = model.temporal_adapter(x)
            semantic_feat = model.semantic_adapter(x)
            concept_feat = model.concept_adapter(x, idx)
            
            stacked = torch.stack([symbol_feat, temporal_feat, semantic_feat, concept_feat], dim=-2)
            
            _, gate_weights, _ = model.fusion_gate(stacked, temperature=1.0, use_gumbel=False)
            
            avg_weights = gate_weights.mean(dim=[0, 1]).cpu().numpy()
            stream_names = ['Symbol', 'Temporal', 'Semantic', 'Concept']
    
    return avg_weights, stream_names

def run_analysis():
    """Run full interpretability analysis."""
    print("=" * 70)
    print("KOLOSIS INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()
    
    results = {}
    
    # Analyze Kolosis-X
    print("\n" + "=" * 50)
    print("ANALYZING KOLOSIS-X")
    print("=" * 50)
    
    model_x, device = load_kolosis_x()
    if model_x is not None:
        results['kolosis_x'] = analyze_model(model_x, tokenizer, device, 'x')
    else:
        print("Kolosis-X model not found!")
    
    # Analyze Kolosis-S
    print("\n" + "=" * 50)
    print("ANALYZING KOLOSIS-S")
    print("=" * 50)
    
    model_s, device = load_kolosis_s()
    if model_s is not None:
        results['kolosis_s'] = analyze_model(model_s, tokenizer, device, 's')
    else:
        print("Kolosis-S model not found!")
    
    # Save results
    output_path = '/home/imsarthakshrma/Projects/RIIK/experiments/wikitext_results/interpretability_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results

def analyze_model(model, tokenizer, device, model_type):
    """Analyze routing patterns for a model."""
    category_results = {}
    
    for category, sentences in TEST_SENTENCES.items():
        print(f"\n--- {category.upper()} sentences ---")
        
        all_weights = []
        for sentence in sentences:
            weights, stream_names = analyze_routing(model, tokenizer, sentence, device, model_type)
            all_weights.append(weights)
        
        # Average across sentences in category
        avg_weights = np.mean(all_weights, axis=0)
        std_weights = np.std(all_weights, axis=0)
        
        # Find dominant stream
        dominant_idx = np.argmax(avg_weights)
        dominant_stream = stream_names[dominant_idx]
        
        print(f"  Average routing: {' / '.join([f'{stream_names[i]}: {avg_weights[i]:.1%}' for i in range(len(stream_names))])}")
        print(f"  Dominant stream: {dominant_stream} ({avg_weights[dominant_idx]:.1%})")
        
        category_results[category] = {
            'avg_weights': avg_weights.tolist(),
            'std_weights': std_weights.tolist(),
            'dominant_stream': dominant_stream,
            'stream_names': stream_names
        }
    
    # Check if there's meaningful differentiation
    print("\n" + "-" * 50)
    print("SPECIALIZATION ANALYSIS")
    print("-" * 50)
    
    # Calculate variance across categories for each stream
    all_avgs = np.array([category_results[cat]['avg_weights'] for cat in TEST_SENTENCES.keys()])
    stream_variance = np.var(all_avgs, axis=0)
    
    print(f"\nStream variance across categories:")
    for i, name in enumerate(stream_names):
        print(f"  {name}: {stream_variance[i]:.4f}")
    
    total_variance = stream_variance.sum()
    print(f"\nTotal variance: {total_variance:.4f}")
    
    if total_variance > 0.01:
        print("✅ MEANINGFUL SPECIALIZATION DETECTED!")
        print("   Streams show different routing patterns for different sentence types.")
    else:
        print("⚠️ LIMITED SPECIALIZATION")
        print("   Streams show similar routing regardless of sentence type.")
    
    # Check expected alignments
    print("\n" + "-" * 50)
    print("EXPECTED ALIGNMENT CHECK")
    print("-" * 50)
    
    if model_type == 'x':
        expected = {
            'temporal': 'Temporal',
            'causal': 'Causal', 
            'conceptual': 'Concept',
            'semantic': 'Semantic'
        }
    else:
        expected = {
            'temporal': 'Temporal',
            'conceptual': 'Concept',
            'semantic': 'Semantic'
        }
    
    matches = 0
    total = 0
    for category, exp_stream in expected.items():
        if category in category_results:
            actual = category_results[category]['dominant_stream']
            match = actual == exp_stream
            status = "✅" if match else "❌"
            print(f"  {category}: Expected {exp_stream}, Got {actual} {status}")
            if match:
                matches += 1
            total += 1
    
    alignment_score = matches / total if total > 0 else 0
    print(f"\nAlignment score: {matches}/{total} ({alignment_score:.0%})")
    
    category_results['summary'] = {
        'total_variance': float(total_variance),
        'specialization_detected': total_variance > 0.01,
        'alignment_score': alignment_score,
        'matches': matches,
        'total': total
    }
    
    return category_results

if __name__ == "__main__":
    results = run_analysis()
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    for model_name, model_results in results.items():
        if 'summary' in model_results:
            summary = model_results['summary']
            print(f"\n{model_name.upper()}:")
            print(f"  Specialization detected: {'YES' if summary['specialization_detected'] else 'NO'}")
            print(f"  Total variance: {summary['total_variance']:.4f}")
            print(f"  Expected alignment: {summary['matches']}/{summary['total']} ({summary['alignment_score']:.0%})")
