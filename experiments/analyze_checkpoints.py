"""
Analyze WikiText-103 training checkpoints to evaluate model effectiveness.
"""
import pickle
import json
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """Load and analyze a checkpoint file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {checkpoint_path.name}")
    print('='*80)
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"\nCheckpoint Keys: {list(checkpoint.keys())}")
    
    # Extract and display metrics
    metrics_to_show = [
        'epoch', 'best_val_loss', 'best_val_perplexity', 
        'train_loss', 'val_loss', 'metrics', 'config'
    ]
    
    for key in metrics_to_show:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, (dict, list)):
                print(f"\n{key}:")
                print(json.dumps(value, indent=2, default=str))
            else:
                print(f"{key}: {value}")
    
    # Count parameters if model state is available
    if 'model_state_dict' in checkpoint:
        total_params = sum(p.numel() if hasattr(p, 'numel') else len(p.flatten()) 
                          for p in checkpoint['model_state_dict'].values())
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Model Size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return checkpoint

def main():
    results_dir = Path('/home/imsarthakshrma/Projects/RIIK/experiments/wikitext_results')
    
    # Analyze Kolosis checkpoint
    kolosis_path = results_dir / 'kolosis_v2_minimal_4gb_best.pt'
    kolosis_ckpt = analyze_checkpoint(kolosis_path)
    
    # Analyze Baseline checkpoint
    baseline_path = results_dir / 'baseline_gpt_best.pt'
    baseline_ckpt = analyze_checkpoint(baseline_path)
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print('='*80)
    
    if 'best_val_loss' in kolosis_ckpt and 'best_val_loss' in baseline_ckpt:
        kolosis_loss = kolosis_ckpt['best_val_loss']
        baseline_loss = baseline_ckpt['best_val_loss']
        improvement = ((baseline_loss - kolosis_loss) / baseline_loss) * 100
        
        print(f"\nValidation Loss:")
        print(f"  Baseline GPT:       {baseline_loss:.4f}")
        print(f"  Kolosis V2 Minimal: {kolosis_loss:.4f}")
        print(f"  Improvement:        {improvement:+.2f}%")
    
    if 'best_val_perplexity' in kolosis_ckpt and 'best_val_perplexity' in baseline_ckpt:
        kolosis_ppl = kolosis_ckpt['best_val_perplexity']
        baseline_ppl = baseline_ckpt['best_val_perplexity']
        improvement = ((baseline_ppl - kolosis_ppl) / baseline_ppl) * 100
        
        print(f"\nValidation Perplexity:")
        print(f"  Baseline GPT:       {baseline_ppl:.4f}")
        print(f"  Kolosis V2 Minimal: {kolosis_ppl:.4f}")
        print(f"  Improvement:        {improvement:+.2f}%")
    
    # File size comparison
    kolosis_size = kolosis_path.stat().st_size / 1024 / 1024
    baseline_size = baseline_path.stat().st_size / 1024 / 1024
    size_reduction = ((baseline_size - kolosis_size) / baseline_size) * 100
    
    print(f"\nModel Size:")
    print(f"  Baseline GPT:       {baseline_size:.2f} MB")
    print(f"  Kolosis V2 Minimal: {kolosis_size:.2f} MB")
    print(f"  Size Reduction:     {size_reduction:.2f}%")

if __name__ == '__main__':
    main()
