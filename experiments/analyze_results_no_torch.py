"""
Analyze WikiText-103 training results without requiring torch.
Extract metrics from checkpoint metadata using Python's zipfile.
"""
import zipfile
import pickle
import json
from pathlib import Path
import struct

def extract_checkpoint_metadata(checkpoint_path):
    """
    Extract metadata from PyTorch checkpoint without loading torch.
    PyTorch .pt files are ZIP archives containing pickle data.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {checkpoint_path.name}")
    print('='*80)
    
    try:
        # Try to read as a ZIP file (PyTorch format)
        with zipfile.ZipFile(checkpoint_path, 'r') as z:
            # List contents
            print(f"\nArchive contents: {z.namelist()[:5]}...")  # Show first 5 files
            
            # Try to read data.pkl which contains the main checkpoint dict
            if 'data.pkl' in z.namelist():
                with z.open('data.pkl') as f:
                    # Read the pickle protocol and basic structure
                    data = f.read(1000)  # Read first 1000 bytes
                    print(f"\nCheckpoint appears to be in PyTorch ZIP format")
                    print(f"Size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
                    return None
    except zipfile.BadZipFile:
        print(f"Not a ZIP file, trying direct pickle read...")
        # Older PyTorch format - direct pickle
        try:
            with open(checkpoint_path, 'rb') as f:
                # Read magic bytes
                magic = f.read(2)
                print(f"File magic bytes: {magic}")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    return None

def analyze_from_training_output():
    """
    Since we can't easily load the checkpoints without torch,
    let's check if there are any training logs or output files.
    """
    results_dir = Path('/home/imsarthakshrma/Projects/RIIK/experiments/wikitext_results')
    
    print(f"\n{'='*80}")
    print("SEARCHING FOR TRAINING RESULTS")
    print('='*80)
    
    # Check for JSON results files
    json_files = list(results_dir.glob('*.json'))
    if json_files:
        print(f"\nFound {len(json_files)} JSON result files:")
        for jf in json_files:
            print(f"  - {jf.name}")
            with open(jf) as f:
                data = json.load(f)
                print(f"    Keys: {list(data.keys())}")
    else:
        print("\n⚠️  No JSON result files found")
        print("The training scripts should have created *_results.json files")
    
    # List all files
    print(f"\nAll files in results directory:")
    for f in results_dir.iterdir():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.2f} MB)")

def main():
    print("="*80)
    print("WIKITEXT-103 TRAINING RESULTS ANALYSIS")
    print("="*80)
    
    results_dir = Path('/home/imsarthakshrma/Projects/RIIK/experiments/wikitext_results')
    
    # Try to extract metadata from checkpoints
    for ckpt_file in ['baseline_gpt_best.pt', 'kolosis_v2_minimal_4gb_best.pt']:
        ckpt_path = results_dir / ckpt_file
        if ckpt_path.exists():
            extract_checkpoint_metadata(ckpt_path)
    
    # Look for training results
    analyze_from_training_output()
    
    print(f"\n{'='*80}")
    print("ANALYSIS NOTES")
    print('='*80)
    print("""
To fully analyze the checkpoints, we need to either:
1. Find the JSON results files (*_results.json) that should have been created during training
2. Re-run the evaluation with torch installed
3. Check training logs/output for the final metrics

The checkpoint files themselves require PyTorch to load properly.
""")

if __name__ == '__main__':
    main()
