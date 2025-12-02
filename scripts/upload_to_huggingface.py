"""
Hugging Face integration for Kolosis models.
Enables uploading and downloading from Hugging Face Hub.
"""

import torch
from huggingface_hub import hf_hub_download, upload_file, create_repo
from pathlib import Path
import json


def save_for_huggingface(model, tokenizer, save_directory, model_name="kolosis-v2-minimal"):
    """
    Save model in Hugging Face format.
    
    Args:
        model: Kolosis model instance
        tokenizer: Tokenizer (GPT-2)
        save_directory: Local directory to save files
        model_name: Name for the model
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
    
    # Save config
    config = {
        "model_type": "kolosis",
        "architecture": "KolosisV2MinimalSingleHead",
        "vocab_size": model.vocab_size,
        "n_embd": model.n_embd,
        "n_layer": model.n_layer,
        "block_size": model.block_size,
        "dropout": model.dropout,
    }
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    print(f"✅ Model saved to {save_dir}")
    print(f"Files: pytorch_model.bin, config.json, tokenizer files")


def upload_to_huggingface(
    local_directory,
    repo_id,
    token=None,
    private=False,
    commit_message="Upload Kolosis model"
):
    """
    Upload model to Hugging Face Hub.
    
    Args:
        local_directory: Directory containing model files
        repo_id: Hugging Face repo ID (username/model-name)
        token: HF token (or use huggingface-cli login)
        private: Whether repo should be private
        commit_message: Commit message
    """
    from huggingface_hub import HfApi
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠️ Repo creation: {e}")
    
    # Upload all files
    local_dir = Path(local_directory)
    
    files_to_upload = [
        "pytorch_model.bin",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    
    for filename in files_to_upload:
        filepath = local_dir / filename
        if filepath.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    token=token,
                    commit_message=commit_message
                )
                print(f"✅ Uploaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to upload {filename}: {e}")
        else:
            print(f"⚠️ File not found: {filename}")
    
    # Upload README (model card)
    readme_path = Path("MODEL_CARD.md")
    if readme_path.exists():
        try:
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=token,
                commit_message="Add model card"
            )
            print(f"✅ Uploaded: README.md (model card)")
        except Exception as e:
            print(f"❌ Failed to upload README: {e}")
    
    print(f"\n🎉 Model uploaded to: https://huggingface.co/{repo_id}")


def download_from_huggingface(repo_id, local_directory="./downloaded_model"):
    """
    Download model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repo ID (username/model-name)
        local_directory: Where to save downloaded files
    """
    local_dir = Path(local_directory)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = [
        "pytorch_model.bin",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    
    for filename in files_to_download:
        try:
            filepath = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=local_dir
            )
            print(f"✅ Downloaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    
    print(f"\n✅ Model downloaded to: {local_dir}")


# Example usage
if __name__ == "__main__":
    """
    Example: Upload Kolosis V2 Minimal to Hugging Face
    """
    
    # 1. Load your trained model
    from neural_networks.kolosis.kolosis_v2_minimal_single_head import KolosisV2MinimalSingleHead
    from transformers import GPT2Tokenizer
    
    # Load model
    model = KolosisV2MinimalSingleHead(
        vocab_size=50257,
        n_embd=128,
        n_layer=4,
        block_size=128,
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint_path = "experiments/wikitext_results/kolosis_v2_minimal_single_head_best.pt"
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 2. Save in HF format
    save_for_huggingface(
        model=model,
        tokenizer=tokenizer,
        save_directory="./hf_model",
        model_name="kolosis-v2-minimal"
    )
    
    # 3. Upload to HF (replace with your username)
    upload_to_huggingface(
        local_directory="./hf_model",
        repo_id="yourusername/kolosis-v2-minimal",
        commit_message="Initial upload: Kolosis V2 Minimal (Research Preview)"
    )
    
    print("\n🎉 Done! Your model is now on Hugging Face!")
    print("View it at: https://huggingface.co/yourusername/kolosis-v2-minimal")
