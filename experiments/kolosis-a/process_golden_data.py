import json
import re
from typing import Dict, Optional

def extract_routing_json(text: str) -> Optional[Dict]:
    """
    Parses the 'final_synthesis' text to find and extract the 
    JSON routing block requested in the Orchestration Guide.
    """
    # Look for JSON block delimited by ```json or just { }
    pattern = r"\{[\s\S]*\"routing_recommendation\"[\s\S]*\}"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def process_golden_file(input_path: str, output_path: str):
    """
    Cleans the raw Kolosis-A output and prepares it for model training.
    """
    processed_count = 0
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            
            # Extract weights if they exist in the text but not the field
            text_weights = extract_routing_json(data.get("final_synthesis", ""))
            
            if text_weights:
                data["routing_weights"] = text_weights.get("routing_recommendation", {})
            
            # Basic validation: ensure weights sum to ~1.0 if they exist
            weights = data.get("routing_weights", {})
            if weights:
                total = sum(weights.values())
                if not (0.95 <= total <= 1.05):
                    print(f"⚠️ Warning: Weights for prompt '{data['prompt'][:30]}...' sum to {total:.2f}")
            
            f_out.write(json.dumps(data) + "\n")
            processed_count += 1
            
    print(f"✅ Processed {processed_count} entries. Saved to: {output_path}")

if __name__ == "__main__":
    # Test on the user's data
    # Assuming the user saves the 'result' from the chat to 'raw_golden_data.jsonl'
    process_golden_file("golden_data.jsonl", "processed_golden_data.jsonl")
