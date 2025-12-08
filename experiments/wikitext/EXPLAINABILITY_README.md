# Explainability Test Dataset

## Purpose
This dataset tests whether Kolosis routing weights correlate with linguistic patterns.
If routing is interpretable, we expect:
- **Temporal sentences** → High Temporal stream activation
- **Causal sentences** → High Causal stream activation  
- **Conceptual sentences** → High Concept stream activation
- **Semantic sentences** → High Semantic stream activation
- **Neutral sentences** → Balanced activation

## Success Criteria
- **Alignment score >60%**: Routing is interpretable ✅
- **Alignment score 40-60%**: Partial interpretability ⚠️
- **Alignment score <40%**: Not interpretable ❌

## Categories

### Temporal (20 sentences)
Markers: "after", "before", "then", "first", "later", "eventually", "meanwhile"
Expected: Temporal stream dominant

### Causal (20 sentences)
Markers: "because", "therefore", "caused", "due to", "as a result", "led to"
Expected: Causal stream dominant

### Conceptual (20 sentences)
Markers: "is defined as", "is a", "consists of", "refers to", "includes"
Expected: Concept stream dominant

### Semantic (20 sentences)
Rich descriptive language, adjectives, emotional content
Expected: Semantic stream dominant

### Neutral (20 sentences)
Simple factual statements without strong markers
Expected: Balanced routing (no dominant stream)

## Usage

```python
import json

# Load dataset
with open('explainability_test_dataset.json') as f:
    dataset = json.load(f)

# Test a category
for sentence in dataset['temporal']['sentences']:
    routing_weights = model.get_routing(sentence)
    print(f"{sentence} -> {routing_weights}")
```

## Next Steps
1. Retrain Kolosis-S with checkpointing
2. Run inference on this dataset
3. Calculate alignment: expected_stream == argmax(routing_weights)
4. Generate report with visualizations
