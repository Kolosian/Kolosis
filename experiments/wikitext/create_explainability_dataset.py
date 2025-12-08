"""
Explainability Test Dataset Generator
=====================================
Creates a labeled dataset to test if Kolosis routing is interpretable.

This is CRITICAL for validating the business value proposition.
If routing doesn't correlate with sentence types, the whole pitch fails.
"""

import json
from pathlib import Path

# Expanded test dataset with clear linguistic patterns
TEST_DATASET = {
    "temporal": {
        "description": "Sentences with temporal/sequential markers",
        "expected_stream": "Temporal",
        "sentences": [
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
            "Later that evening, the storm intensified.",
            "Following the announcement, stock prices dropped.",
            "Initially, the plan seemed feasible.",
            "Afterwards, everyone celebrated the victory.",
            "Soon after sunrise, the birds began singing.",
            "By the end of the decade, everything had changed.",
            "Next week, the conference will begin.",
            "Years later, they finally understood the truth.",
            "At that moment, everything became clear.",
            "Throughout history, empires have risen and fallen.",
        ]
    },
    "causal": {
        "description": "Sentences with cause-effect relationships",
        "expected_stream": "Causal",
        "sentences": [
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
            "Since the deadline passed, we cannot submit anymore.",
            "Thanks to modern medicine, mortality rates decreased.",
            "The drought led to widespread crop failures.",
            "Because of the traffic, we arrived late.",
            "The policy change triggered massive protests.",
            "Smoking causes lung cancer and other diseases.",
            "The earthquake resulted in thousands of casualties.",
            "Her hard work brought about remarkable success.",
            "The scandal forced the minister to resign.",
            "Climate change is driving species extinction.",
        ]
    },
    "conceptual": {
        "description": "Sentences with definitions and hierarchical concepts",
        "expected_stream": "Concept",
        "sentences": [
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
            "Entropy measures the disorder in a system.",
            "A triangle is a polygon with three sides.",
            "Justice refers to fairness in the treatment of people.",
            "Capitalism is an economic system based on private ownership.",
            "The periodic table organizes elements by atomic number.",
            "A metaphor is a figure of speech comparing two things.",
            "Velocity is the rate of change of position.",
            "The Renaissance was a period of cultural rebirth.",
            "DNA contains the genetic instructions for organisms.",
            "Ethics is the branch of philosophy concerning morality.",
        ]
    },
    "semantic": {
        "description": "Sentences rich in semantic/descriptive content",
        "expected_stream": "Semantic",
        "sentences": [
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
            "The magnificent cathedral displayed stunning architecture.",
            "Her radiant smile brightened everyone's day.",
            "The turbulent ocean waves crashed against the rocks.",
            "The serene lake reflected the surrounding forest.",
            "The passionate musician performed with incredible emotion.",
            "The cozy cottage nestled among the rolling hills.",
            "The fierce lion prowled through the tall grass.",
            "The elegant dancer moved with graceful precision.",
            "The mysterious stranger wore a dark cloak.",
            "The cheerful melody lifted everyone's spirits.",
        ]
    },
    "neutral": {
        "description": "Generic sentences without strong linguistic markers",
        "expected_stream": "None (balanced)",
        "sentences": [
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
            "The car is red.",
            "People live in houses.",
            "The phone rang twice.",
            "She has brown hair.",
            "The clock shows three.",
            "Dogs bark at strangers.",
            "The window is closed.",
            "He drinks coffee daily.",
            "The room is quiet.",
            "They own two bicycles.",
        ]
    }
}

def save_test_dataset():
    """Save the test dataset to JSON."""
    output_path = Path('/home/imsarthakshrma/Projects/RIIK/experiments/wikitext/explainability_test_dataset.json')
    
    with open(output_path, 'w') as f:
        json.dump(TEST_DATASET, f, indent=2)
    
    print(f"✅ Test dataset saved to {output_path}")
    print(f"\nDataset statistics:")
    for category, data in TEST_DATASET.items():
        print(f"  {category}: {len(data['sentences'])} sentences")
    
    total = sum(len(data['sentences']) for data in TEST_DATASET.values())
    print(f"\nTotal: {total} sentences")
    
    return output_path

def create_readme():
    """Create README for the test dataset."""
    readme = """# Explainability Test Dataset

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
"""
    
    readme_path = Path('/home/imsarthakshrma/Projects/RIIK/experiments/wikitext/EXPLAINABILITY_README.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    print(f"✅ README saved to {readme_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("CREATING EXPLAINABILITY TEST DATASET")
    print("=" * 70)
    print()
    
    dataset_path = save_test_dataset()
    create_readme()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. ✅ Test dataset created (100 sentences)
2. ⏭️  Retrain Kolosis-S with checkpointing enabled
3. ⏭️  Run inference on test dataset
4. ⏭️  Calculate alignment score
5. ⏭️  Generate validation report

To retrain with checkpointing:
  cd experiments/wikitext
  # Modify train_kolosis_s_colab.py to save checkpoints
  # Run training for 3-5 epochs (enough to see specialization)

To test explainability:
  python test_explainability.py  # (will create this next)
""")
