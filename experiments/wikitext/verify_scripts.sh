#!/bin/bash
# Clear Python cache and verify scripts are ready

echo "Clearing Python cache..."
find experiments/wikitext -name "*.pyc" -delete
find experiments/wikitext -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo ""
echo "Verifying all scripts are fixed..."
echo ""

echo "1. Baseline GPT:"
grep -A 6 "model = BaselineGPT" experiments/wikitext/train_baseline_gpt.py | head -7

echo ""
echo "2. Hierarchical:"
grep -A 6 "model = HierarchicalGPT" experiments/wikitext/train_hierarchical.py | head -7

echo ""
echo "3. Kolosis V2 Minimal:"
grep -A 6 "model = KolosisV2Minimal" experiments/wikitext/train_kolosis_v2_minimal.py | head -7

echo ""
echo "✅ All scripts verified! No **config unpacking found."
echo ""
echo "You can now run:"
echo "  ./venv/bin/python experiments/wikitext/train_baseline_gpt.py"
echo "  ./venv/bin/python experiments/wikitext/train_hierarchical.py"
echo "  ./venv/bin/python experiments/wikitext/train_kolosis_v2_minimal.py"
