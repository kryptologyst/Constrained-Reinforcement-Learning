#!/bin/bash

# Quick start script for Constrained Reinforcement Learning

echo "🤖 Constrained Reinforcement Learning - Quick Start"
echo "=================================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.10+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Create output directory
mkdir -p outputs

# Run a quick test
echo ""
echo "🧪 Running quick test..."
python modernized_example.py --episodes 50 --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Quick start completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Train a model: python scripts/train.py --algorithm constrained_q --num_episodes 1000"
    echo "2. Run the demo: streamlit run demo/app.py"
    echo "3. Run tests: pytest tests/"
    echo ""
    echo "⚠️  DISCLAIMER: This is for research/educational purposes only."
    echo "   Not for production control of real systems."
else
    echo "❌ Error: Quick test failed"
    exit 1
fi
