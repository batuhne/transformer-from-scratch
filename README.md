# Transformer from Scratch (NumPy Only)

A **fully trainable decoder-only transformer** (GPT-style) built entirely with NumPy. No PyTorch, no TensorFlow. Just pure math.

## What is this?

This project teaches you how a **GPT-like language model** works by building one from zero. Every component (attention, backpropagation, optimizer) is implemented by hand in NumPy.

The model learns to predict the next character in Shakespeare text. After ~60 seconds of training, it generates text like:

```
Before training:  "First>;&qZ!mK..."        (random garbage)
After training:   "First Citizen:\nI the great toe! why the great toe!"  (learned patterns!)
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Token Indices (characters)       │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  Character Embedding (vocab=65, dim=64)  │
│  + Sinusoidal Positional Encoding        │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│        3x Transformer Block              │
│  ┌───────────────────────────────────┐   │
│  │ LayerNorm → Multi-Head Attention  │   │
│  │ (4 heads, d_k=16) + Residual     │   │
│  ├───────────────────────────────────┤   │
│  │ LayerNorm → FFN (256, ReLU)      │   │
│  │ + Residual                        │   │
│  └───────────────────────────────────┘   │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  LayerNorm → Linear → Softmax           │
│  Output: probability of next character   │
└─────────────────────────────────────────┘
```

**~150K parameters** | Trains on CPU in ~60 seconds | 3,000 training steps

## Project Structure

```
transformer-from-scratch/
├── README.md
├── requirements.txt          # numpy, matplotlib, jupyter
├── data/
│   └── input.txt             # Shakespeare training corpus
└── notebooks/
    ├── 01_foundations.ipynb           # Linear layer, ReLU, softmax, gradients
    ├── 02_embeddings_and_data.ipynb   # Tokenizer, embeddings, positional encoding
    ├── 03_self_attention.ipynb        # Scaled dot-product attention, causal mask
    ├── 04_multihead_attention.ipynb   # Multi-head attention + gradient checking
    ├── 05_ffn_and_layernorm.ipynb     # Feed-forward network, layer normalization
    ├── 06_transformer_model.ipynb     # Assemble full decoder-only transformer
    └── 07_training_and_generation.ipynb  # Train the model + generate text
```

## How to Use

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook notebooks/
```

Then open the notebooks in order: **01 → 02 → 03 → 04 → 05 → 06 → 07**

Each notebook is self-contained and can run independently.

## What Each Notebook Teaches

| # | Notebook | You'll Learn |
|---|----------|-------------|
| 01 | Foundations | How neural networks compute gradients (backprop) |
| 02 | Embeddings & Data | How text becomes numbers a model can process |
| 03 | Self-Attention | The core idea behind transformers: "which words should I look at?" |
| 04 | Multi-Head Attention | Running multiple attention patterns in parallel |
| 05 | FFN & LayerNorm | The "thinking" layers and training stabilization |
| 06 | Transformer Model | Putting all pieces together into one model |
| 07 | Training & Generation | Teaching the model to write Shakespeare |

## Training Results

| Metric | Value |
|--------|-------|
| Initial loss | ~3.08 (random guessing) |
| Final loss | ~0.46 |
| Training time | ~60 seconds |
| Speed | ~50 steps/second |

## Dependencies

- **numpy**: all math and model implementation
- **matplotlib**: visualizations only
- **jupyter**: notebook environment

No PyTorch. No TensorFlow. No pre-trained embeddings. Truly from scratch.
