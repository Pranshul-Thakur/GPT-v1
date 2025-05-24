# GPT v1 - Character-Level Transformer

A lightweight implementation of a GPT-style transformer model for character-level text generation. This project demonstrates the core concepts of transformer architecture including self-attention, multi-head attention, and autoregressive text generation.

## Features

- **Character-level tokenization**: Learns patterns at the character level
- **Multi-head self-attention**: Implements scaled dot-product attention with multiple heads
- **Transformer blocks**: Layer normalization, feedforward networks, and residual connections
- **Configurable architecture**: Easily adjustable model parameters
- **Memory-mapped file reading**: Efficient handling of large text files
- **Model persistence**: Save and load trained models

## Architecture

- **Embedding dimension**: 384
- **Number of attention heads**: 4
- **Number of transformer layers**: 4
- **Context window**: 128 tokens
- **Feedforward expansion**: 4x embedding dimension
- **Dropout rate**: 0.2

## Requirements

```bash
pip install torch
```

## Quick Start

### 1. Prepare Your Data

Create a `data.txt` file in the project root with your training text. This can be any text content you want the model to learn from:

```
Your training text goes here.
It can be multiple lines.
The model will learn character-level patterns from this data.
```

### 2. Train the Model

```bash
python BPE.py
```

The script will:
- Load and process your `data.txt` file
- Train the model for the specified number of iterations
- Save the trained model as `model-01.pkl`
- Generate sample text using the prompt "Hello! Can you see me?"

### 3. Monitor Training

The script outputs training progress including:
- Current iteration number
- Training and validation loss (every 100 iterations by default)
- Final loss value

## Configuration

You can modify the model hyperparameters at the top of `BPE.py`:

```python
# Training parameters
batch_size = 32          # Number of sequences per batch
block_size = 128         # Context window size
max_iters = 200          # Training iterations (increase for better results)
learning_rate = 2e-5     # Adam optimizer learning rate
eval_iters = 100         # Evaluation frequency

# Model architecture
n_embd = 384            # Embedding dimension
n_head = 4              # Number of attention heads
n_layer = 4             # Number of transformer blocks
dropout = 0.2           # Dropout rate
```

## Using Different Data

To train on different text data:

1. Replace the contents of `data.txt` with your desired training text
2. For larger datasets, consider increasing `max_iters` for better convergence
3. The model automatically adapts to the vocabulary size of your text

### Recommended Data Formats

- **Books/Literature**: Plain text novels, poetry, or stories
- **Code**: Programming language files (Python, JavaScript, etc.)
- **Dialogue**: Conversational text or scripts
- **Domain-specific**: Medical texts, legal documents, technical manuals

## Model Loading and Generation

To load a pre-trained model and generate text:

```python
import torch
import pickle

# Load the trained model
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)

# Generate text
prompt = "Your prompt here"
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
generated = model.generate(context.unsqueeze(0), max_new_tokens=200)
output = decode(generated[0].tolist())
print(output)
```

## Performance Tips

### For Better Results:
- **Increase training iterations**: Set `max_iters` to 1000+ for meaningful output
- **Larger datasets**: Use substantial text corpora (>1MB) for better learning
- **Adjust learning rate**: Lower rates (1e-5) for stable convergence with large datasets
- **GPU acceleration**: The model automatically uses CUDA if available

### For Faster Training:
- **Reduce model size**: Decrease `n_embd`, `n_head`, or `n_layer`
- **Smaller batches**: Reduce `batch_size` if running out of memory
- **Shorter context**: Decrease `block_size` for faster processing

## Output Examples

With sufficient training, the model can generate coherent text in the style of your training data:

```
Input: "Hello! Can you see me?"
Output: "Hello! Can you see me? I think I can see you too. The world is full of..."
```

## File Structure

```
├── BPE.py              # Main training script
├── data.txt            # Training data (create this file)
├── model-01.pkl        # Saved model (generated after training)
├── README.md           # This file
├── LICENSE             # Apache 2.0 License
└── .gitignore          # Git ignore patterns
```

## Technical Details

### Character-Level Tokenization
The model uses character-level tokenization, creating a vocabulary from all unique characters in your training data. This approach:
- Handles any text without preprocessing
- Learns subword patterns naturally
- Works well for multiple languages
- Suitable for code generation tasks

### Memory-Mapped File Reading
Efficient data loading using memory mapping allows training on large files without loading everything into RAM.

### Autoregressive Generation
The model generates text one character at a time, using previously generated characters as context for predicting the next character.

## Troubleshooting

### Common Issues:

**"No such file or directory: 'data.txt'"**
- Create a `data.txt` file in the project root with your training text

**CUDA out of memory**
- Reduce `batch_size` or `block_size`
- Use CPU by setting `device = 'cpu'`

**Poor generation quality**
- Increase `max_iters` (try 1000+)
- Ensure sufficient training data (>100KB recommended)
- Check that your data has consistent patterns to learn

**Model not learning**
- Verify `data.txt` contains substantial, varied text
- Try adjusting `learning_rate` (1e-4 to 1e-6)
- Increase model capacity (`n_embd`, `n_layer`)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation.

---

**Note**: This is an educational implementation. For production use, consider using established libraries like Hugging Face Transformers with pre-trained models.
