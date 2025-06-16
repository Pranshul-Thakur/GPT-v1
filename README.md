# GPT Character-Level Text Generator

A simple GPT-style transformer model that generates text character by character. Built for learning how transformers work with some modern improvements like LoRA for efficient training.

## What it does

- Trains on any text file you give it
- Learns to generate similar text
- Uses character-level tokenization (learns letter by letter)
- Includes LoRA for faster training with less memory

## Setup

1. Install requirements:
```bash
pip install torch mysql-connector-python
```

2. Create a `data.txt` file with your training text

3. Run the training:
```bash
python BPE.py
```

## Model Settings

You can change these in `BPE.py`:

```python
batch_size = 64         # How many examples at once
block_size = 256        # Context window size
max_iters = 2000        # Training steps (increase this!)
learning_rate = 6e-4    # How fast it learns
n_embd = 512           # Model size
n_head = 8             # Attention heads
n_layer = 6            # Number of layers
```

## What's special about this model

- **LoRA**: Makes training faster and uses less memory
- **RMSNorm**: Better than regular layer normalization
- **SwiGLU**: Better activation function than ReLU
- **Cosine scheduling**: Learning rate gets smaller over time
- **Gradient clipping**: Prevents training from exploding

## Training Tips

- **For good results**: Set `max_iters` to 5000+ and use lots of training data
- **For faster training**: Reduce `batch_size` or model size if running out of memory
- **GPU recommended**: Will automatically use CUDA if available

## Docker Support

Build and run with Docker:
```bash
docker build -t gpt-model .
docker run gpt-model
```

## Files

- `BPE.py` - Main training script
- `data.txt` - Your training data (you create this)
- `Dockerfile` - For containerization
- `Jenkinsfile` - CI/CD pipeline
- `requirements.txt` - Dependencies

## How to use a trained model

```python
import pickle
import torch

# Load saved model
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)

# Generate text
prompt = "Hello"
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
output = model.generate(context.unsqueeze(0), max_new_tokens=100)
print(decode(output[0].tolist()))
```

## Common Problems

- **"data.txt not found"**: Create the file with your text
- **Out of memory**: Reduce `batch_size` to 32 or 16
- **Bad output**: Increase `max_iters` and use more training data
- **MySQL errors**: You can remove the database code if you don't need it

## Note

This is a learning project to understand how GPT works. The database connection is just for practice - you can remove it if you want.
