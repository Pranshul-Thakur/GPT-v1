import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
import mysql.connector

db = mysql.connector.connect(
    host="pranshul",
    user="N3verl1vin",
    password="123456"
)

cursor = db.cursor()

cursor.execute("CREATE DATABASE mydatabase")

parser = argparse.ArgumentParser(description='Placeholder') # Here we add an argument to the parser, specifying the expected type, a help message, etc.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
block_size = 256
max_iters = 2000 #increase to get a meaningful output
learning_rate = 6e-4
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 6
dropout = 0.1

# LoRA parameters
lora_rank = 32
lora_alpha = 64
lora_dropout = 0.05

chars = ""
with open("data.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
        
vocab_size = len(chars)


string_to_int = { ch:i for i,ch in enumerate(chars) }   
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])


def get_random_chunk(split):
    filename = "data.txt" # filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, max(0, file_size - block_size*batch_size))
            mm.seek(start_pos)
            block = mm.read(min(block_size*batch_size-1, file_size - start_pos))
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data

def get_batch(split):
    data = get_random_chunk(split)
    if len(data) <= block_size:
        # Handle case where data is too short
        data = torch.cat([data] * ((block_size + len(data)) // len(data)))
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if rank > 0:
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            self.lora_dropout = nn.Dropout(dropout)
            self.scaling = alpha / rank
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        result = self.linear(x)
        if self.rank > 0:
            lora_result = self.lora_B(self.lora_dropout(self.lora_A(x)))
            result = result + lora_result * self.scaling
        return result


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = LoRALinear(dim, 4 * dim, lora_rank, lora_alpha, lora_dropout, bias=False)
        self.w2 = LoRALinear(4 * dim, dim, lora_rank, lora_alpha, lora_dropout, bias=False)
        self.w3 = LoRALinear(dim, 4 * dim, lora_rank, lora_alpha, lora_dropout, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = LoRALinear(n_embd, head_size, lora_rank, lora_alpha, lora_dropout, bias=False)
        self.query = LoRALinear(n_embd, head_size, lora_rank, lora_alpha, lora_dropout, bias=False)
        self.value = LoRALinear(n_embd, head_size, lora_rank, lora_alpha, lora_dropout, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (self.head_size**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = LoRALinear(head_size * num_heads, n_embd, lora_rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.swiglu = SwiGLU(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.swiglu(x))
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        y = self.sa(self.ln1(x))
        x = x + y
        y = self.ffwd(self.ln2(x))
        x = x + y
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = LoRALinear(n_embd, vocab_size, lora_rank, lora_alpha, lora_dropout, bias=False)
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, LoRALinear)):
            if hasattr(module, 'linear'):
                torch.nn.init.normal_(module.linear.weight, mean=0.0, std=0.02)
                if module.linear.bias is not None:
                    torch.nn.init.zeros_(module.linear.bias)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        
        
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

model = GPTLanguageModel(vocab_size)
# print('loading model parameters...')
# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('loaded successfully!')
m = model.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=learning_rate*0.1)

for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}, lr: {scheduler.get_last_lr()[0]:.6f}")
    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')


prompt = 'Hello! Can you see me?'
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100, temperature=0.8, top_k=40)[0].tolist())
print(generated_chars)
