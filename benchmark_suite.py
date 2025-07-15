"""
Model Benchmarking Suite for GPT Performance Comparison
Usage: python benchmark_suite.py --model_path model-01.pkl --data_path data.txt
"""

import torch
import torch.nn as nn
from BPE import GPTLanguageModel
from torch.nn import functional as F
import time
import json
import os
import pickle
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import psutil
import random
import mmap

class ModelBenchmark:
    def __init__(self, model_name, vocab_size):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.results = {
            'training_time': [],
            'loss_progression': [],
            'evaluation_metrics': {},
            'memory_usage': [],
            'throughput': [],
            'model_size': {},
            'generation_quality': []
        }
    
    def count_parameters(self, model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count LoRA parameters separately if they exist
        lora_params = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_params += module.lora_A.weight.numel() + module.lora_B.weight.numel()
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'lora_params': lora_params,
            'base_params': total_params - lora_params
        }
    
    def time_training_iterations(self, model, optimizer, get_batch_fn, device, num_iterations=50):
        """Time multiple training iterations"""
        model.train()
        times = []
        losses = []
        
        for i in range(num_iterations):
            start_time = time.time()
            
            # Get batch
            x, y = get_batch_fn('train')
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            end_time = time.time()
            
            times.append(end_time - start_time)
            losses.append(loss.item())
            
            if i % 10 == 0:
                print(f"  Iteration {i}/{num_iterations}, Loss: {loss.item():.4f}, Time: {end_time - start_time:.4f}s")
        
        return {
            'avg_time_per_iteration': np.mean(times),
            'std_time_per_iteration': np.std(times),
            'avg_loss': np.mean(losses),
            'final_loss': losses[-1],
            'all_times': times,
            'all_losses': losses
        }
    
    def measure_memory_usage(self, model, device):
        """Measure GPU/CPU memory usage"""
        memory_info = {}
        
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Measure model size
            dummy_input = torch.randint(0, self.vocab_size, (64, 128)).to(device)


            _ = model(dummy_input)
            
            memory_info = {
                'gpu_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'gpu_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
                'gpu_cached_mb': torch.cuda.memory_cached() / 1024**2
            }
        
        # CPU memory
        process = psutil.Process(os.getpid())
        memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024**2
        
        return memory_info
    
    def evaluate_perplexity(self, model, get_batch_fn, device, num_batches=20):
        """Calculate perplexity on validation set"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(num_batches):
                x, y = get_batch_fn('val')
                logits, loss = model(x, y)
                total_loss += loss.item()
                total_tokens += x.numel()
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))
        return {
            'perplexity': perplexity.item(),
            'avg_loss': avg_loss,
            'total_tokens_evaluated': total_tokens
        }
    
    def measure_generation_speed(self, model, encode_fn, decode_fn, device, prompts, max_tokens=100):
        """Measure text generation speed and quality"""
        model.eval()
        results = []
        
        for prompt in prompts:
            start_time = time.time()
            
            context = torch.tensor(encode_fn(prompt), dtype=torch.long, device=device)
            generated = model.generate(context.unsqueeze(0), max_new_tokens=max_tokens)
            generated_text = decode_fn(generated[0].tolist())
            
            generation_time = time.time() - start_time
            
            results.append({
                'prompt': prompt,
                'generated_length': len(generated_text),
                'generation_time': generation_time,
                'tokens_per_second': max_tokens / generation_time,
                'characters_per_second': len(generated_text) / generation_time
            })
        
        return results
    
    def save_results(self, filepath):
        """Save benchmark results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")


# Baseline Model Implementation (Standard Transformer without LoRA)
class BaselineGPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[BaselineBlock(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, index, targets=None):
        B, T = index.shape
        device = index.device
        
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
            index_cond = index[:, -self.block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index


class BaselineBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = BaselineMultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = BaselineFeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        y = self.sa(self.ln1(x))
        x = x + y
        y = self.ffwd(self.ln2(x))
        x = x + y
        return x


class BaselineMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([BaselineHead(head_size, n_embd, dropout, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class BaselineHead(nn.Module):
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class BaselineFeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


# Data Loading Functions (copied from your original code)
def get_random_chunk(split, filename, block_size, batch_size):
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, max(0, file_size - block_size*batch_size))
            mm.seek(start_pos)
            block = mm.read(min(block_size*batch_size-1, file_size - start_pos))
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            return decoded_block


def create_data_functions(data_path, batch_size=64, block_size=256):
    """Create data loading functions"""
    
    # Read and create character mappings
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
    
    vocab_size = len(chars)
    string_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_string = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [string_to_int[c] for c in s]
    decode = lambda l: ''.join([int_to_string[i] for i in l])
    
    def get_batch(split):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_chunk = get_random_chunk(split, data_path, block_size, batch_size)
        data = torch.tensor(encode(data_chunk), dtype=torch.long)
        
        if len(data) <= block_size:
            data = torch.cat([data] * ((block_size + len(data)) // len(data)))
        
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    
    return get_batch, encode, decode, vocab_size


class ComparisonRunner:
    def __init__(self, data_path, model_path=None):
        self.data_path = data_path
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize data functions
        self.get_batch, self.encode, self.decode, self.vocab_size = create_data_functions(data_path)
        
        print(f"Device: {self.device}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def run_comprehensive_benchmark(self, training_iterations=50, generation_samples=5):
        """Run comprehensive benchmark comparing LoRA vs Baseline"""
        
        print("=" * 60)
        print("COMPREHENSIVE MODEL BENCHMARK")
        print("=" * 60)
        
        # Load or create LoRA model
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading LoRA model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                lora_model = pickle.load(f)
            lora_model = lora_model.to(self.device)
        else:
            print("LoRA model not found. Please train and save your model first.")
            return
        
        # Create baseline model
        print("Creating baseline model...")
        block_size = 128
        baseline_model = BaselineGPTLanguageModel(
            vocab_size=self.vocab_size,
            n_embd=512,
            n_head=8,
            n_layer=6,
            block_size=block_size,
            dropout=0.1
        ).to(self.device)
        
        # Initialize benchmarks
        lora_benchmark = ModelBenchmark("LoRA_Enhanced_GPT", self.vocab_size)
        baseline_benchmark = ModelBenchmark("Baseline_GPT", self.vocab_size)
        
        # 1. Model Size Comparison
        print("\n1. ANALYZING MODEL ARCHITECTURE...")
        lora_params = lora_benchmark.count_parameters(lora_model)
        baseline_params = baseline_benchmark.count_parameters(baseline_model)
        
        print(f"LoRA Model - Total: {lora_params['total_params']:,}, LoRA: {lora_params['lora_params']:,}")
        print(f"Baseline Model - Total: {baseline_params['total_params']:,}")
        
        lora_benchmark.results['model_size'] = lora_params
        baseline_benchmark.results['model_size'] = baseline_params
        
        # 2. Memory Usage Comparison
        print("\n2. MEASURING MEMORY USAGE...")
        lora_memory = lora_benchmark.measure_memory_usage(lora_model, self.device)
        baseline_memory = baseline_benchmark.measure_memory_usage(baseline_model, self.device)
        
        print(f"LoRA Memory Usage: {lora_memory}")
        print(f"Baseline Memory Usage: {baseline_memory}")
        
        lora_benchmark.results['memory_usage'] = lora_memory
        baseline_benchmark.results['memory_usage'] = baseline_memory
        
        # 3. Training Speed Comparison
        print(f"\n3. COMPARING TRAINING SPEED ({training_iterations} iterations)...")
        
        # LoRA model training
        lora_optimizer = torch.optim.AdamW(lora_model.parameters(), lr=6e-4)
        print("Testing LoRA model training speed...")
        lora_training_results = lora_benchmark.time_training_iterations(
            lora_model, lora_optimizer, self.get_batch, self.device, training_iterations
        )
        
        # Baseline model training
        baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=6e-4)
        print("Testing baseline model training speed...")
        baseline_training_results = baseline_benchmark.time_training_iterations(
            baseline_model, baseline_optimizer, self.get_batch, self.device, training_iterations
        )
        
        lora_benchmark.results['training_performance'] = lora_training_results
        baseline_benchmark.results['training_performance'] = baseline_training_results
        
        # 4. Evaluation Metrics
        print("\n4. EVALUATING MODEL PERFORMANCE...")
        
        lora_eval = lora_benchmark.evaluate_perplexity(lora_model, self.get_batch, self.device)
        baseline_eval = baseline_benchmark.evaluate_perplexity(baseline_model, self.get_batch, self.device)
        
        print(f"LoRA Perplexity: {lora_eval['perplexity']:.2f}")
        print(f"Baseline Perplexity: {baseline_eval['perplexity']:.2f}")
        
        lora_benchmark.results['evaluation_metrics'] = lora_eval
        baseline_benchmark.results['evaluation_metrics'] = baseline_eval
        
        # 5. Generation Speed Test
        print(f"\n5. TESTING GENERATION SPEED ({generation_samples} samples)...")
        
        test_prompts = [
            "The future of artificial intelligence",
            "In a world where technology",
            "Machine learning algorithms",
            "Deep neural networks",
            "Natural language processing"
        ][:generation_samples]
        
        lora_generation = lora_benchmark.measure_generation_speed(
            lora_model, self.encode, self.decode, self.device, test_prompts
        )
        baseline_generation = baseline_benchmark.measure_generation_speed(
            baseline_model, self.encode, self.decode, self.device, test_prompts
        )
        
        lora_benchmark.results['generation_quality'] = lora_generation
        baseline_benchmark.results['generation_quality'] = baseline_generation
        
        # 6. Calculate Performance Improvements
        print("\n6. CALCULATING PERFORMANCE IMPROVEMENTS...")
        
        # Training speed improvement
        lora_avg_time = lora_training_results['avg_time_per_iteration']
        baseline_avg_time = baseline_training_results['avg_time_per_iteration']
        training_speedup = ((baseline_avg_time - lora_avg_time) / baseline_avg_time) * 100
        
        # Memory efficiency
        if self.device == 'cuda':
            lora_mem = lora_memory.get('gpu_allocated_mb', 0)
            baseline_mem = baseline_memory.get('gpu_allocated_mb', 0)
            memory_reduction = ((baseline_mem - lora_mem) / baseline_mem) * 100 if baseline_mem > 0 else 0
        else:
            memory_reduction = 0
        
        # Parameter efficiency
        param_reduction = ((baseline_params['total_params'] - lora_params['lora_params']) / baseline_params['total_params']) * 100
        
        # Generation speed
        lora_gen_speed = np.mean([g['tokens_per_second'] for g in lora_generation])
        baseline_gen_speed = np.mean([g['tokens_per_second'] for g in baseline_generation])
        generation_speedup = ((lora_gen_speed - baseline_gen_speed) / baseline_gen_speed) * 100
        
        # 7. Print Summary Report
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY REPORT")
        print("="*60)
        
        print(f"Training Speed: {training_speedup:+.1f}% ({'faster' if training_speedup > 0 else 'slower'})")
        print(f"Memory Usage: {memory_reduction:+.1f}% ({'reduced' if memory_reduction > 0 else 'increased'})")
        print(f"Parameter Efficiency: {param_reduction:+.1f}% fewer parameters to train")
        print(f"Generation Speed: {generation_speedup:+.1f}% ({'faster' if generation_speedup > 0 else 'slower'})")
        
        print(f"\nLoRA Model Perplexity: {lora_eval['perplexity']:.2f}")
        print(f"Baseline Model Perplexity: {baseline_eval['perplexity']:.2f}")
        
        # 8. Save Results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lora_benchmark.save_results(f"lora_benchmark_{timestamp}.json")
        baseline_benchmark.save_results(f"baseline_benchmark_{timestamp}.json")
        
        # Save comparison summary
        summary = {
            'timestamp': timestamp,
            'training_speedup_percent': training_speedup,
            'memory_reduction_percent': memory_reduction,
            'parameter_efficiency_percent': param_reduction,
            'generation_speedup_percent': generation_speedup,
            'lora_perplexity': lora_eval['perplexity'],
            'baseline_perplexity': baseline_eval['perplexity'],
            'device': self.device,
            'vocab_size': self.vocab_size
        }
        
        with open(f"comparison_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDetailed results saved to:")
        print(f"- lora_benchmark_{timestamp}.json")
        print(f"- baseline_benchmark_{timestamp}.json")
        print(f"- comparison_summary_{timestamp}.json")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Benchmark GPT Model Performance')
    parser.add_argument('--data_path', type=str, default='data.txt', help='Path to training data')
    parser.add_argument('--model_path', type=str, default='model-01.pkl', help='Path to trained LoRA model')
    parser.add_argument('--training_iterations', type=int, default=50, help='Number of training iterations to benchmark')
    parser.add_argument('--generation_samples', type=int, default=5, help='Number of generation samples to test')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.data_path):
        print(f"Error: Data file '{args.data_path}' not found!")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        print("Please train your model first using BPE.py")
        return
    
    # Run benchmark
    runner = ComparisonRunner(args.data_path, args.model_path)
    summary = runner.run_comprehensive_benchmark(
        training_iterations=args.training_iterations,
        generation_samples=args.generation_samples
    )


if __name__ == "__main__":
    # Manually set arguments
    class Args:
        data_path = 'data.txt'
        model_path = 'model-01.pkl'
        training_iterations = 50
        generation_samples = 5

    args = Args()

    # Run benchmark
    runner = ComparisonRunner(args.data_path, args.model_path)
    summary = runner.run_comprehensive_benchmark(
        training_iterations=args.training_iterations,
        generation_samples=args.generation_samples
    )

    if summary:
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Training Speed Improvement: {summary['training_speedup_percent']:+.1f}%")
        print(f"Memory Efficiency: {summary['memory_reduction_percent']:+.1f}%")
        print(f"Parameter Efficiency: {summary['parameter_efficiency_percent']:+.1f}%")
        print(f"Generation Speed: {summary['generation_speedup_percent']:+.1f}%")
