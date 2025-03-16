#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from transformers import AutoTokenizer
import time
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func



#%%
device = 'cpu'  # for now
# if torch.cuda.is_available() else 'cpu'



#%%
parser = argparse.ArgumentParser(description='This is a demonstration program')
    
    

#%%
block_size = 8 # input tokens
batch_size = 4 # training samples
max_iterations = 1000 # training steps
learning_rate = 3e-4 # step size updation at rate of 0.0003
model_evaluation_iterations = 250
embedded_dim = 256
parallel_head = 4
no_layer = 4 # 
dropout = 0.2 # to prevent overfitting



#%%
stop_words = set(stopwords.words('english')) # 



#%%
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # loading uncased bert tokenizer
vocab_size = len(tokenizer)



#%%
encode = lambda s: tokenizer.encode(s, add_special_tokens=True) # encoding and decoding func
decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)



#%%
def preprocess_text(text):
    text = text.lower().strip()  # Convert text to lowercase and remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text



#%%
def load_half_dataset_into_memory(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(0, 2) # moving file pointer to end to find the end location
        half_point = f.tell() // 2 # determining half point
        f.seek(0) # moving file pointer back to start
        data = f.read(half_point) 
    return preprocess_text(data)  # Apply preprocessing before returning data



#%%
def get_random_chunk(split):
    filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)
            if start_pos > 0:
                mm.seek(start_pos - 1)
                while mm.read(1) != b"\n" and mm.tell() < file_size:
                    pass
                start_pos = mm.tell()
            end_pos = start_pos + block_size * batch_size
            if end_pos > file_size:
                start_pos = max(0, file_size - block_size * batch_size)
                mm.seek(start_pos)
            block = mm.read(block_size * batch_size)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '').strip() # data normalization and handling missing values and corrupt data
            if not decoded_block:
                print("Warning: Encountered empty chunk, retrying...")
                return get_random_chunk(split)
            processed_block = preprocess_text(decoded_block)
            data = torch.tensor(encode(processed_block), dtype=torch.long)
    return data



#%%
def get_batch(split):
    valid_batch = False
    while not valid_batch:
        data = get_random_chunk(split)
        if data.size(0) > block_size:
            try:
                ix = torch.randint(0, data.size(0) - block_size, (batch_size,))
                x = torch.stack([data[i:i+block_size] for i in ix])
                y = torch.stack([data[i+1:i+block_size+1] for i in ix])
                valid_batch = True
            except RuntimeError as e:
                print(f"Error encountered: {e}. Trying to fetch a new chunk...")
        else:
            print("Fetched chunk is too small, fetching a new chunk...")
    x, y = x.to(device), y.to(device)
    return x, y