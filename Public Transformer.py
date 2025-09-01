import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import re
import numpy as np
import os
import time as tm
import keyboard
import threading
from datetime import datetime
import json
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datasets import load_dataset #
import random #
rlhf_buffer = {'prompt': None, 'response': None}
# Load dataset (already present in your script)
# Remove the Hashtag to use one of these public datasets
ds = "" 
# ds = load_dataset("HelpingAI/Intermediate-Thinking-130k") #
# ds = load_dataset("Anthropic/hh-rlhf",)

human_training_data = []
# A function to save the data to a file
def save_human_training_data(data, path="human_training_data.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Human training data saved to {path}")

# A function to load the data
def load_human_training_data(path="human_training_data.json"):
    if os.path.exists(path):
        with open(path, "r",errors="ignore") as f:
            data = json.load(f)
        print(f"Human training data loaded from {path}")
        return data
    return []
human_training_data = load_human_training_data()  # Load existing data if available
# A function to save the data to a file

# Extract text samples (combine 'question' and 'chosen' fields)

new_samples = [] #
for split in ds: #
    for item in ds[split]: #
        if 'question' in item and 'chosen' in item: #
            # Combine question and chosen answer as a single string, separated by tab
            new_samples.append(f"{item['question']}\t{item['chosen']}") #
print(new_samples[:2]) #

def reward_weighted_loss(output, target, reward): #
    loss_fn = torch.nn.CrossEntropyLoss() #
    loss = loss_fn(output.view(-1, output.size(-1)), target.view(-1)) #
    return loss * reward #

def fine_tune_from_rewards(model, optimizer, reward_buffer, tokenizer, config): #
    if not reward_buffer: #
        return #

    model.train() #
    for input_ids, target_ids, reward in reward_buffer: #
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(config.device) #
        target_tensor = torch.tensor([target_ids], dtype=torch.long).to(config.device) #

        output = model(input_tensor) #
        loss = reward_weighted_loss(output, target_tensor, reward) #
        
        optimizer.zero_grad() #
        loss.backward() #
        optimizer.step() #

    print(f"📚 Fine-tuned on {len(reward_buffer)} reward samples.") #
    reward_buffer.clear() #




# === Custom Formatting ===
class BString(str):
    def bold(self):
        return f"\033[1m{self}\033[0m"  # ANSI escape code for bold text

    def underline(self):
        return f"\033[4m{self}\033[0m"  # ANSI escape code for underlined text

    def italic(self):
        return f"\033[3m{self}\033[0m"  # ANSI escape code for italic text

    def red(self):
        return f"\033[31m{self}\033[0m"  # ANSI escape code for red text
# Change the parameters to see what fits your needs feel free to ask me what does what 
# Labeling by chatgpt cuz i was to lazy

class Config:
    Username = "New User"
    vocab_size = 30000              # ✅ Set dynamically after tokenizer is built
    block_size = 512               # 🧠 Suggest 256–512 for richer context understanding
    n_layers = 8                  # 📏 Scales well for 9k vocab size and decent generation
    n_heads = 16                   # ⚖️ Equal to d_model // 64 is a good rule
    d_model = 256                  # 🧠 Standard BERT/GPT-2 small config — more tokens, more rep
    d_ff = 2048                    # 💡 Typically 4x d_model
    dropout = 0.1                 # ✅ Lower dropout helps when training stability is good
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
user = Config.Username #
user = BString(user)  # Convert to BString for formatting

special_tokens = [t.lower() for t in ['Maverick','<PAD>','<PROMPT>','</PROMPT>','<THOUGHT>','</THOUGHT>',':',';','!','Hi','Hello', '<EOS>', '<TEXT>', '</TEXT>', '<UNK>', '*', '**', '<user>']] #
word_set = set() #
training_data = new_samples #
vocab = list(dict.fromkeys(special_tokens + sorted(word_set))) #

def load_random_training_samples(lines, tokenizer, count=50): #
    import random #
    subset = random.sample(lines, k=min(count, len(lines))) #
    processed = [preprocess(l) for l in subset] #
    encoded = [tokenizer.encode(p) for p in processed] #
    return create_input_target_pairs(encoded) #

def preprocess(text): #
    # Replace tags
    text = text.lower() #
    text = text.replace("<text>", "<TEXT>") #
    text = text.replace("</text>", "</TEXT>") #
    text = text.replace("<user>", user.bold()) #
    return text.lower() #
def resize_model_if_needed(model, vocab_size, device):
    old_vocab_size = model.token_embedding.num_embeddings
    if vocab_size > old_vocab_size:
        print(f"✨ Vocabulary grew: Resizing model from {old_vocab_size} → {vocab_size}")

        # Resize embeddings
        embedding_dim = model.token_embedding.embedding_dim
        old_embed_weight = model.token_embedding.weight.data.clone()
        model.token_embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        model.token_embedding.weight.data[:old_vocab_size] = old_embed_weight

        # Resize output head
        old_output_weight = model.head.weight.data.clone()
        old_output_bias = model.head.bias.data.clone()
        model.head = nn.Linear(embedding_dim, vocab_size).to(device)
        model.head.weight.data[:old_vocab_size] = old_output_weight
        model.head.bias.data[:old_vocab_size] = old_output_bias

        print("✅ Model resized successfully.")



class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = list(vocab)  # Ensure it's mutable
        self.token_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_token = {i: w for i, w in enumerate(self.vocab)}
        self.pad_token_id = self.token_to_id.get('<pad>', -1)
        self.eos_token_id = self.token_to_id.get('<eos>', -1)
        self.unk_token_id = self.token_to_id.get('<unk>', -1)
        
        # Add these properties for compatibility with training function
        self.pad_token = '<pad>' if '<pad>' in self.token_to_id else None
        self.eos_token = '<eos>' if '<eos>' in self.token_to_id else None
        self.unk_token = '<unk>' if '<unk>' in self.token_to_id else None
    
    def __len__(self):
        return len(self.vocab)

    def encode(self, text, add_special_tokens=True):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens (for compatibility)
        """
        tokens = re.findall(r'<[^>]+>|[\w]+|[^\s\w]', text.lower())
        ids = []
        
        for token in tokens:
            if token in self.token_to_id:
                # Token exists in vocabulary
                ids.append(self.token_to_id[token])
            else:
                # Dynamically add new token
                new_id = len(self.vocab)
                self.vocab.append(token)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token
                print(f"[+] Added new token to vocab: '{token}' → {new_id}")
                ids.append(new_id)  # Use the NEW token ID, not eos_token_id!
        
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens like <pad>
        """
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '<unk>')
            
            if skip_special_tokens and token in ['<pad>', '<eos>', '<bos>']:
                continue
                
            tokens.append(token)
        
        return ' '.join(tokens)

    def get_vocab(self):
        """Return the vocabulary as a dictionary for compatibility."""
        return self.token_to_id

    def add_special_tokens(self, special_tokens_dict):
        """
        Add special tokens to the vocabulary.
        
        Args:
            special_tokens_dict: Dict like {'eos_token': '<eos>', 'pad_token': '<pad>'}
        
        Returns:
            Number of tokens added
        """
        added_count = 0
        
        for token_type, token_value in special_tokens_dict.items():
            if token_value not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab.append(token_value)
                self.token_to_id[token_value] = new_id
                self.id_to_token[new_id] = token_value
                added_count += 1
                print(f"[+] Added special token: '{token_value}' → {new_id}")
            
            # Set the token attribute
            setattr(self, token_type, token_value)
            setattr(self, f"{token_type}_id", self.token_to_id[token_value])
        
        return added_count

    def save_vocab(self, path):
        """Save vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            for token in self.vocab:
                f.write(token + '\n')
        print(f"Vocabulary saved to {path} ({len(self.vocab)} tokens)")

    def load_vocab(self, path):
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f]
        
        self.token_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_token = {i: w for i, w in enumerate(self.vocab)}
        
        # Update special token IDs
        self.pad_token_id = self.token_to_id.get('<pad>', -1)
        self.eos_token_id = self.token_to_id.get('<eos>', -1)
        self.unk_token_id = self.token_to_id.get('<unk>', -1)
        
        print(f"Vocabulary loaded from {path} ({len(self.vocab)} tokens)")

    def get_vocab_size(self):
        """Get the current vocabulary size."""
        return len(self.vocab)

    def token_to_string(self, token_id):
        """Convert single token ID to string."""
        return self.id_to_token.get(token_id, '<unk>')

    def string_to_token(self, token_str):
        """Convert string to token ID."""
        return self.token_to_id.get(token_str, self.unk_token_id)

def create_input_target_pairs(data): #
    pairs = [] #
    for seq in data: #
        input_seq = seq[:-1] #
        target_seq = seq[1:] #
        pairs.append((input_seq, target_seq)) #
    return pairs #

config = Config() #

print("Vocab size:", len(vocab)) #
print("Vocab:", vocab) #
# Dynamic vocab size setting (added as per previous analysis)
config.vocab_size = len(vocab) 
print("Config Vocab size set to:", config.vocab_size)
tokenizer = SimpleTokenizer(vocab) #
encoded_data = [tokenizer.encode(preprocess(line)) for line in new_samples] #
tokenized_data = create_input_target_pairs(encoded_data) #

class MemoryBuffer: #
    def __init__(self): #
        self.buffer = [] #

    def add(self, text): #
        self.buffer.append(text) #

    def get_recent(self, n=5): #
        return self.buffer[-n:] #

    def get_all(self): #
        return self.buffer  #

class PositionalEncoding(nn.Module): #
    def __init__(self, d_model, max_len=9000): #
        super().__init__() #
        self.d_model = d_model #
        self.max_len = max_len #

        pe = torch.zeros(max_len, d_model) #
        position = torch.arange(0, max_len).unsqueeze(1) #
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) #
        pe[:, 0::2] = torch.sin(position * div_term) #
        pe[:, 1::2] = torch.cos(position * div_term) #
        self.register_buffer('pe', pe.unsqueeze(0)) #

    def forward(self, x): #
        seq_len = x.size(1) #
        if seq_len > self.max_len: #
            raise ValueError(f"Input sequence length {seq_len} exceeds PositionalEncoding max_len {self.max_len}") #
        return x + self.pe[:, :seq_len].to(x.device) #

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_k).transpose(1, 2) for t in qkv]
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Create causal mask - lower triangular matrix
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        
        # Apply causal mask first
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Apply padding mask if provided
        if mask is not None:
            # Convert padding mask to attention mask
            # mask should be [B, T] where True = valid token, False = padding
            if mask.dim() == 2:
                # Create attention mask: [B, T] -> [B, 1, 1, T]
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
                # For each query position, we can attend to all valid key positions
                # So we need to broadcast this to [B, n_heads, T, T]
                key_mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
                key_mask = key_mask.expand(B, self.n_heads, T, T)
                scores = scores.masked_fill(~key_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

class FeedForward(nn.Module): #
    def __init__(self, d_model, d_ff): #
        super().__init__() #
        self.net = nn.Sequential( #
            nn.Linear(d_model, d_ff), #
            nn.ReLU(), #
            nn.Linear(d_ff, d_model), #
            nn.Dropout(config.dropout) #
        )

    def forward(self, x): #
        return self.net(x) #

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config.d_model, config.n_heads)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, config.block_size)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.embedding_dim = d_model

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        x = F.dropout(x, p=config.dropout, training=self.training)
        return self.head(x)

def save_model(model, tokenizer, config, path):
    """
    Saves the model, its configuration, and the tokenizer's vocabulary.
    """
    # Save tokenizer's vocabulary separately, which is more robust
    vocab_path = os.path.splitext(path)[0] + "_vocab.json"
    with open(vocab_path, "w", encoding='utf-8') as f:
        json.dump(tokenizer.vocab, f)
        
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': config.vocab_size,
            'block_size': config.block_size,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'd_model': config.d_model,
            'd_ff': config.d_ff,
            'dropout': config.dropout,
        }
    }, path)
    print(f"✅ Model saved to {path} and vocabulary to {vocab_path}")

def load_model(path, device):
    """
    Loads a model and intelligently resizes it if the vocabulary has grown.
    """
    vocab_path = os.path.splitext(path)[0] + "_vocab.json"

    if not os.path.exists(path) or not os.path.exists(vocab_path):
        return None, None, None # Return None if files don't exist

    # 1. Load the saved vocabulary and create the tokenizer
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab)

    # 2. Load the checkpoint and the saved config
    checkpoint = torch.load(path, map_location=device)
    saved_config_dict = checkpoint['config']
    
    # Create a config object from the loaded dictionary
    config = Config()
    for key, value in saved_config_dict.items():
        setattr(config, key, value)
    
    # Update vocab_size to the new size from the loaded tokenizer
    config.vocab_size = len(tokenizer.vocab)

    # 3. Initialize a new model with the potentially larger vocabulary size
    model = Transformer(vocab_size=config.vocab_size, d_model=config.d_model).to(device)
    
    # 4. Intelligently copy weights from the old state_dict to the new model
    old_state_dict = checkpoint['model_state_dict']
    new_state_dict = model.state_dict()
    
    # FIX: Removed the incorrect nested loop and cleaned up the logic.
    for key in new_state_dict.keys():
        # If the key exists in the old model and shapes match, copy it directly.
        if key in old_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
            new_state_dict[key] = old_state_dict[key]
        # If the key is for a layer that needs resizing (due to vocab change)...
        elif key in ['token_embedding.weight', 'head.weight', 'head.bias']:
            old_tensor = old_state_dict[key]
            old_vocab_size = old_tensor.shape[0]
            
            print(f"Resizing layer '{key}' from {old_vocab_size} to {config.vocab_size}")
            
            # For 2D Tensors (weights)
            if old_tensor.dim() > 1:
                new_state_dict[key][:old_vocab_size, :] = old_tensor
            # For 1D Tensors (bias)
            else:
                new_state_dict[key][:old_vocab_size] = old_tensor
            
    # Load the newly constructed state dictionary into the model
    model.load_state_dict(new_state_dict)
    model.eval()

    print(f"✅ Model loaded from {path} with updated vocabulary.")
    return model, tokenizer, config


def apply_repetition_penalty(logits, generated, penalty=1.2): #
    for token_id in set(generated[0].tolist()): #
        logits[0, token_id] /= penalty #
    return logits #

def top_k_logits(logits, k): #
    values, _ = torch.topk(logits, k) #
    min_values = values[:, -1].unsqueeze(-1) #
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits) #

@torch.no_grad() #
def generate(model, input_ids, tokenizer, max_new_tokens=50, temperature=0.5, top_k=50, repetition_penalty=1.2, return_only_generated_text=True): 
    """
    Generates text from the model with a prompt attached.
    The input_ids tensor is the prompt.
    
    Args:
        model (Maverick): The model to use for generation.
        input_ids (torch.Tensor): The input tensor containing the prompt.
        tokenizer (SimpleTokenizer): The tokenizer for encoding/decoding.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): The temperature for sampling.
        top_k (int): The number of top-k tokens to consider for sampling.
        repetition_penalty (float): The penalty for repeating tokens.
        return_only_generated_text (bool): If True, returns only the new tokens.

    Returns:
        torch.Tensor: The generated token IDs.
    """
    model.eval()
    generated = input_ids.clone()
    prompt_length = generated.size(1)

    UNK_INDEX = tokenizer.token_to_id.get('<unk>', -1)
    EOS_INDEX = tokenizer.token_to_id.get('<eos>', -1)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Suppress <unk>
            if UNK_INDEX != -1:
                next_token_logits[:, UNK_INDEX] = -float("inf")

            # Apply repetition penalty
            # NOTE: Your original code calls a function that isn't defined.
            # I've left the call here and provided a placeholder function.
            # You will need to implement the actual logic for this.
            next_token_logits = apply_repetition_penalty(next_token_logits, generated, repetition_penalty)

            # Top-k filtering
            # NOTE: Your original code calls a function that isn't defined.
            # I've provided a simple implementation for it here.
            next_token_logits = top_k_logits(next_token_logits, k=top_k)

            # Sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append token
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS is reached
            if EOS_INDEX != -1 and next_token.item() == EOS_INDEX:
                break

    if return_only_generated_text:
        return generated[:, prompt_length:]
    else:
        return generated

def stream_generate(model, input_ids, tokenizer, temperature=0.9, top_k=50, max_new_tokens=50, repetition_penalty=1.2):
    model.eval()
    generated = input_ids.clone()
    prompt_length = generated.size(1)
    
    UNK_INDEX = tokenizer.token_to_id.get('<unk>', -1)
    EOS_INDEX = tokenizer.token_to_id.get('<eos>', -1)

    print("Maverick: ", end='', flush=True)

    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            if UNK_INDEX != -1:
                next_token_logits[:, UNK_INDEX] = -float("inf")
            if step < 5 and EOS_INDEX != -1:
                next_token_logits[:, EOS_INDEX] = -float('inf')
            # Apply repetition penalty
            for token_id in set(generated[0].tolist()):
                next_token_logits[0, token_id] /= repetition_penalty if generated[0, -1] == token_id else 1.0


            # Apply top-k filtering
            values, _ = torch.topk(next_token_logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            next_token_logits = torch.where(next_token_logits < min_value, torch.full_like(next_token_logits, float('-inf')), next_token_logits)
            # print(f"Possible next tokens: {tokenizer.decode(next_token_logits[0].topk(top_k)[1].tolist())}")
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            token_str = tokenizer.decode([token_id])
            token_str = preprocess(token_str).strip()
            print(token_str, end=' ', flush=True)

            generated = torch.cat([generated, next_token], dim=1)

            if token_id == EOS_INDEX:
                break
            tm.sleep(0.1)  # Optional: slow down output for readability
    print()  # newline after full output
    return generated

def resize_token_embeddings(model, new_vocab_size, device=None):
    """
    Safely resize token embeddings and output projection head.
    Works for both vocab expansion and shrinkage.
    Preserves old weights, initializes new ones.
    """
    if device is None:
        device = next(model.parameters()).device

    # --- Input embeddings ---
    old_emb = model.token_emb.weight.data
    old_vocab_size, emb_dim = old_emb.shape
    new_emb = nn.Embedding(new_vocab_size, emb_dim).to(device)

    # Copy over existing weights
    num_to_copy = min(old_vocab_size, new_vocab_size)
    new_emb.weight.data[:num_to_copy] = old_emb[:num_to_copy]
    model.token_emb = new_emb

    # --- Output projection (LM head) ---
    old_head = model.head.weight.data
    new_head = nn.Linear(emb_dim, new_vocab_size, bias=False).to(device)
    new_head.weight.data[:num_to_copy] = old_head[:num_to_copy]
    model.head = new_head

    print(f"[INFO] Vocab resized: {old_vocab_size} → {new_vocab_size} tokens.")
    return model


def update_vocab_and_resize(model, tokenizer, device=None):
    """
    Updates model embedding layers after tokenizer vocab change.
    """
    new_vocab_size = len(tokenizer)
    model = resize_token_embeddings(model, new_vocab_size, device)
    return model

def extract_all_between_tags(text, start_tag, end_tag):
    """Extracts all occurrences between tags."""
    pattern = f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
    return [m.strip() for m in re.findall(pattern, text, re.DOTALL)]

def load_cst_dataset(file_path):
    """
    Load training data from .json or .txt files.
    
    Supports:
    - JSON: array of strings, or dicts with keys like 'prompt', 'thought', 'response'.
    - TXT: one sample per line.
    
    Can also accept a list of file paths.
    Returns: list of (prompt, response) tuples.
    """
    if isinstance(file_path, (list, tuple)):
        all_data = []
        for path in file_path:
            all_data.extend(load_cst_dataset(path))
        return all_data

    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            samples = []
            if isinstance(data, dict):
                for entry in data.values():
                    if isinstance(entry, dict):
                        prompt = str(entry.get("prompt", "")).strip()
                        response = str(entry.get("response", "")).strip()
                        if prompt and response:
                            samples.append((prompt, response))
                    elif isinstance(entry, str) and entry.strip():
                        # Fallback if dict entry is a string
                        samples.append((entry.strip(), entry.strip()))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        prompt = str(item.get("prompt", "")).strip()
                        response = str(item.get("response", "")).strip()
                        if prompt and response:
                            samples.append((prompt, response))
                    elif isinstance(item, str) and item.strip():
                        samples.append((item.strip(), item.strip()))
            return samples

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            # Treat each line as (prompt, response) pair for now (can be same)
            return [(line, line) for line in lines]

    else:
        raise ValueError("Unsupported file type. Use .json or .txt")

def format_training_sample(prompt, response, thought=None):
    """Formats a sample with prompt, free thought, and response for Maverick."""
    formatted_prompt = f"<PROMPT> {prompt.strip()} </PROMPT>"
    formatted_thought = f"<THOUGHT> {thought.strip() if thought else ''} </THOUGHT>"
    formatted_response = f"<TEXT> {response.strip()} </TEXT> <EOS>"
    return f"{formatted_prompt} {formatted_thought} {formatted_response}"

def train_uni_full_sequence(model, tokenizer, config, data=None, dataset_name=None, dataset_subset="train", num_samples=None, epochs=10, batch_size=2, lr=3e-4, min_lr=5e-6, save_file=None):
    """
    Universal training function that works with any dataset format and adds <eos> token to every entry.
    Can load datasets directly from Hugging Face or use provided data.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer (should have eos_token defined)
        config: Training configuration with device and block_size
        data: Dataset in any format (list of strings, dicts, tuples, etc.) - optional if dataset_name provided
        dataset_name: Hugging Face dataset name (e.g., "Anthropic/hh-rlhf", "openai/webgpt_comparisons")
        dataset_subset: Dataset subset/split to use ("train", "test", "validation")
        num_samples: Number of samples to load from dataset (None for all)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate (note: optimizer uses 1e-6 regardless)
        min_lr: Minimum learning rate for scheduler
        save_file: Path to save model checkpoints
    """
    # Load dataset if needed
    if data is None and dataset_name is not None:
        print(f"Loading dataset: {dataset_name} ({dataset_subset})...")
        try:
            from datasets import load_dataset
            import random
            
            dataset = load_dataset(dataset_name, split=dataset_subset)
            
            if num_samples and num_samples < len(dataset):
                # Sample random subset
                indices = random.sample(range(len(dataset)), num_samples)
                dataset = dataset.select(indices)
                print(f"Selected {num_samples} random samples from {len(dataset)} total")
            
            # Convert dataset to list for processing
            data = list(dataset)
            print(f"Loaded {len(data)} samples from {dataset_name}")
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            print("Please provide data directly or check dataset name.")
            return model
    
    elif data is None:
        print("Error: Either 'data' or 'dataset_name' must be provided.")
        return model
    
    print(f"Training on {len(data)} samples...")
    
    level = 300
    upper_level = 3
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-8)
    total_steps = (len(data) // batch_size) * epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, eta_min=2e-8)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.065, ignore_index=tokenizer.pad_token_id)
    
    # Ensure tokenizer has eos_token
    if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None or tokenizer.eos_token == '':
        print("Warning: tokenizer.eos_token not found. Using '<eos>' as default EOS token.")
        tokenizer.eos_token = '<eos>'
        if tokenizer.eos_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'eos_token': tokenizer.eos_token})
            print(f"Added '{tokenizer.eos_token}' to tokenizer vocabulary.")
    else:
        print(f"Using existing EOS token: '{tokenizer.eos_token}'")
    
    model.to(config.device)
    model.train()

    print("Starting universal batched training with EOS tokens...")
    print(f"EOS token: '{tokenizer.eos_token}'")
    global_step = 0
    
    def process_sample_to_text(sample):
        """Convert any sample format to a single text string."""
        if isinstance(sample, str):
            # Handle tab-separated format
            if '\t' in sample:
                parts = sample.split('\t')
                if len(parts) >= 3:
                    prompt, thought, response = parts[0], parts[1], parts[2]
                    return f"<PROMPT> {prompt.lower()} </PROMPT> <THOUGHT> {thought.lower()} </THOUGHT> {response.lower()} <EOS>"
                else:
                    return sample.lower()
            else:
                return sample.lower()
                
        elif isinstance(sample, dict):
            # Handle different dataset formats
            
            # HH-RLHF format
            if 'chosen' in sample and 'rejected' in sample:
                chosen_text = sample['chosen']
                # Parse conversation format
                if 'Human:' in chosen_text and 'Assistant:' in chosen_text:
                    parts = chosen_text.split('Assistant:')
                    if len(parts) >= 2:
                        human_part = parts[0].replace('Human:', '').strip()
                        assistant_part = parts[1].strip()
                        return f"<PROMPT> {human_part.lower()} </PROMPT> {assistant_part.lower()} <EOS>"
                return chosen_text.lower()
            
            # Standard dictionary format
            memory = sample.get('Memory', sample.get('memory', ''))
            prompt = sample.get('prompt', sample.get('question', sample.get('input', sample.get('text', ''))))
            thought = sample.get('thought', sample.get('reasoning', ''))
            response = sample.get('response', sample.get('answer', sample.get('output', '')))
            
            # Handle cases where the entire conversation is in one field
            if not prompt and not response:
                # Try common single-field formats
                full_text = (sample.get('text') or sample.get('conversation') or 
                           sample.get('dialogue') or str(sample))
                
                # Check if it's a conversation format
                if 'Human:' in full_text and 'Assistant:' in full_text:
                    parts = full_text.split('Assistant:')
                    if len(parts) >= 2:
                        human_part = parts[0].replace('Human:', '').strip()
                        assistant_part = parts[1].strip()
                        return f"<PROMPT> {human_part.lower()} </PROMPT> {assistant_part.lower()} <EOS>"
                
                return full_text.lower()
            
            # Build text based on available fields
            text_parts = []
            if memory:
                text_parts.append(f"<PREV> {memory.lower()} </PREV>")
            if prompt:
                text_parts.append(f"<PROMPT> {prompt.lower()} </PROMPT>")
            if thought:
                text_parts.append(f"<THOUGHT> {thought.lower()} </THOUGHT>")
            if response:
                text_parts.append(response.lower())
                
            return ' '.join(text_parts) if text_parts else str(sample).lower()
            
        elif isinstance(sample, (list, tuple)):
            # Handle list/tuple format
            if len(sample) >= 2:
                if len(sample) == 2:
                    # Assume [input, output] format
                    return f"<PROMPT> {str(sample[0]).lower()} </PROMPT> {str(sample[1]).lower()}"
                else:
                    # Assume [prompt, thought, response] or similar
                    prompt = str(sample[0]).lower()
                    thought = str(sample[1]).lower() if len(sample) > 1 else ""
                    response = str(sample[2]).lower() if len(sample) > 2 else ""
                    
                    if thought:
                        return f"<PROMPT> {prompt} </PROMPT> <THOUGHT> {thought} </THOUGHT> {response}"
                    else:
                        return f"<PROMPT> {prompt} </PROMPT> {response}"
            else:
                return str(sample[0]).lower() if sample else ""
        else:
            # Fallback: convert to string
            return str(sample).lower()
    
    for epoch in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            try:
                batch_samples = data[i:i + batch_size]
                
                # Check model resizing
                if hasattr(model, 'token_embedding') and model.token_embedding.num_embeddings != len(tokenizer):
                    print("✨ Vocabulary has grown. Resizing model...")
                    resize_model_if_needed(model, len(tokenizer), config.device)
                    print("✅ Model resized successfully.")
                    if save_file:
                        save_model(model, tokenizer, config, save_file)
                
                input_texts = []
                for sample in batch_samples:
                    try:
                        # Convert sample to text
                        text = process_sample_to_text(sample)
                        
                        # Add EOS token if not already present
                        if text and not text.endswith(tokenizer.eos_token):
                            text = text + tokenizer.eos_token
                            
                        if text:  # Only add non-empty texts
                            input_texts.append(text)
                            
                    except Exception as e:
                        print(f"Error processing sample: {e}")
                        continue

                if not input_texts:
                    print("No valid texts in batch, skipping...")
                    continue

                # Tokenize and pad batch
                encoded_batch = []
                for text in input_texts:
                    try:
                        encoded = tokenizer.encode(text, add_special_tokens=False)
                        if encoded:  # Only add non-empty encodings
                            encoded_batch.append(encoded)
                    except Exception as e:
                        print(f"Error encoding text: {e}")
                        continue
                
                if not encoded_batch:
                    print("No valid encodings in batch, skipping...")
                    continue
                
                padded_batch = nn.utils.rnn.pad_sequence(
                    [torch.tensor(ids) for ids in encoded_batch], 
                    batch_first=True, 
                    padding_value=tokenizer.pad_token_id
                )
                
                if padded_batch.shape[1] > config.block_size - int(level):
                    print(f"Skipping batch due to excessive length: {padded_batch.shape[1]} > {config.block_size - int(level)}")
                    level -= 1
                    upper_level += 0.5
                    continue

                input_tensor = padded_batch.to(config.device)
                
                # Create padding mask
                padding_mask = (input_tensor != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
                
                # Create targets (shift by one for next-token prediction)
                targets = input_tensor.clone()
                targets[:, :-1] = input_tensor[:, 1:]
                targets[:, -1] = tokenizer.pad_token_id
                
                # Forward pass
                output_logits = model(input_tensor, padding_mask)

                # Reshape for loss calculation
                logits_flat = output_logits.reshape(-1, output_logits.size(-1))
                targets_flat = targets.reshape(-1)
                
                loss = criterion(logits_flat, targets_flat)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1

                # Logging
                if global_step % 2 == 0:
                    print(f"Epoch {epoch+1}, Step {global_step}: Loss = {loss.item():.4f}, LR = {optimizer.param_groups[0]['lr']:.8f}")
                    
                if global_step % 10 == 0 and input_texts:
                    print(f"Sample input: {input_texts[0][:200]}...")
                    try:
                        generated_ids = output_logits[0].argmax(dim=-1)
                        generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=False)
                        print(f"Generated: {generated_text[:200]}...")
                    except Exception as e:
                        print(f"Error decoding generation: {e}")
                    print("_" * 80)
                
                level -= 1
                upper_level += 0.6
                
                # Save checkpoint
                if global_step % 50 == 0 and save_file:
                    try:
                        save_model(model, tokenizer, config, save_file)
                        print(f"Model saved at step {global_step} to {save_file}")
                    except Exception as e:
                        print(f"Error saving model: {e}")
                    
            except Exception as e:
                print(f"Error in training step {global_step}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    print("Training complete.")
    return model

def train_full_sequence(model, tokenizer, config, data, epochs=10, batch_size=2, lr=3e-4, min_lr=5e-6, save_file=None):
    """
    Fixed version of batched training function.
    """
    level = 300
    upper_level = 3
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-7)
    total_steps = (len(data) // batch_size) * epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, eta_min=5e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.055, ignore_index=tokenizer.pad_token_id)
    
    model.to(config.device)
    model.train()

    print("Starting batched, full-sequence training...")
    global_step = 0
    
    for epoch in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            try:
                batch_samples = data[i:i + batch_size]
                
                # Check model resizing
                if model.token_embedding.num_embeddings != len(tokenizer):
                    print("✨ Vocabulary has grown. Resizing model...")
                    resize_model_if_needed(model, len(tokenizer), config.device)
                    print("✅ Model resized successfully.")
                    if save_file:
                        save_model(model, tokenizer, config, save_file)
                
                input_texts = []
                for sample in batch_samples:
                    try:
                        if isinstance(sample, str) and '\t' in sample:
                            prompt, thought, response = sample.split('\t')
                        elif isinstance(sample, dict):
                            memory = sample.get('Memory','')
                            prompt = sample.get('prompt', '')
                            thought = sample.get('thought', '')
                            response = sample.get('response', '')
                        else:
                            continue
                            
                        choice = random.choice([True,False])
                        think = random.choice([True,False])
                        think1 = random.choice([True,False])
                        think2 = random.choice([True,False])
                        
                        if think1 == think2:
                            think = True
                        elif think1 == True and think2 != True:
                            think = True
                        else:
                            think = False
                            
                        if choice == True and 'memory' in locals():
                            input_texts.append(f"<PREV>{memory.lower()} </PREV> , <PROMPT> {prompt.lower()} </PROMPT> {thought.lower()} {response.lower()}")
                        elif think == True:
                            input_texts.append(f"<PROMPT> {prompt.lower()} </PROMPT> {thought.lower()} {response.lower()}")
                        else:
                            input_texts.append(f"<PROMPT> {prompt.lower()} </PROMPT> <THOUGHT> No Thought </THOUGHT>{response.lower()}")
                            
                    except (ValueError, AttributeError) as e:
                        print(f"Error processing sample: {e}")
                        continue

                if not input_texts:
                    continue

                encoded_batch = [tokenizer.encode(text) for text in input_texts]
                padded_batch = nn.utils.rnn.pad_sequence(
                    [torch.tensor(ids) for ids in encoded_batch], 
                    batch_first=True, 
                    padding_value=tokenizer.pad_token_id
                )
                
                if padded_batch.shape[1] > config.block_size - int(level):
                    print("Skipping batch due to excessive length.")
                    level -= 1
                    upper_level += 0.5
                    continue

                input_tensor = padded_batch.to(config.device)
                
                # Create padding mask
                padding_mask = (input_tensor != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
                
                targets = input_tensor.clone()
                targets[:, :-1] = input_tensor[:, 1:]
                targets[:, -1] = tokenizer.pad_token_id
                
                # Forward pass
                output_logits = model(input_tensor, padding_mask)

                # FIXED: Use .reshape() instead of .view()
                logits_flat = output_logits.reshape(-1, output_logits.size(-1))
                targets_flat = targets.reshape(-1)
                
                loss = criterion(logits_flat, targets_flat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1

                if global_step % 2 == 0:
                    print(f"Epoch {epoch+1}, Step {global_step}: Loss = {loss.item():.4f}, LR = {optimizer.param_groups[0]['lr']:.18f}")
                    if global_step % len(input_texts) == 0 and input_texts:
                        print(f"Data:{input_texts[0]}")
                        print(f"Generated: {tokenizer.decode(output_logits[0].argmax(dim=-1).tolist())}")
                        print("_" * 80)
                
                level -= 1
                upper_level += 0.6
                
                if global_step % 50 == 0 and save_file:
                    save_model(model, tokenizer, config, save_file)
                    print(f"Model saved at step {global_step} to {save_file}")
                    
            except Exception as e:
                print(f"Error in step {global_step}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    print("Training complete.")
    return model
# In your main script, before the loop() function


# Then, inside loop() or at the start of your script
human_training_data = load_human_training_data()
from torch.optim.lr_scheduler import LambdaLR ,LinearLR
def train_batched_full_sequence(model, tokenizer, config, data, epochs=10, batch_size=2, lr=1e-9, min_lr=1e-15, max_lr=1e-8, save_file=None):
    """
    Trains the model on batches of full sequences, a more stable and efficient method,
    with a learning rate scheduler and a clamp to keep the LR within a safe range.
    """
    level = 300
    upper_level = 3
    # The initial LR from the arguments is used to set up the optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-6)
    
    total_steps = (len(data) // batch_size) * epochs

    # Use a more robust LinearLR scheduler for a linear decay.
    # It scales the initial LR from 1 down to (min_lr / initial_lr).
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=400, eta_min=2e-6)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.005, ignore_index=tokenizer.pad_token_id)
    model.to(config.device)
    model.train()

    print("Starting batched, full-sequence training...")
    global_step = 0
    for epoch in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch_samples = data[i:i + batch_size]
            
            # This logic for resizing the model is correct and can be kept.
            if model.token_embedding.num_embeddings != len(tokenizer):
                print("✨ Vocabulary has grown. Resizing model...")
                resize_model_if_needed(model, len(tokenizer), config.device)
                print("✅ Model resized successfully.")
                save_model(model, tokenizer, config, save_file)
            
            input_texts = []
            for sample in batch_samples:
                try:
                    prompt, thought, response = sample.split('\t')
                except (AttributeError, ValueError):
                    # This block is fragile. Consider pre-processing your data to a consistent format.
                    memory = sample.get('Memory','')
                    prompt = sample.get('prompt', '')
                    thought = sample.get('thought', '')
                    response = sample.get('response', '')
                choice = random.choice([True,False])
                think = random.choice([True,False])
                think1 = random.choice([True,False])
                think2 = random.choice([True,False])
                if think1 == think2:
                    think = True
                elif think1 == True and think2 != True:
                    think = True
                else:
                    think = False
                if choice == True:
                    input_texts.append(f"<PREV>{memory.lower()} </PREV> , <PROMPT> {prompt.lower()} </PROMPT> {thought.lower()} {response.lower()}")
                elif think == True:
                    input_texts.append(f"<PROMPT> {prompt.lower()} </PROMPT> {thought.lower()} {response.lower()}")
                else:
                    input_texts.append(f"<PROMPT> {prompt.lower()} </PROMPT> <THOUGHT>I will respond politely and informatively </THOUGHT>{response.lower()}")
                    

            encoded_batch = [tokenizer.encode(text) for text in input_texts]
            padded_batch = nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in encoded_batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
            
            if padded_batch.shape[1] > config.block_size - np.round(level,0):  # Adjusted to allow for longer sequences
                print("Skipping batch due to excessive length.")
                level -= 1
                upper_level += 0.5
                continue
            

            input_tensor = padded_batch.to(config.device)
            
            # Create a padding mask to ensure attention ignores padded tokens.
            # 1 for tokens to attend to, 0 for padded tokens to ignore.
            padding_mask = (input_tensor != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

            
            targets = input_tensor.clone()
            targets[:, :-1] = input_tensor[:, 1:]
            targets[:, -1] = tokenizer.pad_token_id 
            if model.token_embedding.num_embeddings != len(tokenizer):
                print("✨ Vocabulary has grown. Resizing model...")
                resize_model_if_needed(model, len(tokenizer), config.device)
                print("✅ Model resized successfully.")
                save_model(model, tokenizer, config, save_file)
            
            loss_mask = torch.ones_like(targets, dtype=torch.bool).to(config.device)
            for b_idx, ids in enumerate(encoded_batch):
                try:
                    prompt_end = ids.index(tokenizer.encode("</PROMPT>")[0])  # index of </PROMPT>
                    loss_mask[b_idx, :prompt_end+1] = 0  # mask everything before and including </PROMPT>
                except ValueError:
                    pass  # if </PROMPT> not found, don’t mask
            # The model's forward pass must be updated to accept and use this padding_mask.
            output_logits = model(input_tensor, padding_mask)

            logits_flat = output_logits.view(-1, output_logits.size(-1))
            targets_flat = targets.view(-1)
            loss_mask_flat = loss_mask.view(-1)

            # 🔹 Apply mask so loss is only counted after </PROMPT>
            loss = criterion(logits_flat[loss_mask_flat], targets_flat[loss_mask_flat])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            global_step += 1

            if global_step % 2 == 0:
                print(f"Epoch {epoch+1}, Step {global_step}: Loss = {loss.item():.4f}, LR = {optimizer.param_groups[0]['lr']:.18f}")
                print(f"Data:{input_texts[global_step % len(input_texts)]}")
                print("___")
                print(f"Generated: {tokenizer.decode(output_logits[0].argmax(dim=-1).tolist())}")
                print("________________________________________________________________________________")
            level -= 1
            upper_level += 0.6
            if global_step % 50 == 0 and save_file:
                save_model(model, tokenizer, config, save_file)
                print(f"Model saved at step {global_step} to {save_file}")
    print("Training complete.")
    return model
import torch
def feedback_listener(model, tokenizer, optimizer): # Accept optimizer as an argument
    global rlhf_buffer
    criterion = nn.CrossEntropyLoss()
    while True:
        if rlhf_buffer['prompt'] and rlhf_buffer['response']:
            print("Press 'g' for good, 'b' for bad feedback on the last response.")
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                if event.name == 'g':
                    reward = 1.0
                elif event.name == 'b':
                    reward = -1.0 # Negative reward is more direct than 0.1
                else:
                    continue
                
                model.train()
                
                # Correctly encode and tensorize inputs and targets
                prompt_ids = tokenizer.encode(rlhf_buffer['prompt'])
                response_ids = tokenizer.encode(rlhf_buffer['response'])
                input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(config.device)
                target_ids = torch.tensor([response_ids], dtype=torch.long).to(config.device)
                
                # Combine prompt and response for full sequence loss
                full_sequence = torch.cat([input_ids, target_ids], dim=1)
                
                # Create targets by shifting the sequence
                targets = full_sequence.clone()
                targets[:, :-1] = full_sequence[:, 1:]
                targets = targets[:, -target_ids.shape[1]:].contiguous() # Only calculate loss on the response part

                outputs = model(full_sequence)
                
                # Calculate loss only on the generated response part of the sequence
                response_logits = outputs[:, -target_ids.shape[1]:, :].contiguous()
                
                loss = criterion(response_logits.view(-1, response_logits.size(-1)), targets.view(-1))
                
                weighted_loss = loss * reward 
                
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()
                
                print(f"\nApplied RLHF update with reward={reward}\n")
                
                rlhf_buffer = {'prompt': None, 'response': None}
        tm.sleep(0.1)
def chat_loop(model,tokenizer):
    global rlhf_buffer
    while True:
        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.strip().lower() == 'exit':
            break
        # Generate response as before
        structured_prompt = f"<PROMPT> {prompt.lower()} </PROMPT>"#
        input_ids = tokenizer.encode(structured_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(config.device)
        response = stream_generate(model,input_ids=input_tensor ,tokenizer=tokenizer, temperature=0.2, top_k=90, max_new_tokens=100, repetition_penalty=1.3)
        response = tokenizer.decode(response[0].tolist())
        print(response)
        # Store for RLHF feedback
        rlhf_buffer['prompt'] = prompt
        rlhf_buffer['response'] = response  


def loop(): #
    global model, tokenizer, config, reward_buffer
    model = Transformer(vocab_size=len(vocab), d_model=config.d_model).to(config.device) #
    tokenizer = SimpleTokenizer(vocab) #
    save_file = "LLm-1.pth" #
    vocab_path = "Vocab-1.txt" #
    if os.path.exists(vocab_path):
        print("Loading Vocabulary...")
        tokenizer = SimpleTokenizer([])
        tokenizer.load_vocab(vocab_path)  # No need to assign this to 'vocab'
    else:
        print("Vocabulary not found, using default tokenizer.")
        initial_vocab = ["<PAD>", "<UNK>", "<EOS>", "<TEXT>", "</TEXT>", "<THOUGHT>", "</THOUGHT>", "<USER>", "<PROMPT>", "</PROMPT>","<PREV>","</PREV>"] + list(vocab)  # Add default tokens
        tokenizer = SimpleTokenizer(initial_vocab)
        tokenizer.save_vocab(vocab_path)
        repr(f"Vocabulary saved to {vocab_path}")

    # Initialize reward_buffer here as it's used globally in run_idle_test
    global reward_buffer
    reward_buffer = []
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Corrected Startup Logic ---
    model, tokenizer, config = load_model(save_file, device)
    
    if model is None:
        print("⚠️ No existing model found, starting fresh.")
        config = Config() # Initialize default config
        
        # Create initial vocab from the dataset
        special_tokens = [t.lower() for t in ['<pad>', '<prompt>', '</prompt>', '', ':', ';', '!', 'hi', 'hello', '<eos>', '<text>', '</text>', '<unk>', '*', '**', '<user>']]
        word_set = set()
        for line in new_samples:
            words = re.findall(r'<[^>]+>|[\w]+|[^\s\w]', line)
            word_set.update(w.lower() for w in words)
        initial_vocab = list(dict.fromkeys(special_tokens + sorted(word_set)))
        
        tokenizer = SimpleTokenizer(initial_vocab)
        config.vocab_size = len(tokenizer.vocab)
        config.device = device
        
        model = Transformer(vocab_size=config.vocab_size, d_model=config.d_model).to(device)
        save_model(model, tokenizer, config, save_file)
    rlhf_optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
    t = threading.Thread(target=feedback_listener, args=(model, tokenizer,rlhf_optimizer), daemon=True)
    t.start()
    tokenized_data = [tokenizer.encode(preprocess(sample)) for sample in new_samples] #
    training_data = create_input_target_pairs(tokenized_data) #

    # Initialize generation parameters
    # Increasing the temperature makes the model more creative by increasing the chances it chooses a different word/token
    # tok_k is the amount of tokens so if you increase that and the temperature it will begin to be alot more creative due to more options
    # repetition_penalty is self explanitory decreases the oporitunity for repeating stuff like words

    current_temperature = 0.25
    current_top_k = 15
    current_repetition_penalty = 1.2
    last_prompt = None
    last_thought = ""
    last_response = None
    while True: #
        # +++ New, Corrected Code +++

        prompt = input("Enter a prompt (or 'exit' to quit): ") #
        
        if prompt.lower().startswith('!t1'): # 
            parts = prompt.strip().split() #
            epochs = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10 #
            print(f"Training model on 30 random samples for {epochs} epochs...") #
            sampled_data = load_random_training_samples(new_samples, tokenizer, count=800) # the amount of data extracted choose based off of how much data you want loaded the more data the more opportunities for learning
            loaded_data = load_cst_dataset(sampled_data)
            model = train_uni_full_sequence(model=model, tokenizer=tokenizer, config=config,dataset_name="Anthropic/hh-rlhf",dataset_subset="train", num_samples=500,epochs=15,save_file=save_file,batch_size=4) # num_samples does the same thing but i am to lazy to remove the sampleing plus it tend to lower ram suggested you use the first sample limiter for better control
            save_model(model, tokenizer, config ,save_file) #
            print("Model trained and saved.") #
            continue #    
        if prompt.lower().startswith('!temp'):
            parts = prompt.strip().split()
            if len(parts) > 1:
                try:
                    new_temp = float(parts[1])
                    if 0.1 <= new_temp <= 2.0: # Example range
                        current_temperature = new_temp
                        print(f"Generation temperature set to: {current_temperature}")
                    else:
                        print("Temperature must be between 0.1 and 2.0.")
                except ValueError:
                    print("Invalid temperature value. Please enter a number.")
            else:
                print(f"Current temperature: {current_temperature}. Usage: !temp <value>")
            continue
        if prompt.lower().startswith('!topk'):
            parts = prompt.strip().split()
            if len(parts) > 1:
                try:
                    new_top_k = int(parts[1])
                    if 1 <= new_top_k <= config.vocab_size:
                        current_top_k = new_top_k
                        print(f"Top-K value set to: {current_top_k}")
                    else:
                        print(f"Top-K must be between 1 and {config.vocab_size}.")
                except ValueError:
                    print("Invalid Top-K value. Please enter an integer.")
            else:
                print(f"Current Top-K: {current_top_k}. Usage: !topk <value>")
            continue
        if prompt.lower().startswith('!rep'):
            parts = prompt.strip().split()
            if len(parts) > 1:
                try:
                    new_rep = float(parts[1])
                    if 1.0 <= new_rep <= 5.0: # Example range
                        current_repetition_penalty = new_rep
                        print(f"Repetition penalty set to: {current_repetition_penalty}")
                    else:
                        print("Repetition penalty must be between 1.0 and 5.0.")
                except ValueError:
                    print("Invalid repetition penalty value. Please enter a number.")
            else:
                print(f"Current Repetition Penalty: {current_repetition_penalty}. Usage: !rep <value>")
            continue    
        if prompt.lower() == '!recent': # #
            print("Recent memory contents:") #
            print(last_thought[1:]) #
            continue # 
        if prompt.lower() == '!clear': #
            last_thought = "" #
            print("Memory cleared.") #
            continue #      
        if prompt.lower() == '!good':
            if last_prompt and last_thought and last_response:
                new_entry = {"prompt": last_prompt,"thought":last_thought ,"response": last_response}
                human_training_data.append(new_entry)
                save_human_training_data(human_training_data)
                print("👍 Saved last conversation pair to human training data.")
            else:
                print("No recent conversation to save.")
            continue
        if prompt.lower() == '!bad':
            if last_prompt:
                better_thought = input("What would have been a better thought? ")
                better_response = input("What would have been a better response? ")
                new_entry = {"prompt": last_prompt,"thought":better_thought, "response": better_response}
                human_training_data.append(new_entry)
                save_human_training_data(human_training_data)
                print("👎 Saved your improved response for future training.")
            else:
                print("No recent conversation to correct.")
            continue
        if prompt.lower() == '!exit': 
            break 
        if prompt.lower() == '!t5':
            # Pass config and capture the returned model
            model = train_batched_full_sequence(model, tokenizer, config, human_training_data, epochs=15,batch_size=2, lr=2e-5 , min_lr=1e-5, max_lr=5e-7 ,save_file=save_file) #
            save_model(model, tokenizer, config, save_file)
            print("Command: !t5 executed. Single-token training.")
            continue
        if prompt.lower() == '!t3': # a use of the second training function suggested you use a secondary dataset (feel free to ask me for another one i have 3 with 2.8k samples)
            print("Improved Training")
            model = train_full_sequence(model,tokenizer,config,human_training_data,epochs=15,batch_size=2, lr= 1e-5,save_file=save_file)
            continue
        if prompt.lower() == '!rlhf':
            print("RLHF mode activated. Provide feedback on responses.")
            chat_loop(model,tokenizer)
            save_model(model, tokenizer, config, save_file)
            continue
        if prompt.lower() == '!save':
            print("Saving model and tokenizer...")
            save_model(model, tokenizer, config, save_file)
            print("Model and tokenizer saved successfully.")
            continue
        # Prepare the prompt for the model
        

        structured_prompt = f"<PROMPT> {prompt.lower()} </PROMPT>"# This version doesnt include memory for simplicity
        structured_prompt = preprocess(structured_prompt) #
        input_ids = tokenizer.encode(structured_prompt) #
        if model.token_embedding.num_embeddings != len(tokenizer):
            print("✨ Vocabulary has grown. Resizing model...")
            # This is the correct function call to resize the model.
            resize_model_if_needed(model, len(tokenizer), config.device)
            print("✅ Model resized successfully.")
            # You may want to retrain the model on the new tokens if they are added
            # during runtime.
            save_model(model, tokenizer, config, save_file)
            continue
        
        
        
        input_ids = tokenizer.encode(structured_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(config.device)
        output = stream_generate(
            model=model,
            input_ids=input_tensor,
            tokenizer=tokenizer,
            temperature=current_temperature,
            top_k=current_top_k,
            max_new_tokens=100,#it can generate up to 100 tokens before it is forcefully stopped feel free to edit the number but beware it will slow down your computer per word
            repetition_penalty=current_repetition_penalty
        )
        print("_______________________________________________________________________________________________",end="\n")
        response = tokenizer.decode(output[0].tolist()).strip()
        thought = extract_all_between_tags(response, "<thought>", "</thought>") or ""
        cleaned = extract_all_between_tags(response, "<text>", "</text>") or response
        
        last_prompt = prompt
        last_thought = thought
        last_response = cleaned

if __name__ == "__main__": #
    loop() #