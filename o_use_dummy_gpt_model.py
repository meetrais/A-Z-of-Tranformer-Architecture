import torch
import torch.nn as nn
import tiktoken
import n_dummy_gpt_model as dgpt

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
print("\n################################################################\n")

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}

torch.manual_seed(123)
model = dgpt.DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
decoded_texts = []
for logit in logits:
    decoded_text = tokenizer.decode(torch.argmax(logit, dim=-1).tolist())
    decoded_texts.append(decoded_text)

for i, decoded_text in enumerate(decoded_texts):
    print(f"Decoded text {i+1}: {decoded_text}")# Dont be surprised if this returns garbage. :)
print("\n################################################################\n")