import torch

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print("################################################################")
print(embedding_layer.weight)
print("################################################################")
print(embedding_layer(torch.tensor([3])))
print("################################################################")
print(embedding_layer(input_ids))
print("################################################################")