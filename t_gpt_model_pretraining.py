import torch
from s_gpt_model import GPTModel
import tiktoken
from s_gpt_model import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    #Dont be surprised if you see garbage output. Our model is not pretrained yet.
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    print("\n##############################################\n")

    #Lets begin pretraining
    inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                           [40, 1107, 588]]) # "I really like"]
    targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
                            [1107, 588, 11311]]) # " really like chocolate"]
    
    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)
    print("\n##############################################\n")

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)
    print("\n##############################################\n")

    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    #Still returns garbage output.
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    print("\n##############################################\n")

    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)
    print("\n##############################################\n")

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)
    print("\n##############################################\n")

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    print("\n##############################################\n")

    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)
    print("\n##############################################\n")

    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)
    print("\n##############################################\n")

    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)
    print("\n##############################################\n")

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    print("\n##############################################\n")

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    #Loss is very high.
    print(loss)
    print("\n##############################################\n")