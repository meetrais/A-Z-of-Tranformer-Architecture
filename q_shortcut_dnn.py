import torch
import torch.nn as nn
import p_layernorm_gelu_feedforward as p

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), p.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), p.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), p.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), p.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), p.GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

if __name__ == "__main__":

    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )
    print_gradients(model_without_shortcut, sample_input)
    print("\n################################################################\n")

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    print_gradients(model_with_shortcut, sample_input)
    print("\n################################################################\n")

