import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
    
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()

if __name__ == "__main__":
    tensor0d = torch.tensor(1)
    tensor1d = torch.tensor([1, 2, 3])
    tensor2d = torch.tensor([[1, 2], [3, 4]])
    tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor1d = torch.tensor([1, 2, 3])
    print(tensor1d.dtype)
    print("\n##############################################\n")

    floatvec = torch.tensor([1.0, 2.0, 3.0])
    print(floatvec.dtype)
    print("\n##############################################\n")

    floatvec = tensor1d.to(torch.float32)
    print(floatvec.dtype)
    print("\n##############################################\n")

    tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(tensor2d)
    print("\n##############################################\n")

    print(tensor2d.shape)
    print("\n##############################################\n")

    print(tensor2d.reshape(3, 2))
    print("\n##############################################\n")

    print(tensor2d.view(3, 2))
    print("\n##############################################\n")

    print(tensor2d.T)
    print("\n##############################################\n")

    print(tensor2d.matmul(tensor2d.T))
    print("\n##############################################\n")

    print(tensor2d @ tensor2d.T)
    print("\n##############################################\n")

    #A logistic regression forward pass
    import torch.nn.functional as F
    
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2])
    b = torch.tensor([0.0])
    z = x1 * w1 + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)
    print(loss)
    print("\n##############################################\n")

    #Computing gradients via autograd
    import torch.nn.functional as F
    from torch.autograd import grad
    
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    z = x1 * w1 + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)
    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b = grad(loss, b, retain_graph=True)
    
    print(grad_L_w1)
    print(grad_L_b)
    print("\n##############################################\n")

    loss.backward()
    print(w1.grad)
    print(b.grad)
    print("\n##############################################\n")

    model = NeuralNetwork(50, 3)
    print(model)
    print("\n##############################################\n")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters:", num_params)
    print("\n##############################################\n")
    print(model.layers[0].weight.shape)
    print(model.layers[0].weight)
    print("\n##############################################\n")

    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)
    print(model.layers[0].weight)
    print("\n##############################################\n")

    torch.manual_seed(123)
    X = torch.rand((1, 50))
    out = model(X)
    print(out)
    print("\n##############################################\n")

    with torch.no_grad():
        out = model(X)
    print(out)
    print("\n##############################################\n")
    
    with torch.no_grad():
        out = torch.softmax(model(X), dim=1)
    print(out)
    print("\n##############################################\n")

    #Creating a small toy dataset
    X_train = torch.tensor([
                [-1.2, 3.1],
                [-0.9, 2.9],
                [-0.5, 2.6],
                [2.3, -1.1],
                [2.7, -1.5]
                ])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor([
                [-0.8, 2.8],
                [2.6, -1.6],
                ])
    y_test = torch.tensor([0, 1])

    #Defining a custom Dataset class
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    
    print(len(train_ds))
    print("\n##############################################\n")

    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0    
    )
    
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    for idx, (x, y) in enumerate(train_loader):
        print(f"Batch {idx+1}:", x, y)
    print("\n##############################################\n")

    #A training loader that drops the last batch
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    for idx, (x, y) in enumerate(train_loader):
        print(f"Batch {idx+1}:", x, y)
    print("\n##############################################\n")

    #Neural network training in PyTorch
    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.5
    )
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train Loss: {loss:.2f}")
    model.eval()
    print("\n##############################################\n")
    # Insert optional model evaluation code
    model.eval()
    with torch.no_grad():
        outputs = model(X_train)
    print(outputs)
    print("\n##############################################\n")

    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(outputs, dim=1)
    print(probas)
    print("\n##############################################\n")

    predictions = torch.argmax(probas, dim=1)
    print(predictions)
    print("\n##############################################\n")

    predictions = torch.argmax(outputs, dim=1)
    print(predictions)
    print("\n##############################################\n")

    predictions == y_train
    sum = torch.sum(predictions == y_train)
    print(sum)
    print("\n##############################################\n")

    print(compute_accuracy(model, train_loader))
    print(compute_accuracy(model, test_loader))
    print("\n##############################################\n")

    torch.save(model.state_dict(), "model.pth")
    model = NeuralNetwork(2, 2)
    model.load_state_dict(torch.load("model.pth"))
    print(torch.cuda.is_available())
    print("\n##############################################\n")

    tensor_1 = torch.tensor([1., 2., 3.])
    tensor_2 = torch.tensor([4., 5., 6.])
    print(tensor_1 + tensor_2)
    print("\n##############################################\n")

    tensor_1 = tensor_1.to("cpu")
    print(tensor_1 + tensor_2)
    print("\n##############################################\n")

    #A training loop on a GPU
    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    device = torch.device("cuda")
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, labels) # Loss function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train/Val Loss: {loss:.2f}")
    model.eval()
 
