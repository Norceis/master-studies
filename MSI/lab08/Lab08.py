import numpy as np
import torch
import torchvision
from sklearn.metrics import balanced_accuracy_score
from torchvision.transforms import transforms
from tqdm import tqdm

torch.manual_seed(2137)
np.random.seed(2137)

def gate(a, b, weight_idx):
    match weight_idx:
        case 0:
            return torch.zeros_like(a)
        case 1:
            return a * b
        case 2:
            return a - a * b
        case 3:
            return a
        case 4:
            return b - a * b
        case 5:
            return b
        case 6:
            return a + b - 2 * a * b
        case 7:
            return a + b - a * b
        case 8:
            return 1 - (a + b - a * b)
        case 9:
            return 1 - (a + b - 2 * a * b)
        case 10:
            return 1 - b
        case 11:
            return 1 - b + a * b
        case 12:
            return 1 - a
        case 13:
            return 1 - a + a * b
        case 14:
            return 1 - a * b
        case 15:
            return torch.ones_like(a)


def gate_loop(a, b, weights):
    gate_sum = torch.zeros_like(a)
    for idx in range(16):
        gate_output = gate(a, b, idx)
        gate_sum = gate_sum + weights[..., idx] * gate_output
    return gate_sum


class LogicLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):

        super().__init__()
        self.weights = torch.nn.parameter.Parameter(torch.randn(output_dim, 16, device='cpu'))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.indices = self.get_connections()
        self.num_weights = output_dim * 16

    def forward(self, x):

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = gate_loop(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = gate_loop(a, b, weights)
        return x

    def get_connections(self):
        randomized_connections = torch.randperm(2 * self.output_dim) % self.input_dim
        randomized_connections = torch.randperm(self.input_dim)[randomized_connections]
        randomized_connections = randomized_connections.reshape(2, self.output_dim)
        return randomized_connections[0], randomized_connections[1]


class Pool(torch.nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.class_count = class_count

    def forward(self, x):
        return x.reshape(*x.shape[:-1], self.class_count, x.shape[-1] // self.class_count).sum(-1)


def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    training_set = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    return training_loader, test_loader


def get_model(input_dim: int = 784, class_count: int = 10, num_neurons: int = 1000, num_layers: int = 5):
    layers = [torch.nn.Flatten(), LogicLayer(input_dim=input_dim, output_dim=num_neurons)]

    for _ in range(num_layers - 1):
        layers.append(LogicLayer(input_dim=num_neurons, output_dim=num_neurons))

    model = torch.nn.Sequential(
        *layers,
        Pool(class_count)
    )

    print(f'Total number of neurons: {sum(map(lambda x: x.output_dim, layers[1:-1]))}')
    print(f'Total number of weights: {sum(map(lambda x: x.num_weights, layers[1:-1]))}')
    model = model.to('cpu')
    print(model)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, loss_function, optimizer

EPOCHS = 1

train_loader_normal, test_loader_normal = load_dataset()
model, loss_function, optimizer = get_model()

for _ in range(EPOCHS):
    for x, y in tqdm(train_loader_normal, desc='Training samples'):
        x = model(x)
        loss = loss_function(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

predictions = []
true_classes = []
for x, y in tqdm(test_loader_normal, desc='Test samples'):
    pred = model(x)
    little_prediction = pred.detach().numpy()
    little_classes = y.detach().numpy()
    for idx in range(len(little_prediction)):
        predictions.append(little_prediction[idx])
        true_classes.append(little_classes[idx])

class_predictions = np.argmax(predictions, axis=1)
print(f'BAS: {balanced_accuracy_score(true_classes, class_predictions)}')