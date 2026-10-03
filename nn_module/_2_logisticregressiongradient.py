import torch
from torch.nn import functional as F


def run_demo() -> None:
    x = torch.tensor([1.1, 2.1])
    y = torch.tensor([1.0])
    w = torch.tensor([1.0, 0.5])
    b = torch.tensor([0.0])

    z = torch.dot(x, w) + b
    print(f"Raw Score: {z}")
    a = torch.sigmoid(z)
    print(f"Prediction: {a}")
    loss = F.binary_cross_entropy(a, y)
    print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    run_demo()
