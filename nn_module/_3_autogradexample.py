import torch
from torch.nn import functional as F


def run_demo() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    x = torch.tensor([[1.0, 2.0], [4.0, 5.0], [6.0, 7.0]])
    w = torch.tensor([[0.5], [1.0]], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([[1.0], [0.0], [1.0]])
    z = x @ w + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)

    # Retain the graph so the second approach can differentiate the same loss.
    dw, db = torch.autograd.grad(loss, [w, b], retain_graph=True)
    print("dL/dw1:", dw[0].item())
    print("dL/dw2:", dw[1].item())
    print("dL/db:", db.item())

    loss.backward()
    print("Weight gradients from backward:\n", w.grad)
    print("Bias gradient from backward:\n", b.grad)


if __name__ == "__main__":
    run_demo()
