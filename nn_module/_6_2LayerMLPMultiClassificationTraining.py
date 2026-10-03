import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ._4_datasetloaderexample import ToyDataset
from ._5_2layerMLPexample import TwoLayerMLP

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "nn_module" / "model.pth"
)


def create_training_dataset() -> ToyDataset:
    # Features encode teeth, diet, and legs; classes are carnivore, herbivore, omnivore.
    features = torch.tensor(
        [[0, 0, 1], [1, 1, 1], [2, 2, 0], [2, 2, 0], [2, 2, 1], [2, 2, 1]],
        dtype=torch.float,
    )
    labels = torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.long)
    return ToyDataset(features, labels)


def train_model(
    model: TwoLayerMLP, train_loader: DataLoader, num_epochs: int, learning_rate: float
) -> list[float]:
    if num_epochs <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epoch count and learning rate must be positive and finite.")
    if not len(train_loader):
        raise ValueError("The training loader has no batches; reduce the batch size.")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(num_epochs):
        model.train()
        for index, (features, labels) in enumerate(train_loader, start=1):
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(
                f"Epoch {epoch + 1:03d}/{num_epochs:03d}"
                f" | Batch {index:03d}/{len(train_loader):03d}"
                f" | Train Loss: {loss.item():.2f}"
            )
    return losses


def evaluate_model(model: TwoLayerMLP) -> torch.Tensor:
    model.eval()
    features = torch.tensor([[0, 0, 1], [1, 1, 1], [2, 2, 0]], dtype=torch.float)
    with torch.no_grad():
        logits = model(features)
        probabilities = F.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
    print(f"Raw Prediction Scores:\n{logits}")
    print(f"Probabilities:\n{probabilities}")
    print(f"Predicted Classes:\n{predicted_classes}")
    return predicted_classes


def run_demo(
    num_epochs: int = 4,
    batch_size: int = 2,
    learning_rate: float = 0.4,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> tuple[TwoLayerMLP, torch.Tensor, list[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    dataset = create_training_dataset()
    if batch_size > len(dataset):
        raise ValueError("batch_size must not exceed the training dataset size.")
    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    model = TwoLayerMLP(3, 3)
    print(f"Training Dataset Samples: {len(dataset)}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    losses = train_model(model, train_loader, num_epochs, learning_rate)
    predictions = evaluate_model(model)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model weights saved to {output_path}")
    return model, predictions, losses


if __name__ == "__main__":
    run_demo()
