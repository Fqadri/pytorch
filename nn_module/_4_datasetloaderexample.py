import torch
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        if X.ndim != 2 or Y.ndim != 1:
            raise ValueError("Features must be a matrix and labels must be a vector.")
        if X.shape[0] != Y.shape[0] or Y.shape[0] == 0:
            raise ValueError(
                "Features and labels must have the same non-zero sample count."
            )
        self.features = X
        self.labels = Y

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def run_demo() -> None:
    x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0], [5.0, 1.5], [2.7, -1.1], [3.1, 3.0]])
    y = torch.tensor([0, 0, 0, 1, 1])
    train_dataset = ToyDataset(x, y)
    print("Training samples:", len(train_dataset))
    torch.manual_seed(123)
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True
    )
    print("The final incomplete batch is dropped in this example.")
    for idx, (features, labels) in enumerate(train_loader, start=1):
        print(f"Batch {idx}:")
        print("X:", features)
        print("Y:", labels)


if __name__ == "__main__":
    run_demo()
