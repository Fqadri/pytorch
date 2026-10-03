from itertools import islice
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from ._1_customtokenizer import DEFAULT_TEXT_PATH, load_training_text


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, context_size: int, stride: int):
        if context_size <= 0 or stride <= 0:
            raise ValueError("context_size and stride must be positive.")
        token_ids = tokenizer.encode(txt)
        if len(token_ids) <= context_size:
            raise ValueError("Text must contain more tokens than the context size.")
        self.input_tokenids = []
        self.target_tokenids = []
        for index in range(0, len(token_ids) - context_size, stride):
            self.input_tokenids.append(
                torch.tensor(token_ids[index : index + context_size], dtype=torch.long)
            )
            self.target_tokenids.append(
                torch.tensor(
                    token_ids[index + 1 : index + context_size + 1], dtype=torch.long
                )
            )

    def __len__(self) -> int:
        return len(self.input_tokenids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_tokenids[idx], self.target_tokenids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    context_size: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError(
            "batch_size must be positive and num_workers must be non-negative."
        )
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, context_size, stride)
    if drop_last and len(dataset) < batch_size:
        raise ValueError(
            "The dataset is smaller than one complete batch; reduce batch_size."
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def run_demo(
    data_path: str | Path = DEFAULT_TEXT_PATH,
    batch_size: int = 1,
    context_size: int = 4,
    stride: int = 1,
    max_batches: int = 2,
    interactive: bool = False,
) -> None:
    if max_batches <= 0:
        raise ValueError("max_batches must be positive.")
    dataloader = create_dataloader_v1(
        load_training_text(data_path),
        batch_size=batch_size,
        context_size=context_size,
        stride=stride,
        shuffle=False,
    )
    torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(50257, 3)
    position_embedding_layer = torch.nn.Embedding(context_size, 3)
    positions = position_embedding_layer(torch.arange(context_size))
    print(f"Showing {min(max_batches, len(dataloader))} of {len(dataloader)} batches.")
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(
            islice(dataloader, max_batches), start=1
        ):
            print(f"Batch {index}:")
            print("Input:", inputs.numpy())
            print("Target:", targets.numpy())
            token_embeddings = token_embedding_layer(inputs)
            print("Token Embeddings Shape:", tuple(token_embeddings.shape))
            print("Token Embeddings:\n", token_embeddings)
            print("Input Embedding:\n", token_embeddings + positions)
            if interactive and index < min(max_batches, len(dataloader)):
                input("Press Enter to continue...")


if __name__ == "__main__":
    run_demo()
