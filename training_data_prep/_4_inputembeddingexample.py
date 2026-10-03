import torch
from torch import nn


def run_demo() -> None:
    torch.manual_seed(123)
    vocab_size, embedding_dim, context_size = 5, 3, 4
    token_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    position_embedding_layer = nn.Embedding(context_size, embedding_dim)
    input_token_ids = torch.tensor([[1, 2, 3, 4]])
    token_embeddings = token_embedding_layer(input_token_ids)
    position_embeddings = position_embedding_layer(torch.arange(context_size))

    # The (T, D) position vectors broadcast across each batch of token embeddings.
    input_embeddings = token_embeddings + position_embeddings
    print("Token Embedding Layer:\n", token_embedding_layer.weight)
    print("Token Embedding:\n", token_embeddings)
    print("Position Embedding Layer:\n", position_embedding_layer.weight)
    print("Position Embedding:\n", position_embeddings)
    print("Input Embedding:\n", input_embeddings)
    print("Input Embedding Shape:", tuple(input_embeddings.shape))


if __name__ == "__main__":
    run_demo()
