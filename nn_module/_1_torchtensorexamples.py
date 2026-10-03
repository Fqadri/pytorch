import torch


def run_demo() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    tensor0d = torch.tensor(1)
    tensor1d = torch.tensor([1.1, 2.1, 3.2])
    tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor3d = torch.tensor(
        [
            [[1, 2, 4, 5], [3, 4, 3, 4], [5, 6, 1, 1]],
            [[7, 8, 3, 4], [9, 10, 1, 2], [11, 12, 12, 3]],
        ]
    )
    for tensor in (tensor0d, tensor1d, tensor2d, tensor3d):
        print(
            f"{tensor.ndim}-D tensor, shape={tuple(tensor.shape)}, dtype={tensor.dtype}:"
        )
        print(tensor)

    # Reshaping preserves the number of elements; transposing swaps the axes.
    print("Reshaped to (3, 2):\n", tensor2d.view(3, 2))
    print("Transposed:\n", tensor2d.T)
    print("Matrix product (2, 3) @ (3, 2):\n", tensor2d.matmul(tensor2d.T))


if __name__ == "__main__":
    run_demo()
