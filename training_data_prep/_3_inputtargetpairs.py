from pathlib import Path

import tiktoken

from ._1_customtokenizer import DEFAULT_TEXT_PATH, load_training_text


def create_input_target_pair(
    token_ids: list[int], context_size: int
) -> tuple[list[int], list[int]]:
    if context_size <= 0 or len(token_ids) <= context_size:
        raise ValueError(
            "context_size must be positive and smaller than the token count."
        )
    return token_ids[:context_size], token_ids[1 : context_size + 1]


def run_demo(data_path: str | Path = DEFAULT_TEXT_PATH, context_size: int = 4) -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(load_training_text(data_path))
    inputs, targets = create_input_target_pair(encoded_text, context_size)
    print("Length:", len(encoded_text))
    print("Encoded text:", encoded_text[:100])
    print("Input:", inputs)
    print("Target:", targets)
    for index in range(1, context_size + 1):
        context = encoded_text[:index]
        desired = encoded_text[index]
        print(context, "---->", desired)
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


if __name__ == "__main__":
    run_demo()
