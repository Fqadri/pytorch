import tiktoken


def run_demo() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = f"{text1}<|endoftext|> {text2}"
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("Encoded text:", ids)
    print("Number of tokens:", len(ids))
    for token_id in ids:
        print(f"{token_id}: {tokenizer.decode([token_id])}")
    print("Decoded text:", tokenizer.decode(ids))


if __name__ == "__main__":
    run_demo()
