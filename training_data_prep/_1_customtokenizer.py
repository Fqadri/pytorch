import re
from pathlib import Path

DEFAULT_TEXT_PATH = Path(__file__).with_name("the-verdict.txt")


def load_training_text(data_path: str | Path = DEFAULT_TEXT_PATH) -> str:
    text = Path(data_path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Training text must not be empty.")
    return text


def tokenize_text(text: str) -> list[str]:
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [item.strip() for item in tokens if item.strip()]


def build_vocabulary(text: str) -> dict[str, int]:
    tokens = sorted(set(tokenize_text(text)))
    tokens.extend(["<|unk|>", "<|endoftext|>"])
    return {token: index for index, token in enumerate(tokens)}


class CustomTokenizerV1:
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {index: token for token, index in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = tokenize_text(text)
        tokens = [token if token in self.str_to_int else "<|unk|>" for token in tokens]
        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[index] for index in ids)
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)


def run_demo(data_path: str | Path = DEFAULT_TEXT_PATH) -> None:
    raw_text = load_training_text(data_path)
    print("Total number of characters:", len(raw_text))
    print(raw_text[:1000])
    tokens = tokenize_text(raw_text)
    print("Total number of tokens:", len(tokens))
    print(tokens[:100])
    vocab = build_vocabulary(raw_text)
    print("Vocabulary size:", len(vocab))
    print("Vocabulary sample:", list(vocab.items())[:51])

    tokenizer = CustomTokenizerV1(vocab)
    examples = (
        '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.',
        "Hello, do you like tea?<|endoftext|> In the sunlit terraces of the palace.",
    )
    for text in examples:
        ids = tokenizer.encode(text)
        print("Text:", text)
        print("Encoded text:", ids)
        print("Decoded text:", tokenizer.decode(ids))


if __name__ == "__main__":
    run_demo()
