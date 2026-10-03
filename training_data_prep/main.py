import argparse
import importlib
from pathlib import Path

from ._1_customtokenizer import DEFAULT_TEXT_PATH

DEMOS = {
    "tokenizer": "_1_customtokenizer",
    "bpe": "_2_bpetiktokenizer",
    "pairs": "_3_inputtargetpairs",
    "embeddings": "_4_inputembeddingexample",
    "dataset": "_5_customgptdataset",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tokenization and language-model data preparation."
    )
    commands = parser.add_subparsers(dest="demo", required=True)
    for name in DEMOS:
        command = commands.add_parser(name, help=f"Run the {name} example.")
        if name in ("tokenizer", "pairs", "dataset"):
            command.add_argument("--data", type=Path, default=DEFAULT_TEXT_PATH)
        if name in ("pairs", "dataset"):
            command.add_argument("--context-size", type=int, default=4)
        if name == "dataset":
            command.add_argument("--batch-size", type=int, default=1)
            command.add_argument("--stride", type=int, default=1)
            command.add_argument("--max-batches", type=int, default=2)
            command.add_argument(
                "--interactive", action="store_true", help="Pause between batches."
            )
    args = parser.parse_args(argv)
    if args.demo in ("pairs", "dataset") and args.context_size <= 0:
        parser.error("context-size must be positive")
    if (
        args.demo == "dataset"
        and min(args.batch_size, args.stride, args.max_batches) <= 0
    ):
        parser.error("batch-size, stride, and max-batches must be positive")
    module = importlib.import_module(f".{DEMOS[args.demo]}", package=__package__)
    if args.demo == "tokenizer":
        module.run_demo(data_path=args.data)
    elif args.demo == "pairs":
        module.run_demo(data_path=args.data, context_size=args.context_size)
    elif args.demo == "dataset":
        module.run_demo(
            data_path=args.data,
            batch_size=args.batch_size,
            context_size=args.context_size,
            stride=args.stride,
            max_batches=args.max_batches,
            interactive=args.interactive,
        )
    else:
        module.run_demo()


if __name__ == "__main__":
    main()
