import argparse
import importlib
import math
from pathlib import Path

DEMOS = {
    "tensors": "_1_torchtensorexamples",
    "logistic": "_2_logisticregressiongradient",
    "autograd": "_3_autogradexample",
    "dataset": "_4_datasetloaderexample",
    "mlp": "_5_2layerMLPexample",
    "train-mlp": "_6_2LayerMLPMultiClassificationTraining",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch tensor and neural-network demos."
    )
    commands = parser.add_subparsers(dest="demo", required=True)
    for name in DEMOS:
        command = commands.add_parser(name, help=f"Run the {name} example.")
        if name == "train-mlp":
            command.add_argument("--epochs", type=int, default=4)
            command.add_argument("--batch-size", type=int, default=2)
            command.add_argument("--learning-rate", type=float, default=0.4)
            command.add_argument("--output", type=Path, help="Checkpoint file path.")
    args = parser.parse_args(argv)
    if args.demo == "train-mlp":
        if args.epochs <= 0 or not 1 <= args.batch_size <= 6:
            parser.error(
                "epochs must be positive and batch-size must be between 1 and 6"
            )
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("learning-rate must be positive and finite")
    module = importlib.import_module(f".{DEMOS[args.demo]}", package=__package__)
    if args.demo == "train-mlp":
        module.run_demo(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_path=args.output
            if args.output is not None
            else module.DEFAULT_OUTPUT_PATH,
        )
    else:
        module.run_demo()


if __name__ == "__main__":
    main()
