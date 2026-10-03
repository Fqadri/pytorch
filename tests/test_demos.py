import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from nn_module import _6_2LayerMLPMultiClassificationTraining as training
from nn_module import main as nn_cli
from nn_module._4_datasetloaderexample import ToyDataset
from nn_module._5_2layerMLPexample import TwoLayerMLP
from training_data_prep import _1_customtokenizer as custom
from training_data_prep import _2_bpetiktokenizer as bpe
from training_data_prep import _3_inputtargetpairs as pairs
from training_data_prep import _5_customgptdataset as windows
from training_data_prep import main as data_cli


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return list(range(12))

    def decode(self, ids):
        return " ".join(map(str, ids))


@pytest.fixture(autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def text_path(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Hello, world! This is a small training example.", encoding="utf-8")
    return path


@pytest.fixture
def fake_encoding(monkeypatch):
    get_encoding = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr(bpe.tiktoken, "get_encoding", get_encoding)
    return get_encoding


def test_imports_have_no_demo_side_effects():
    script = """
import importlib
import pkgutil
from pathlib import Path
from unittest.mock import patch
import tiktoken
import torch

def forbidden(*args, **kwargs):
    raise AssertionError("Importing a module must not execute its demo")

with patch.object(torch, "manual_seed", forbidden), \
     patch.object(torch, "tensor", forbidden), \
     patch.object(torch, "save", forbidden), \
     patch.object(torch.nn.Linear, "__init__", forbidden), \
     patch.object(tiktoken, "get_encoding", forbidden), \
     patch.object(Path, "read_text", forbidden), \
     patch("builtins.input", forbidden):
    for name in ("nn_module", "training_data_prep"):
        package = importlib.import_module(name)
        for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module.name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("demo", ["tensors", "logistic", "autograd", "dataset", "mlp"])
def test_nn_demo_dispatch(demo, capsys):
    nn_cli.main([demo])
    assert capsys.readouterr().out


@pytest.mark.parametrize("demo", ["tokenizer", "bpe", "pairs", "embeddings", "dataset"])
def test_data_demo_dispatch(demo, text_path, fake_encoding, monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input", Mock(side_effect=AssertionError("Unexpected pause"))
    )
    args = [demo]
    if demo in ("tokenizer", "pairs", "dataset"):
        args.extend(["--data", str(text_path)])
    data_cli.main(args)
    assert capsys.readouterr().out


def test_dataset_rejects_inconsistent_sample_counts():
    with pytest.raises(ValueError, match="sample count"):
        ToyDataset(torch.zeros(6, 3), torch.zeros(5, dtype=torch.long))
    with pytest.raises(ValueError, match="matrix"):
        ToyDataset(torch.zeros(6), torch.zeros(6, dtype=torch.long))


def test_training_dataset_includes_all_six_labels():
    dataset = training.create_training_dataset()
    assert len(dataset) == 6
    assert dataset.labels.tolist() == [0, 1, 2, 2, 2, 2]
    features, label = dataset[5]
    assert features.tolist() == [2, 2, 1]
    assert label.item() == 2


def test_train_cli_saves_a_reloadable_checkpoint(tmp_path, capsys):
    path = tmp_path / "nested" / "model.pth"
    torch.manual_seed(123)
    initial = TwoLayerMLP(3, 3)
    nn_cli.main(["train-mlp", "--epochs", "2", "--output", str(path)])
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = TwoLayerMLP(3, 3)
    model.load_state_dict(state, strict=True)
    assert all(torch.isfinite(value).all() for value in state.values())
    assert any(
        not torch.equal(value, initial.state_dict()[name])
        for name, value in state.items()
    )
    assert model(torch.zeros(4, 3)).shape == (4, 3)
    assert capsys.readouterr().out.count("Train Loss:") == 6


def test_training_is_reproducible_and_checkpoint_preserves_predictions(tmp_path):
    model, predictions, losses = training.run_demo(output_path=tmp_path / "first.pth")
    repeated_model, repeated_predictions, repeated_losses = training.run_demo(
        output_path=tmp_path / "second.pth"
    )
    assert len(losses) == 12
    assert all(math.isfinite(loss) for loss in losses)
    assert losses == repeated_losses
    assert torch.equal(predictions, repeated_predictions)
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, repeated_model.state_dict()[name])
    restored = TwoLayerMLP(3, 3)
    restored.load_state_dict(torch.load(tmp_path / "first.pth", weights_only=True))
    assert torch.equal(training.evaluate_model(restored), predictions)


def test_default_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    assert custom.DEFAULT_TEXT_PATH == root / "training_data_prep" / "the-verdict.txt"
    assert custom.load_training_text()
    assert training.DEFAULT_OUTPUT_PATH == root / "outputs" / "nn_module" / "model.pth"
    assert not (tmp_path / "model.pth").exists()
    with pytest.raises(FileNotFoundError):
        custom.load_training_text(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        custom.load_training_text(empty)


def test_custom_tokenizer_roundtrip_and_unknown_tokens():
    tokenizer = custom.CustomTokenizerV1(custom.build_vocabulary("Hello, world!"))
    assert tokenizer.decode(tokenizer.encode("Hello, world!")) == "Hello, world!"
    assert tokenizer.decode(tokenizer.encode("missing")) == "<|unk|>"


def test_input_target_pairs_are_shifted():
    inputs, targets = pairs.create_input_target_pair([0, 1, 2, 3, 4], 4)
    assert inputs == [0, 1, 2, 3]
    assert targets == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        pairs.create_input_target_pair([0, 1], 2)


def test_gpt_windows_respect_context_and_stride():
    dataset = windows.GPTDatasetV1("text", FakeTokenizer(), context_size=4, stride=3)
    assert len(dataset) == 3
    inputs, targets = dataset[1]
    assert inputs.tolist() == [3, 4, 5, 6]
    assert targets.tolist() == [4, 5, 6, 7]
    assert inputs.dtype == targets.dtype == torch.long
    with pytest.raises(ValueError, match="stride"):
        windows.GPTDatasetV1("text", FakeTokenizer(), 4, 0)
    with pytest.raises(ValueError, match="more tokens"):
        windows.GPTDatasetV1("text", FakeTokenizer(), 12, 1)


def test_gpt_loader_supports_spawned_workers(fake_encoding):
    loader = windows.create_dataloader_v1(
        "text", batch_size=2, context_size=4, stride=1, shuffle=False, num_workers=1
    )
    inputs, targets = next(iter(loader))
    assert inputs.shape == targets.shape == (2, 4)
    assert torch.equal(targets, inputs + 1)
    with pytest.raises(ValueError, match="complete batch"):
        windows.create_dataloader_v1("text", batch_size=20, context_size=4, stride=1)


def test_dataset_demo_is_bounded_and_non_interactive_by_default(
    text_path, fake_encoding, monkeypatch, capsys
):
    pause = Mock(side_effect=AssertionError("Unexpected pause"))
    monkeypatch.setattr("builtins.input", pause)
    windows.run_demo(data_path=text_path, max_batches=2)
    output = capsys.readouterr().out
    assert output.count("Batch ") == 2
    assert "Token Embeddings Shape: (1, 4, 3)" in output
    assert "Batch 3:" not in output
    pause.assert_not_called()


def test_interactive_mode_pauses_only_between_batches(
    text_path, fake_encoding, monkeypatch
):
    pause = Mock(return_value="")
    monkeypatch.setattr("builtins.input", pause)
    windows.run_demo(data_path=text_path, max_batches=2, interactive=True)
    pause.assert_called_once()


@pytest.mark.parametrize(
    ("main", "args"),
    [
        (nn_cli.main, []),
        (nn_cli.main, ["unknown"]),
        (nn_cli.main, ["train-mlp", "--epochs", "0"]),
        (nn_cli.main, ["train-mlp", "--batch-size", "7"]),
        (nn_cli.main, ["train-mlp", "--learning-rate", "nan"]),
        (data_cli.main, []),
        (data_cli.main, ["pairs", "--context-size", "0"]),
        (data_cli.main, ["dataset", "--max-batches", "0"]),
        (data_cli.main, ["dataset", "--stride", "-1"]),
        (data_cli.main, ["dataset", "--batch-size", "0"]),
    ],
)
def test_cli_rejects_invalid_arguments(main, args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("module", ["nn_module.main", "training_data_prep.main"])
def test_help_does_not_import_torch_or_tiktoken(module):
    script = f"""
import importlib
import sys
main = importlib.import_module({module!r}).main
try:
    main(["--help"])
except SystemExit as error:
    assert error.code == 0
assert "torch" not in sys.modules
assert "tiktoken" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "usage:" in result.stdout
