# PyTorch Examples

## Description

Hands-on examples for learning PyTorch and preparing text for language-model
training. The lessons cover tensors, logistic regression, automatic
differentiation, datasets, multilayer perceptrons, tokenization, input-target
pairs, and embeddings.

This is a collection of small learning demos, not a pretrained model or a
production training framework. Numbered lesson files retain the implementation;
each folder has a command-line entry point to select a demo. Importing a module
does not run its demo, train a model, read a dataset, or save a checkpoint.

## Folder organization

- [nn_module/](nn_module/) - Tensor and neural-network lessons.
  - [main.py](nn_module/main.py) selects a demo.
  - Lessons 1-3 cover tensors, logistic regression, and autograd.
  - [_4_datasetloaderexample.py](nn_module/_4_datasetloaderexample.py) defines
    the reusable dataset and demonstrates batching.
  - [_5_2layerMLPexample.py](nn_module/_5_2layerMLPexample.py) defines the reusable MLP.
  - [_6_2LayerMLPMultiClassificationTraining.py](nn_module/_6_2LayerMLPMultiClassificationTraining.py)
    trains that MLP on a six-sample, three-class toy dataset and saves its weights.
- [training_data_prep/](training_data_prep/) - Text-processing lessons.
  - [main.py](training_data_prep/main.py) selects a demo.
  - Lessons 1-4 cover a custom tokenizer, GPT-2 BPE tokenization, shifted
    input-target pairs, and token/position embeddings.
  - [_5_customgptdataset.py](training_data_prep/_5_customgptdataset.py) provides
    a sliding-window dataset, data loader, and bounded embedding demo.
  - [the-verdict.txt](training_data_prep/the-verdict.txt) is the bundled sample text.
- [tests/](tests/) - Offline regression tests for imports, commands, data
  validation, tokenization, batching, training, and checkpoints.
- [pyproject.toml](pyproject.toml) - Project metadata, dependencies, and lint settings.
- [uv.lock](uv.lock) - Resolved dependency versions.
- [.python-version](.python-version) - Python 3.14, managed by uv.

Generated checkpoints go under `outputs/` by default. Checkpoints and this output
directory are excluded from Git.

## Run locally

### Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Open a terminal in this repository's root directory.
3. Install the runtime and development dependencies:

   ```powershell
   uv sync
   ```

uv manages the Python 3.14 virtual environment and installs PyTorch, tiktoken,
NumPy, pytest, and Ruff. The project uses the Microsoft package index configured
in [pyproject.toml](pyproject.toml).

This is a non-package uv project: no build backend or `module-name` is needed.
Use `uv run -m ...` from the repository root so package imports resolve
correctly. In VS Code, select `.venv\Scripts\python.exe` as the interpreter.

### Neural-network demos

```powershell
uv run -m nn_module.main tensors
uv run -m nn_module.main logistic
uv run -m nn_module.main autograd
uv run -m nn_module.main dataset
uv run -m nn_module.main mlp
uv run -m nn_module.main train-mlp
```

The training demo reuses the dataset and MLP from the earlier lessons. It uses
six labeled samples, hidden layers of 30 and 20 units, and three output classes.
Defaults are 4 epochs, batch size 2, and SGD learning rate 0.4. These introductory
demos use CPU tensors; reporting CUDA availability does not move them to a GPU.

Customize a training run:

```powershell
uv run -m nn_module.main train-mlp --epochs 10 --batch-size 2 --learning-rate 0.1 --output outputs\experiment\model.pth
```

Epoch count and learning rate must be positive; batch size must be between 1
and 6. The final incomplete batch is dropped. With the default batch size, all
six training samples are included each epoch.

The default checkpoint is `outputs\nn_module\model.pth`. It contains the model's
PyTorch state dictionary, not optimizer state. Repeated runs overwrite that
file; use `--output` to keep separate checkpoints.

### Text and training-data demos

```powershell
uv run -m training_data_prep.main tokenizer
uv run -m training_data_prep.main bpe
uv run -m training_data_prep.main pairs
uv run -m training_data_prep.main embeddings
uv run -m training_data_prep.main dataset
```

The `tokenizer`, `pairs`, and `dataset` commands use the bundled text by default.
The path is resolved relative to the lesson module, not the terminal directory.
Use `--data` to supply a different UTF-8 text file:

```powershell
uv run -m training_data_prep.main tokenizer --data path\to\corpus.txt
uv run -m training_data_prep.main pairs --data path\to\corpus.txt --context-size 8
```

The BPE, pairs, and dataset demos use tiktoken's GPT-2 encoding. Its first use may
need internet access to download tokenizer data; subsequent runs use the cache.

The dataset demo defaults to batch size 1, context size 4, stride 1, and at most
2 displayed batches. It does not wait for keyboard input unless requested:

```powershell
uv run -m training_data_prep.main dataset --batch-size 2 --context-size 8 --stride 4 --max-batches 3
uv run -m training_data_prep.main dataset --max-batches 3 --interactive
```

`--interactive` pauses only between displayed batches. All sizes and limits
must be positive, and the text must provide enough tokens for a complete batch.

### Help and validation

```powershell
uv run -m nn_module.main --help
uv run -m nn_module.main train-mlp --help
uv run -m training_data_prep.main dataset --help
uv run python -m pytest -q
uv run ruff check nn_module training_data_prep tests
uv run ruff format --check nn_module training_data_prep tests
uv pip check
```

The tests use small models, temporary output files, and fake BPE tokenizers.
They do not download tokenizer data or modify the bundled corpus.