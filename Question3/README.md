# Q3: Aksharantar Hindi Transliteration

This project implements character-level transliteration from Romanized Hindi to Devanagari using PyTorch.

Example:

```text
shastragaar -> शस्त्रागार
```

## Dataset

The raw Aksharantar Hindi files are headerless two-column CSV files:

| Column | Meaning |
| --- | --- |
| 1 | Romanized input |
| 2 | Devanagari target |

The original files in `data/raw/` are never modified.

| Split | Rows |
| --- | ---: |
| Train | 51,200 |
| Validation | 4,096 |
| Test | 4,096 |

`src/dataset.py` inspects the splits and defines the PyTorch Dataset. Source and target strings are encoded separately. Target sequences receive `<SOS>` and `<EOS>` markers. Samples remain variable-length; `collate_pairs()` pads only when forming a batch.

## Vocabulary

`src/vocabulary.py` builds separate source and target character vocabularies from the training split only. Both contain `<PAD>`, `<SOS>`, `<EOS>`, and `<UNK>`. Validation and test data are not used to create the mappings.

| Vocabulary | Size |
| --- | ---: |
| Source | 30 |
| Target | 68 |

## Model

```text
Romanized IDs -> embedding -> Encoder -> Decoder -> Devanagari logits
```

- `src/encoder.py` encodes the padded Romanized sequence.
- `src/decoder.py` generates one target character at a time.
- `src/seq2seq.py` connects both modules, starts decoding with `<SOS>`, and supports teacher forcing.
- The recurrent cell is configurable as `RNN`, `GRU`, or `LSTM`.
- LSTM carries both hidden and cell states.

Current settings are `E=128`, `H=256`, one encoder layer, and one decoder layer. Other hyperparameters are centralized in `src/config.py`.

## Training

`src/train.py` builds train and validation DataLoaders, uses cross-entropy with the target `<PAD>` ID as `ignore_index`, and saves the best validation checkpoint to `checkpoints/best_model.pt`.

```bash
python3 -m src.train
```

A one-epoch smoke test is available without changing the normal configuration:

```bash
python3 -c "from src.train import run_training_smoke_test; run_training_smoke_test()"
```

## Test Evaluation

`src/evaluate.py` builds vocabularies from training data, loads the best checkpoint, and evaluates only `hin_test.csv`. Generation is autoregressive, starts with `<SOS>`, and stops at `<EOS>` or a maximum length.

```bash
python3 -m src.evaluate
```

Predictions are written to `results/predictions/test_predictions.csv`.

## Complexity Analysis

`src/complexity.py` counts multiply-accumulate operations (MACs). The convention includes encoder and decoder recurrent matrix multiplications and the decoder output projection. It excludes embedding lookups, biases, activations, padding, and control flow.

```text
C = 2*T*G*(E*H + H*H) + T*H*V
```

Here, `G=1` for RNN, `G=3` for GRU, and `G=4` for LSTM. With `E=128`, `H=256`, `T=21`, and `V=68`:

| Cell | Computation |
| --- | ---: |
| RNN | 4,494,336 MACs |
| GRU | 12,751,872 MACs |
| LSTM | 16,880,640 MACs |

## Parameter Analysis

### Assignment theoretical count

The assignment formula assumes one common vocabulary size `V`. Using the requested `V=68`:

```text
P = V*E + 2*G*(E*H + H*H + H) + H*V + V
```

### Actual implementation count

The implementation uses separate vocabularies: source vocabulary size `Vs=30` for the encoder embedding and target vocabulary size `Vt=68` for the decoder output embedding and projection. Its corresponding formula is:

```text
P = Vs*E + Vt*E + 2*G*(E*H + H*H + H) + H*Vt + Vt
```

The table keeps the theoretical values from the assignment-style formula and compares them with the actual PyTorch model. The project settings are `E=128`, `H=256`, `V=68`, `Vs=30`, and `Vt=68`:

| Cell | Theoretical parameters | Actual PyTorch parameters |
| --- | ---: | ---: |
| RNN | 218,436 | 227,652 |
| GRU | 612,676 | 622,916 |
| LSTM | 809,796 | 820,548 |

Actual values use:

```python
sum(parameter.numel() for parameter in model.parameters())
```

The difference from the simplified theoretical count comes from separate source and target embeddings and PyTorch recurrent-layer bias tensors.

## RNN, GRU, and LSTM Experiment

`src/experiment.py` trains each cell type with the same data and configuration, selects the best validation state, evaluates it on the same test set, and records runtime. Results are saved to `results/experiments.csv`.

| Cell | Parameters | Best validation loss | Test character accuracy | Training time (seconds) |
| --- | ---: | ---: | ---: | ---: |
| RNN | 227,652 | 2.9964 | 0.1266 | 225.18 |
| GRU | 622,916 | 1.1321 | 0.6135 | 530.37 |
| LSTM | 820,548 | 1.0963 | 0.6422 | 827.23 |

Run the comparison with:

```bash
python3 -m src.experiment
```

Character accuracy is positional: matching characters are counted against the longer of the target and prediction strings, so insertions and deletions count as mismatches.

## Project Structure

```text
data/raw/       Original Aksharantar CSV files
src/            Dataset, vocabulary, model, training, evaluation, and analysis code
checkpoints/    Saved best model
results/        Predictions and experiment results
```
