# Q3 — Aksharantar Character-Level Seq2Seq

## Project Goal

The objective of this project is to build a character-level sequence-to-sequence model that converts a Romanized Hindi word into its corresponding Devanagari form.

Example:

```text
Input:  ghar
Output: घर
```

The model will use an encoder-decoder architecture with character embeddings and a configurable recurrent neural network.

---

# Step 1 — Project Setup

## Objective

Create a clean project structure before implementing the neural network.

Current structure:

```text
Q3-aksharantar-seq2seq/
│
├── data/
│   ├── raw/
│   │   ├── hin_train.csv
│   │   ├── hin_valid.csv
│   │   └── hin_test.csv
│   │
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── vocabulary.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── seq2seq.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── notebooks/
│   └── q3_experiments.ipynb
│
├── checkpoints/
│
├── results/
│   ├── predictions/
│   └── figures/
│
├── README.md
├── requirements.txt
└── .gitignore
```

The original CSV files are kept inside `data/raw/` and should not be modified.

The purpose of the directories is to keep data, source code, experiments, trained models, and generated results separate.

---

# Step 2 — Inspect the Dataset

## Objective

Understand the actual structure of the Aksharantar Hindi data before performing preprocessing.

The project uses three files:

```text
hin_train.csv
hin_valid.csv
hin_test.csv
```

They represent:

* training examples
* validation examples
* test examples

Each example contains a Romanized form and its corresponding Devanagari form.

Example:

```text
Romanized       Devanagari
shastragaar     शस्त्रागार
```

### Checks to perform

Before creating the model, we will inspect:

1. Number of examples in each split.
2. Column names.
3. A few sample rows.
4. Missing values.
5. Duplicate pairs.
6. Character composition of both columns.
7. Minimum and maximum word lengths.
8. Whether unexpected characters are present.

### Important rule

The three dataset splits should remain separate.

The training set will be used to learn the model.

The validation set will be used to monitor model performance while developing the system.

The test set will be reserved for the final evaluation.

---

# Step 3 — Build Character Vocabularies

## Objective

Convert characters into integer IDs so that they can be processed by PyTorch.

A neural network cannot directly consume a string such as:

```text
ghar
```

Therefore, every character needs an integer representation.

For example:

```text
g → 12
h → 7
a → 3
r → 18
```

The exact IDs will be generated from the vocabulary rather than manually assigned.

## Separate vocabularies

We will maintain two vocabularies.

### Source vocabulary

This represents characters appearing in the Romanized input.

```text
Romanized character
        ↓
      ID
```

### Target vocabulary

This represents characters appearing in the Devanagari output.

```text
Devanagari character
        ↓
      ID
```

Keeping them separate makes the encoder and decoder vocabularies independent.

## Special symbols

The target sequence needs markers that tell the decoder when generation starts and ends.

We will use:

```text
<PAD>   Padding
<SOS>   Start of sequence
<EOS>   End of sequence
<UNK>   Unknown character
```

These symbols will be included in the appropriate vocabulary representation.

## Vocabulary source

The vocabulary will be constructed using the training data rather than using the validation or test examples to define the character inventory.

---

# Step 4 — Dataset and DataLoader

## Objective

Create a PyTorch dataset that converts each word pair into numerical sequences.

For example:

```text
Romanized:
ghar

Target:
घर
```

will become something conceptually similar to:

```text
Source:
[ID(g), ID(h), ID(a), ID(r)]

Target:
[<SOS>, ID(घ), ID(र), <EOS>]
```

The actual integer values will depend on the vocabulary created in Step 3.

## Why a Dataset is needed

The dataset class will be responsible for:

* reading the prepared examples
* converting characters to IDs
* adding required sequence markers
* returning source and target sequences

## Why a DataLoader is needed

Training examples have different lengths. The DataLoader will allow us to organize examples into batches.

Because sequence lengths can vary, we will later decide how padding and batch collation should be handled.

The preprocessing code should not alter the original files in `data/raw/`.

---

# Step 5 — Encoder

## Objective

Implement the first half of the sequence-to-sequence model.

The encoder reads the Romanized input character by character.

The basic flow is:

```text
Character ID
     ↓
Embedding
     ↓
RNN / LSTM / GRU
     ↓
Hidden state
```

For an input such as:

```text
ghar
```

the encoder processes:

```text
g → h → a → r
```

At every character, the recurrent cell updates its internal state.

The final encoder state will contain information accumulated from the complete input sequence.

Conceptually:

```text
g ──┐
h ──┤
a ──┤──> Encoder ──> final hidden state
r ──┘
```

This final state will later be passed to the decoder as its starting state.

## Configurability

The encoder should not be written specifically for one recurrent cell.

The implementation will eventually support:

```text
RNN
LSTM
GRU
```

The following settings should also be configurable:

```text
embedding dimension
hidden-state dimension
number of encoder layers
```

These settings will be controlled through the project configuration rather than being hard-coded inside the encoder.

---

# Development Order

The first five stages therefore form this pipeline:

```text
Project setup
      ↓
Dataset inspection
      ↓
Character vocabularies
      ↓
Dataset + DataLoader
      ↓
Encoder
```

We will not implement the decoder until the encoder and input pipeline have been verified.

---

# Reference

The project is based on the sequence-to-sequence requirements in the technical aptitude assignment and the PyTorch sequence-to-sequence tutorial supplied as a reference for the assignment.

The implementation and documentation in this repository will be written specifically for this project rather than copied from the reference implementation.

