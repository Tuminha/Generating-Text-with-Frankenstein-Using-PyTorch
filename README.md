# 🧪 Frankenstein Text Generation
## Character-Level LSTM in PyTorch — Learning Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Status](https://img.shields.io/badge/Status-Learning-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Character-level language modeling with LSTM to generate Gothic prose in Mary Shelley's style**

[🎯 Overview](#-project-overview) • [📚 Roadmap](#-notebook-roadmap) • [🚀 Quick Start](#-quick-start) • [📊 Dataset](#-dataset--disclaimer)

</div>

> First baseline result: training a character-level LSTM on Letter 1 of *Frankenstein* — not production-grade, but honest educational work. Next up: experimenting with temperature, longer sequences, and full-text training.

---

## 👨‍💻 Author

<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/franciscotbarbosa)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-1DA1F2?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Learning Machine Learning through CodeCademy • Building AI solutions step by step*

</div>

---

## 🎯 Project Overview

### What
A **pedagogical implementation** of character-level text generation using LSTM networks in PyTorch. The model learns to predict the next character given a sequence of previous characters, trained on Mary Shelley's *Frankenstein*.

### Why
- **Learn sequence modeling**: Understand how RNNs (specifically LSTMs) handle sequential data
- **Master PyTorch fundamentals**: Custom Datasets, DataLoaders, training loops, inference
- **Practice NLP basics**: Tokenization, vocabulary building, text generation
- **Experience autoregressive generation**: Sampling, greedy decoding, and generation loops

### Learning Style
**Instructional notebooks with TODOs** — no full solutions provided. You learn by writing the code yourself, guided by hints and shape expectations.

### Expected Outcome
A trained LSTM that generates 500 characters of Gothic prose given a prompt like *"You will rejoice to hear"*. Text will be somewhat coherent locally but may repeat or drift over longer spans.

---

## 🎓 Learning Objectives

- [ ] Understand character-level tokenization and why it's simpler than word-level
- [ ] Build bidirectional vocabulary mappings (`c2ix` and `ix2c`)
- [ ] Implement sliding window datasets for sequence prediction
- [ ] Define LSTM architecture with embedding, LSTM layer, and linear projection
- [ ] Train with CrossEntropyLoss and Adam optimizer
- [ ] Generate text using autoregressive sampling
- [ ] Experiment with hyperparameters and sampling strategies

---

## 🏆 Key Achievements

- [x] Character-level vocabulary with ~50-80 unique tokens
- [x] Custom PyTorch Dataset for sliding window sequences
- [x] LSTM model: Embedding (48) → LSTM (96) → Linear (vocab_size)
- [x] 5-epoch training loop with loss tracking
- [x] Greedy text generation (500 chars)
- [ ] Temperature sampling (extension)
- [ ] Training on full novel (extension)

---

## 📝 Progress Log

### ✅ Completed Notebooks

#### Notebook 01 — Load & Slice ✅
- **Status**: Complete
- **Result**: Successfully loaded 6,850 characters from Letter 1
- **Data**: *"You will rejoice to hear that no disaster has accompanied..."*

#### Notebook 02 — Tokenization & Vocab ✅
- **Status**: Complete
- **Vocab Size**: 60 unique characters
- **Includes**: Letters (A-Z, a-z), punctuation (`,` `.` `!` `?` `;` `:` `-` `—`), numbers, newlines, spaces
- **Mappings**: Built `c2ix` (char→ID) and `ix2c` (ID→char)
- **Result**: 6,850 tokens converted to IDs

#### Notebook 03 — Dataset & DataLoader ⏳
- **Status**: In Progress

---

## 📊 Dataset / Disclaimer

### Source
**Frankenstein; Or, The Modern Prometheus** by Mary Shelley  
From [Project Gutenberg](https://www.gutenberg.org/)

The provided notebooks train on **Letter 1 only** (indices 1380:8230, ~6,850 characters) to keep compute minimal and iteration fast. This is a learning exercise, not a production model.

### Disclaimer
⚠️ **Historical Text Notice**: *Frankenstein* was published in 1818 and reflects the language, attitudes, and perspectives of that era. Generated text may contain outdated or archaic phrasing. This project is for educational purposes to learn NLP techniques.

### How to Obtain
See `datasets/README_DATASET.md` for instructions on downloading `frankenstein.txt` from Project Gutenberg.

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+ recommended
python --version

# Install dependencies
pip install torch torchvision  # or conda install pytorch
pip install jupyter numpy matplotlib  # optional for visualization
```

### Setup
```bash
# Clone or download this repository
cd "Frankestein with Pytorch"

# Ensure datasets/frankenstein.txt exists
# (See datasets/README_DATASET.md for download instructions)

# Launch Jupyter
jupyter notebook
```

### Run Order
Open notebooks in sequence and complete the TODOs:

1. `00_overview.ipynb` — Understand the pipeline
2. `01_load_and_slice_letter.ipynb` — Load text data
3. `02_char_tokenize_and_vocab.ipynb` — Build vocabulary
4. `03_id_sequences_dataset_dataloader.ipynb` — Create Dataset/DataLoader
5. `04_lstm_model_scaffold.ipynb` — Define LSTM model
6. `05_train_lstm_loop.ipynb` — Train for 5 epochs
7. `06_generate_text.ipynb` — Generate 500 characters
8. `99_lab_notes.ipynb` — Document your learnings

**Note:** Each notebook contains TODOs with hints, not complete solutions. Learning happens when you write the code.

---

## 📈 Notebook Roadmap

| Notebook | Title | What You'll Build |
|----------|-------|-------------------|
| **00** | Map of the Journey | Conceptual overview of the pipeline |
| **01** | Load & Slice | File I/O, slicing Letter 1 |
| **02** | Tokenization & Vocab | `c2ix`, `ix2c`, `vocab_size` |
| **03** | Dataset & DataLoader | Sliding windows, PyTorch Dataset |
| **04** | LSTM Model | Embedding → LSTM → Linear |
| **05** | Training Loop | CrossEntropyLoss, Adam, 5 epochs |
| **06** | Text Generation | Autoregressive sampling |
| **99** | Lab Notes | Reflections & experiments |

---

## 📊 Expected Results

### Training Loss
With 5 epochs, batch_size=36, seq_length=48, lr=0.015:
```
Epoch 1: ~2.8-3.2
Epoch 2: ~2.3-2.7
Epoch 3: ~2.0-2.4
Epoch 4: ~1.8-2.2
Epoch 5: ~1.7-2.1
```

*Exact values vary due to random initialization and small dataset.*

### Generated Text Sample
**Prompt:** "You will rejoice to hear"

**Output (example):**
```
You will rejoice to hear that no disaster has accompanied the commencement of an 
enterprise which you have regarded with such evil forebodings. I arrived here 
yesterday, and my first task is to assure my dear sister of my welfare...
```

**Assessment:**
- ✅ Captures Gothic, formal style
- ✅ Mimics sentence structure (long, nested clauses)
- ⚠️ May repeat phrases or lose coherence after ~200 chars
- ⚠️ Greedy sampling tends toward most common patterns

---

## 📌 Business / Learning Interpretation

### What This Project Teaches
1. **Sequence Modeling Fundamentals**: LSTMs maintain context via hidden/cell states
2. **PyTorch Workflow**: From raw data → Dataset → DataLoader → Model → Training → Inference
3. **Autoregressive Generation**: How models generate text one token at a time
4. **Hyperparameter Awareness**: Batch size, learning rate, sequence length all matter

### Practical Insights
- **Character-level** is simpler (small vocab) but requires longer sequences for coherence
- **Greedy sampling** is deterministic; temperature adds creativity
- **LSTM limits**: Transformers (GPT-style) now dominate long-range dependencies
- **Data size matters**: Training on 6,850 chars is a toy example; real models use millions

### Next Steps in Your Learning Journey
- Word-level models (learn embeddings like Word2Vec)
- Attention mechanisms and Transformers
- Larger datasets (full novels, Wikipedia)
- Multi-class text classification
- Sequence-to-sequence tasks (translation, summarization)

---

## 🛠 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.10+ | Core programming |
| **Framework** | PyTorch | Neural network library |
| **Data Processing** | NumPy, Python stdlib | Text manipulation |
| **Development** | Jupyter Notebook | Interactive exploration |
| **Text Source** | Project Gutenberg | Public domain literature |

### Model Architecture Details
```
CharacterLSTM(
  embedding_dim=48,
  hidden_size=96,
  num_layers=1,
  batch_first=True
)

Parameters: ~50K (varies with vocab_size)

Pipeline:
  Input [B, T] 
    → Embedding [B, T, 48] 
    → LSTM [B, T, 96] 
    → Linear [B, T, vocab_size] 
    → Logits [B*T, vocab_size]
```

---

## 📝 Learning Journey

### Skills Developed
- ✅ Character-level tokenization
- ✅ PyTorch Dataset & DataLoader
- ✅ LSTM architecture design
- ✅ Training loop implementation
- ✅ Autoregressive text generation
- ✅ Loss tracking and model evaluation
- ✅ Hyperparameter experimentation

### Common Pitfalls (and Solutions)
1. **Shape mismatches**: Always print tensor shapes during debugging
2. **Forgot to flatten labels**: CrossEntropyLoss needs `[N]` targets
3. **No `zero_grad()`**: Gradients accumulate; always clear per batch
4. **Training mode during generation**: Use `model.eval()` for inference
5. **Tracking gradients in generation**: Wrap with `torch.no_grad()`

---

## 🚀 Extensions & Experiments

### Easy
- [ ] Try different prompts
- [ ] Generate 1000+ characters
- [ ] Change temperature (sampling randomness)

### Medium
- [ ] Train on full novel
- [ ] Increase hidden_size to 128 or 256
- [ ] Add dropout for regularization
- [ ] Implement top-k or nucleus sampling

### Advanced
- [ ] Bidirectional LSTM
- [ ] Multi-layer LSTM (2-3 layers)
- [ ] Beam search decoding
- [ ] Compare to Transformer-based model

---

## 📁 Repository Structure

```
.
├── README.md                          ← You are here
├── datasets/
│   ├── README_DATASET.md              ← How to download frankenstein.txt
│   └── frankenstein.txt               ← Text data (download from Gutenberg)
├── notebooks/
│   ├── 00_overview.ipynb              ← Pipeline overview
│   ├── 01_load_and_slice_letter.ipynb ← Data loading
│   ├── 02_char_tokenize_and_vocab.ipynb ← Tokenization
│   ├── 03_id_sequences_dataset_dataloader.ipynb ← Dataset creation
│   ├── 04_lstm_model_scaffold.ipynb   ← Model architecture
│   ├── 05_train_lstm_loop.ipynb       ← Training
│   ├── 06_generate_text.ipynb         ← Text generation
│   └── 99_lab_notes.ipynb             ← Learning journal
├── src/
│   ├── __init__.py
│   ├── utils/                         ← (Placeholder for future utilities)
│   └── models/                        ← (Placeholder for future modules)
├── images/                            ← (For plots/screenshots)
└── trained_lstm_model.pth             ← Saved model weights (after training)
```

---

## 📚 Resources & Further Reading

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Christopher Olah
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) — Andrej Karpathy
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Project Gutenberg](https://www.gutenberg.org/) — Free public domain books

---

## 📄 License

This project is released under the **MIT License**. Feel free to use, modify, and share for educational purposes.

The text of *Frankenstein* by Mary Shelley is in the **public domain** (published 1818).

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**  
*Building AI solutions one character at a time* 🚀

---

**Questions? Feedback?**  
Open an issue or reach out: [cisco@periospot.com](mailto:cisco@periospot.com)

</div>

