# 📚 Dataset: Frankenstein by Mary Shelley

## Overview

This project uses **Frankenstein; Or, The Modern Prometheus** by Mary Shelley, sourced from [Project Gutenberg](https://www.gutenberg.org/).

The text is in the **public domain** (originally published 1818).

---

## How to Download

### Option 1: Direct Download from Project Gutenberg

1. Visit: https://www.gutenberg.org/ebooks/84
2. Click **"Plain Text UTF-8"** format
3. Save the file as `frankenstein.txt` in this directory (`datasets/`)

### Option 2: Using wget (Command Line)

```bash
cd datasets/
wget https://www.gutenberg.org/files/84/84-0.txt -O frankenstein.txt
```

### Option 3: Using curl (Command Line)

```bash
cd datasets/
curl https://www.gutenberg.org/files/84/84-0.txt -o frankenstein.txt
```

### Option 4: Using Python

```python
import urllib.request

url = "https://www.gutenberg.org/files/84/84-0.txt"
urllib.request.urlretrieve(url, "datasets/frankenstein.txt")
print("Downloaded frankenstein.txt")
```

---

## File Verification

After downloading, verify the file:

```bash
cd datasets/
ls -lh frankenstein.txt
```

Expected:
- **Size**: ~440 KB (450,000+ characters)
- **Encoding**: UTF-8 plain text

### Quick Check

```python
with open('datasets/frankenstein.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    print(f"Total characters: {len(text)}")
    print(f"First 100 chars: {text[:100]}")
```

You should see:
```
Total characters: ~450000
First 100 chars: The Project Gutenberg eBook of Frankenstein; Or, The Modern Prometheus...
```

---

## What the Notebooks Use

The notebooks train on **Letter 1 only** (a subset of the full text):

```python
with open('datasets/frankenstein.txt', 'r', encoding='utf-8') as f:
    frankenstein = f.read()

first_letter_text = frankenstein[1380:8230]  # ~6,850 characters
```

**Why just Letter 1?**
- Fast training (~seconds per epoch)
- Quick iteration for learning
- Sufficient to demonstrate concepts

**Extension:** Remove the slice to train on the full novel (requires ~10x longer training).

---

## Data Characteristics

### Full Novel
- **Length**: ~450,000 characters
- **Unique Characters**: ~70-80 (letters, punctuation, spaces, newlines)
- **Structure**: Letters, chapters, dialogue
- **Style**: Gothic, formal, 19th-century English

### Letter 1 (Training Subset)
- **Length**: 6,850 characters
- **Unique Characters**: ~50-60
- **Content**: Opening letter from Robert Walton to his sister Margaret
- **Style**: Epistolary, first-person, descriptive

---

## Disclaimer

⚠️ **Historical Text Notice**

*Frankenstein* was published in 1818 and reflects the language, social norms, and perspectives of that era. The text may contain:
- Archaic phrasing and vocabulary
- 19th-century attitudes
- Dated references

This project uses the text **solely for educational purposes** to learn natural language processing and sequence modeling techniques. Generated outputs are purely for demonstrating machine learning capabilities.

---

## License & Attribution

### Source
**Title**: Frankenstein; Or, The Modern Prometheus  
**Author**: Mary Shelley  
**Published**: 1818  
**Source**: [Project Gutenberg](https://www.gutenberg.org/ebooks/84)

### License
**Public Domain** — This work is in the public domain in the United States and most other countries. Project Gutenberg's license allows free distribution for non-commercial purposes.

### Attribution
When sharing this project, please credit:
- **Author**: Mary Shelley
- **Source**: Project Gutenberg

---

## Troubleshooting

### File Not Found Error
If notebooks can't find `frankenstein.txt`:

1. Check file location:
   ```bash
   ls datasets/frankenstein.txt
   ```

2. Verify notebooks use correct path:
   ```python
   with open('../datasets/frankenstein.txt', 'r', encoding='utf-8') as f:
   ```

### Encoding Issues
If you see strange characters, ensure UTF-8 encoding:

```python
with open('datasets/frankenstein.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()
```

### Download Fails
- Try a different download method (wget, curl, browser)
- Check internet connection
- Visit Project Gutenberg directly and manually save

---

## Alternative Datasets

Want to try other texts? Project Gutenberg has thousands of public domain books:

**Similar Gothic/Classic Literature:**
- *Dracula* by Bram Stoker: https://www.gutenberg.org/ebooks/345
- *Jekyll and Hyde* by Robert Louis Stevenson: https://www.gutenberg.org/ebooks/43
- *Wuthering Heights* by Emily Brontë: https://www.gutenberg.org/ebooks/768

**Adjust the slicing indices** in notebooks to match the structure of different texts.

---

## Questions?

If you encounter issues with the dataset:
1. Check the [Project Gutenberg FAQ](https://www.gutenberg.org/help/faq.html)
2. Open an issue in this repository
3. Email: cisco@periospot.com

---

**Happy modeling!** 🚀

