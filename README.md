# Word2Vec PyTorch Implementation

A clean, modular, and efficient implementation of Word2Vec in PyTorch with both Skip-gram and CBOW architectures.

## Features

- **Two Model Architectures**: Skip-gram and CBOW (Continuous Bag of Words)
- **Negative Sampling**: Efficient training with negative sampling
- **Subsampling**: Automatic subsampling of frequent words
- **Flexible Preprocessing**: Configurable text preprocessing pipeline
- **Comprehensive Utilities**: Word similarity, analogies, and visualizations
- **Command-line Interface**: Easy training with extensive configuration options
- **Modular Design**: Clean separation of concerns for easy customization

## Installation

```bash
git clone https://github.com/MarioGzSl/word2vec-improved.git
cd word2vec-improved
pip install -r requirements.txt
```

## Quick Start

### Training a Model

Train a Skip-gram model on Wikipedia data:

```bash
python train.py --model skipgram --epochs 10 --embedding-dim 128
```

Train a CBOW model from a text file:

```bash
python train.py --model cbow --dataset path/to/text.txt --epochs 20
```

### Using a Trained Model

```python
from src.model import SkipGramModel
from src.utils import find_similar_words
import torch

# Load model
checkpoint = torch.load('models/word2vec_model.pt')
model = SkipGramModel(len(checkpoint['vocab']), checkpoint['embedding_dim'])
model.load_state_dict(checkpoint['model_state_dict'])

# Find similar words
vocab = checkpoint['vocab']
inverse_vocab = {idx: word for word, idx in vocab.items()}
similar = find_similar_words(model, 'king', vocab, inverse_vocab, top_k=5)
print(similar)
```

See `example.py` for more usage examples.

## Architecture

### Skip-gram Model

Predicts context words given a target word. Better for smaller datasets and rare words.

```
Target Word → Embedding → Context Words
```

### CBOW Model

Predicts target word given context words. Faster training and better for frequent words.

```
Context Words → Average Embedding → Target Word
```

## Project Structure

```
word2vec-improved/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── model.py              # Skip-gram and CBOW models
│   ├── dataset.py            # Dataset with negative sampling
│   ├── preprocessing.py      # Text preprocessing utilities
│   └── utils.py              # Evaluation and visualization
├── data/                     # Data directory (gitignored)
├── models/                   # Saved models (gitignored)
├── notebooks/                # Jupyter notebooks
├── tests/                    # Unit tests
├── train.py                  # Training script
├── example.py                # Usage examples
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Command-line Arguments

### Data Arguments

- `--dataset`: Dataset to use (default: 'wikipedia' or path to text file)
- `--num-articles`: Number of Wikipedia articles (default: 1000)
- `--min-count`: Minimum word frequency (default: 5)
- `--max-vocab-size`: Maximum vocabulary size (default: None)

### Model Arguments

- `--model`: Model type ('skipgram' or 'cbow', default: 'skipgram')
- `--embedding-dim`: Embedding dimension (default: 128)
- `--window-size`: Context window size (default: 5)
- `--num-negative-samples`: Negative samples per example (default: 5)
- `--subsample-threshold`: Subsampling threshold (default: 1e-3)

### Training Arguments

- `--epochs`: Number of epochs (default: 10)
- `--batch-size`: Batch size (default: 512)
- `--learning-rate`: Learning rate (default: 0.001)
- `--num-workers`: Data loading workers (default: 4)

### Output Arguments

- `--output-dir`: Output directory (default: 'models')
- `--save-every`: Save checkpoint frequency (default: 5)

## Key Improvements Over Original

1. **Better Architecture**: Separate input and context embeddings for better performance
2. **Negative Sampling**: Proper implementation with frequency-based sampling
3. **Subsampling**: Automatic handling of frequent words
4. **Modular Code**: Clean separation into logical modules
5. **Type Hints**: Better code documentation and IDE support
6. **Comprehensive Utils**: Word analogies, visualizations, and more
7. **Efficient Training**: Optimized data pipeline with DataLoader
8. **Flexible Configuration**: Extensive command-line options
9. **Model Saving/Loading**: Standard checkpoint format
10. **Documentation**: Comprehensive README and code comments

## Examples

### Find Similar Words

```python
similar_words = find_similar_words(
    model, 'computer', vocab, inverse_vocab, top_k=10
)
# Output: [('laptop', 0.87), ('pc', 0.82), ...]
```

### Word Analogies

```python
# king - man + woman = ?
results = word_analogy(
    model, 'king', 'man', 'woman', vocab, inverse_vocab
)
# Output: [('queen', 0.89), ('princess', 0.75), ...]
```

### Visualize Embeddings

```python
visualize_embeddings(
    model, vocab, inverse_vocab,
    num_words=100,
    method='tsne',
    save_path='embeddings.png'
)
```

## Performance Tips

1. **GPU Training**: Use CUDA for faster training (automatic if available)
2. **Batch Size**: Increase for better GPU utilization (512-2048)
3. **Negative Samples**: 5-20 samples work well in practice
4. **Window Size**: 5-10 for most applications
5. **Embedding Dimension**: 100-300 for most tasks
6. **Subsampling**: Helps with frequent words (threshold ~1e-3 to 1e-5)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{word2vec_pytorch,
  author = {MarioGzSl},
  title = {Word2Vec PyTorch Implementation},
  year = {2023},
  url = {https://github.com/MarioGzSl/word2vec-improved}
}
```

## References

- Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
