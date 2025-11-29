"""
Text preprocessing utilities for Word2Vec
"""
import re
from collections import Counter
from typing import List, Tuple, Dict


def preprocess_text(text: str, lowercase: bool = True, remove_punctuation: bool = True) -> List[str]:
    """
    Preprocess text into tokens.

    Args:
        text: Input text string
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks

    Returns:
        List of tokens
    """
    if lowercase:
        text = text.lower()

    if remove_punctuation:
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Split into tokens and remove extra whitespace
    tokens = text.split()

    return tokens


def build_vocabulary(
    tokens: List[str],
    min_count: int = 5,
    max_vocab_size: int = None
) -> Tuple[Dict[str, int], Counter]:
    """
    Build vocabulary from tokens.

    Args:
        tokens: List of tokens
        min_count: Minimum frequency for a word to be included
        max_vocab_size: Maximum vocabulary size (keep most frequent)

    Returns:
        vocab: Dictionary mapping word to index
        word_counts: Counter object with word frequencies
    """
    # Count word frequencies
    word_counts = Counter(tokens)

    # Filter by minimum count
    word_counts = Counter({
        word: count for word, count in word_counts.items()
        if count >= min_count
    })

    # Limit vocabulary size if specified
    if max_vocab_size is not None and len(word_counts) > max_vocab_size:
        word_counts = Counter(dict(word_counts.most_common(max_vocab_size)))

    # Build vocabulary (reserve 0 for padding, 1 for unknown)
    vocab = {'<PAD>': 0, '<UNK>': 1}

    for idx, word in enumerate(word_counts.keys(), start=2):
        vocab[word] = idx

    return vocab, word_counts


def tokens_to_indices(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """
    Convert tokens to indices using vocabulary.

    Args:
        tokens: List of tokens
        vocab: Vocabulary dictionary

    Returns:
        List of indices
    """
    return [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]


def load_text_from_file(file_path: str) -> str:
    """
    Load text from a file.

    Args:
        file_path: Path to text file

    Returns:
        Text content as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def prepare_training_data(
    text_or_file: str,
    is_file: bool = False,
    min_count: int = 5,
    max_vocab_size: int = None,
    lowercase: bool = True,
    remove_punctuation: bool = True
) -> Tuple[List[str], Dict[str, int], Counter]:
    """
    Prepare training data from text or file.

    Args:
        text_or_file: Text string or file path
        is_file: Whether input is a file path
        min_count: Minimum word frequency
        max_vocab_size: Maximum vocabulary size
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation

    Returns:
        tokens: List of tokens
        vocab: Vocabulary dictionary
        word_counts: Word frequency counter
    """
    # Load text
    if is_file:
        text = load_text_from_file(text_or_file)
    else:
        text = text_or_file

    # Preprocess
    tokens = preprocess_text(text, lowercase, remove_punctuation)

    # Build vocabulary
    vocab, word_counts = build_vocabulary(tokens, min_count, max_vocab_size)

    # Filter tokens to only include vocabulary words
    tokens = [token for token in tokens if token in vocab]

    print(f"Total tokens: {len(tokens)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Unique words before filtering: {len(set(tokens))}")

    return tokens, vocab, word_counts
