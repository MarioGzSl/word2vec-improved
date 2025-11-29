"""
Word2Vec Implementation in PyTorch

A clean, modular implementation of Word2Vec with Skip-gram and CBOW models.
"""

from .model import SkipGramModel, CBOWModel
from .dataset import Word2VecDataset
from .preprocessing import prepare_training_data, build_vocabulary, preprocess_text
from .utils import (
    find_similar_words,
    word_analogy,
    get_word_embedding,
    visualize_embeddings,
    save_embeddings,
    load_embeddings
)

__version__ = '1.0.0'
__all__ = [
    'SkipGramModel',
    'CBOWModel',
    'Word2VecDataset',
    'prepare_training_data',
    'build_vocabulary',
    'preprocess_text',
    'find_similar_words',
    'word_analogy',
    'get_word_embedding',
    'visualize_embeddings',
    'save_embeddings',
    'load_embeddings'
]
