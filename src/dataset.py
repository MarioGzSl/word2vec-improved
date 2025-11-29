"""
Dataset classes for Word2Vec training
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import Counter


class Word2VecDataset(Dataset):
    """
    Dataset for Word2Vec training with negative sampling.

    Args:
        text (list): List of tokens
        vocab (dict): Vocabulary mapping word to index
        word_counts (Counter): Word frequency counts
        window_size (int): Context window size
        num_negative_samples (int): Number of negative samples per positive example
        subsample_threshold (float): Threshold for subsampling frequent words
        mode (str): 'skipgram' or 'cbow'
    """

    def __init__(
        self,
        text,
        vocab,
        word_counts,
        window_size=5,
        num_negative_samples=5,
        subsample_threshold=1e-3,
        mode='skipgram'
    ):
        self.text = text
        self.vocab = vocab
        self.word_counts = word_counts
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples
        self.subsample_threshold = subsample_threshold
        self.mode = mode.lower()

        # Calculate total word count
        self.total_words = sum(word_counts.values())

        # Calculate word probabilities for negative sampling
        self._calculate_negative_sampling_probs()

        # Calculate subsampling probabilities
        self._calculate_subsample_probs()

        # Filter text based on subsampling
        self._subsample_text()

        # Pre-generate training pairs
        self._generate_pairs()

    def _calculate_negative_sampling_probs(self):
        """Calculate probabilities for negative sampling using frequency^0.75"""
        vocab_size = len(self.vocab)
        self.negative_probs = np.zeros(vocab_size)

        for word, idx in self.vocab.items():
            count = self.word_counts[word]
            self.negative_probs[idx] = count ** 0.75

        self.negative_probs /= self.negative_probs.sum()

    def _calculate_subsample_probs(self):
        """Calculate probabilities for subsampling frequent words"""
        self.subsample_probs = {}

        for word, count in self.word_counts.items():
            freq = count / self.total_words
            keep_prob = (np.sqrt(freq / self.subsample_threshold) + 1) * (
                self.subsample_threshold / freq
            )
            self.subsample_probs[word] = min(keep_prob, 1.0)

    def _subsample_text(self):
        """Subsample frequent words from text"""
        self.subsampled_text = []

        for word in self.text:
            if np.random.random() < self.subsample_probs.get(word, 1.0):
                self.subsampled_text.append(word)

    def _generate_pairs(self):
        """Generate (target, context) or (context, target) pairs"""
        self.pairs = []

        for idx, target_word in enumerate(self.subsampled_text):
            target_idx = self.vocab[target_word]

            # Random window size
            window = np.random.randint(1, self.window_size + 1)

            # Get context indices
            start = max(0, idx - window)
            end = min(len(self.subsampled_text), idx + window + 1)

            for context_idx in range(start, end):
                if context_idx == idx:
                    continue

                context_word = self.subsampled_text[context_idx]
                context_word_idx = self.vocab[context_word]

                if self.mode == 'skipgram':
                    self.pairs.append((target_idx, context_word_idx))
                else:  # cbow
                    # For CBOW, we'll collect all context words later
                    pass

        # For CBOW, generate different pairs
        if self.mode == 'cbow':
            self._generate_cbow_pairs()

    def _generate_cbow_pairs(self):
        """Generate CBOW pairs (context_words, target)"""
        self.pairs = []

        for idx, target_word in enumerate(self.subsampled_text):
            target_idx = self.vocab[target_word]

            # Random window size
            window = np.random.randint(1, self.window_size + 1)

            # Get context indices
            start = max(0, idx - window)
            end = min(len(self.subsampled_text), idx + window + 1)

            context_indices = []
            for context_idx in range(start, end):
                if context_idx != idx:
                    context_word = self.subsampled_text[context_idx]
                    context_indices.append(self.vocab[context_word])

            if context_indices:
                # Pad context to fixed size
                while len(context_indices) < 2 * self.window_size:
                    context_indices.append(0)  # padding index

                self.pairs.append((context_indices[:2 * self.window_size], target_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.mode == 'skipgram':
            target_idx, context_idx = self.pairs[idx]

            # Generate negative samples
            negative_samples = self._get_negative_samples(context_idx)

            return {
                'target': torch.tensor(target_idx, dtype=torch.long),
                'context': torch.tensor(context_idx, dtype=torch.long),
                'negatives': torch.tensor(negative_samples, dtype=torch.long)
            }
        else:  # cbow
            context_indices, target_idx = self.pairs[idx]

            # Generate negative samples
            negative_samples = self._get_negative_samples(target_idx)

            return {
                'context': torch.tensor(context_indices, dtype=torch.long),
                'target': torch.tensor(target_idx, dtype=torch.long),
                'negatives': torch.tensor(negative_samples, dtype=torch.long)
            }

    def _get_negative_samples(self, positive_idx):
        """Sample negative examples"""
        negative_samples = []

        while len(negative_samples) < self.num_negative_samples:
            neg_idx = np.random.choice(len(self.vocab), p=self.negative_probs)

            # Make sure it's not the positive sample
            if neg_idx != positive_idx:
                negative_samples.append(neg_idx)

        return negative_samples
