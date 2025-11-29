"""
Word2Vec Models: Skip-gram and CBOW implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """
    Skip-gram model: predicts context words given a target word.

    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings
        padding_idx (int): Index for padding token (default: 0)
    """

    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Input embeddings (word vectors we want to learn)
        self.embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # Output embeddings (context vectors)
        self.context_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights with uniform distribution"""
        init_range = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, target_words, context_words, negative_samples=None):
        """
        Forward pass for Skip-gram model.

        Args:
            target_words: Tensor of shape (batch_size,)
            context_words: Tensor of shape (batch_size,)
            negative_samples: Tensor of shape (batch_size, num_negatives)

        Returns:
            loss: Scalar loss value
        """
        # Get embeddings
        target_embeds = self.embeddings(target_words)  # (batch_size, embed_dim)
        context_embeds = self.context_embeddings(context_words)  # (batch_size, embed_dim)

        # Positive samples score
        pos_score = torch.sum(target_embeds * context_embeds, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Negative sampling loss (if provided)
        if negative_samples is not None:
            neg_embeds = self.context_embeddings(negative_samples)  # (batch_size, num_neg, embed_dim)
            neg_score = torch.bmm(
                neg_embeds,
                target_embeds.unsqueeze(2)
            ).squeeze(2)  # (batch_size, num_neg)
            neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

            return -(pos_loss + neg_loss).mean()

        return -pos_loss.mean()

    def get_embedding(self, word_idx):
        """Get embedding for a word index"""
        return self.embeddings(word_idx)


class CBOWModel(nn.Module):
    """
    Continuous Bag of Words (CBOW) model: predicts target word given context words.

    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings
        padding_idx (int): Index for padding token (default: 0)
    """

    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        super(CBOWModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Embeddings
        self.embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        self.context_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights with uniform distribution"""
        init_range = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, context_words, target_word, negative_samples=None):
        """
        Forward pass for CBOW model.

        Args:
            context_words: Tensor of shape (batch_size, context_size)
            target_word: Tensor of shape (batch_size,)
            negative_samples: Tensor of shape (batch_size, num_negatives)

        Returns:
            loss: Scalar loss value
        """
        # Average context embeddings
        context_embeds = self.embeddings(context_words)  # (batch_size, context_size, embed_dim)
        context_mean = context_embeds.mean(dim=1)  # (batch_size, embed_dim)

        # Target embedding
        target_embeds = self.context_embeddings(target_word)  # (batch_size, embed_dim)

        # Positive score
        pos_score = torch.sum(context_mean * target_embeds, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Negative sampling loss
        if negative_samples is not None:
            neg_embeds = self.context_embeddings(negative_samples)  # (batch_size, num_neg, embed_dim)
            neg_score = torch.bmm(
                neg_embeds,
                context_mean.unsqueeze(2)
            ).squeeze(2)  # (batch_size, num_neg)
            neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

            return -(pos_loss + neg_loss).mean()

        return -pos_loss.mean()

    def get_embedding(self, word_idx):
        """Get embedding for a word index"""
        return self.embeddings(word_idx)
