"""
Utility functions for Word2Vec evaluation and visualization
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def get_word_embedding(model, word: str, vocab: Dict[str, int], device='cpu'):
    """
    Get embedding vector for a word.

    Args:
        model: Trained Word2Vec model
        word: Word to get embedding for
        vocab: Vocabulary dictionary
        device: Device to use

    Returns:
        Embedding tensor or None if word not in vocabulary
    """
    if word not in vocab:
        return None

    model.eval()
    word_idx = torch.tensor([vocab[word]], dtype=torch.long).to(device)

    with torch.no_grad():
        embedding = model.get_embedding(word_idx)

    return embedding


def find_similar_words(
    model,
    word: str,
    vocab: Dict[str, int],
    inverse_vocab: Dict[int, str],
    top_k: int = 10,
    device='cpu'
) -> List[Tuple[str, float]]:
    """
    Find most similar words using cosine similarity.

    Args:
        model: Trained Word2Vec model
        word: Query word
        vocab: Vocabulary dictionary
        inverse_vocab: Inverse vocabulary (index to word)
        top_k: Number of similar words to return
        device: Device to use

    Returns:
        List of (word, similarity_score) tuples
    """
    word_embedding = get_word_embedding(model, word, vocab, device)

    if word_embedding is None:
        return []

    model.eval()

    # Get all embeddings
    all_indices = torch.arange(len(vocab), dtype=torch.long).to(device)
    with torch.no_grad():
        all_embeddings = model.get_embedding(all_indices)

    # Calculate cosine similarities
    word_embedding = word_embedding.squeeze(0)
    similarities = torch.nn.functional.cosine_similarity(
        word_embedding.unsqueeze(0),
        all_embeddings
    )

    # Get top k
    top_k_values, top_k_indices = similarities.topk(top_k + 1)

    # Convert to list (skip first one as it's the word itself)
    similar_words = []
    for idx, score in zip(top_k_indices[1:].cpu().numpy(), top_k_values[1:].cpu().numpy()):
        similar_words.append((inverse_vocab.get(idx, '<UNK>'), float(score)))

    return similar_words


def word_analogy(
    model,
    word_a: str,
    word_b: str,
    word_c: str,
    vocab: Dict[str, int],
    inverse_vocab: Dict[int, str],
    top_k: int = 5,
    device='cpu'
) -> List[Tuple[str, float]]:
    """
    Solve word analogies: word_a is to word_b as word_c is to ?
    Example: king - man + woman = queen

    Args:
        model: Trained Word2Vec model
        word_a, word_b, word_c: Words for analogy
        vocab: Vocabulary dictionary
        inverse_vocab: Inverse vocabulary
        top_k: Number of results to return
        device: Device to use

    Returns:
        List of (word, score) tuples
    """
    # Get embeddings
    emb_a = get_word_embedding(model, word_a, vocab, device)
    emb_b = get_word_embedding(model, word_b, vocab, device)
    emb_c = get_word_embedding(model, word_c, vocab, device)

    if emb_a is None or emb_b is None or emb_c is None:
        return []

    # Calculate target embedding: b - a + c
    target_embedding = emb_b - emb_a + emb_c
    target_embedding = target_embedding.squeeze(0)

    # Get all embeddings
    all_indices = torch.arange(len(vocab), dtype=torch.long).to(device)
    with torch.no_grad():
        all_embeddings = model.get_embedding(all_indices)

    # Calculate cosine similarities
    similarities = torch.nn.functional.cosine_similarity(
        target_embedding.unsqueeze(0),
        all_embeddings
    )

    # Get top k (excluding input words)
    exclude_indices = {vocab.get(w, -1) for w in [word_a, word_b, word_c]}

    top_k_values, top_k_indices = similarities.topk(top_k + len(exclude_indices))

    # Filter and convert to list
    results = []
    for idx, score in zip(top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()):
        if idx not in exclude_indices and len(results) < top_k:
            results.append((inverse_vocab.get(idx, '<UNK>'), float(score)))

    return results


def visualize_embeddings(
    model,
    vocab: Dict[str, int],
    inverse_vocab: Dict[int, str],
    words: List[str] = None,
    num_words: int = 100,
    method: str = 'tsne',
    device='cpu',
    save_path: str = None
):
    """
    Visualize word embeddings in 2D using dimensionality reduction.

    Args:
        model: Trained Word2Vec model
        vocab: Vocabulary dictionary
        inverse_vocab: Inverse vocabulary
        words: Specific words to visualize (if None, use most frequent)
        num_words: Number of words to visualize
        method: Dimensionality reduction method ('tsne' or 'pca')
        device: Device to use
        save_path: Path to save figure (if None, display only)
    """
    model.eval()

    # Select words to visualize
    if words is None:
        # Use first num_words (most frequent after special tokens)
        word_indices = list(range(2, min(num_words + 2, len(vocab))))
    else:
        word_indices = [vocab[w] for w in words if w in vocab]

    # Get embeddings
    indices_tensor = torch.tensor(word_indices, dtype=torch.long).to(device)
    with torch.no_grad():
        embeddings = model.get_embedding(indices_tensor).cpu().numpy()

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(word_indices) - 1))
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)

    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    # Add labels
    for i, idx in enumerate(word_indices):
        word = inverse_vocab.get(idx, '<UNK>')
        plt.annotate(
            word,
            xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 2),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )

    plt.title(f'Word2Vec Embeddings Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def save_embeddings(model, vocab: Dict[str, int], file_path: str, device='cpu'):
    """
    Save word embeddings to a text file in word2vec format.

    Args:
        model: Trained Word2Vec model
        vocab: Vocabulary dictionary
        file_path: Path to save embeddings
        device: Device to use
    """
    model.eval()

    with open(file_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"{len(vocab)} {model.embedding_dim}\n")

        # Write embeddings
        for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
            idx_tensor = torch.tensor([idx], dtype=torch.long).to(device)
            with torch.no_grad():
                embedding = model.get_embedding(idx_tensor).squeeze(0).cpu().numpy()

            embedding_str = ' '.join(map(str, embedding))
            f.write(f"{word} {embedding_str}\n")

    print(f"Embeddings saved to {file_path}")


def load_embeddings(file_path: str) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load word embeddings from a text file.

    Args:
        file_path: Path to embeddings file

    Returns:
        vocab: Vocabulary dictionary
        embeddings: Numpy array of embeddings
    """
    vocab = {}
    embeddings_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        # Read header
        vocab_size, embedding_dim = map(int, f.readline().split())

        # Read embeddings
        for idx, line in enumerate(f):
            parts = line.strip().split()
            word = parts[0]
            embedding = np.array([float(x) for x in parts[1:]])

            vocab[word] = idx
            embeddings_list.append(embedding)

    embeddings = np.array(embeddings_list)

    print(f"Loaded {len(vocab)} embeddings of dimension {embedding_dim}")

    return vocab, embeddings
