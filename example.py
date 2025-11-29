"""
Example script showing how to use trained Word2Vec model
"""
import torch
from src.model import SkipGramModel, CBOWModel
from src.utils import find_similar_words, word_analogy, visualize_embeddings


def load_model(model_path, device='cpu'):
    """Load a trained Word2Vec model"""
    checkpoint = torch.load(model_path, map_location=device)

    vocab = checkpoint['vocab']
    embedding_dim = checkpoint['embedding_dim']
    model_type = checkpoint['model_type']

    # Create model
    if model_type == 'skipgram':
        model = SkipGramModel(len(vocab), embedding_dim)
    else:
        model = CBOWModel(len(vocab), embedding_dim)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, vocab


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model_path = "models/word2vec_model.pt"
    print(f"\nLoading model from {model_path}...")

    try:
        model, vocab = load_model(model_path, device)
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Embedding dimension: {model.embedding_dim}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Please train a model first using train.py")
        return

    # Create inverse vocabulary
    inverse_vocab = {idx: word for word, idx in vocab.items()}

    # Example 1: Find similar words
    print("\n=== Finding Similar Words ===")
    test_words = ['king', 'woman', 'computer', 'school', 'science']

    for word in test_words:
        if word in vocab:
            similar = find_similar_words(
                model, word, vocab, inverse_vocab, top_k=5, device=device
            )
            print(f"\nMost similar to '{word}':")
            for similar_word, score in similar:
                print(f"  {similar_word}: {score:.4f}")
        else:
            print(f"\nWord '{word}' not in vocabulary")

    # Example 2: Word analogies
    print("\n=== Word Analogies ===")
    analogies = [
        ('king', 'man', 'woman'),  # king - man + woman = queen
        ('paris', 'france', 'italy'),  # paris - france + italy = rome
        ('big', 'bigger', 'small'),  # big - bigger + small = smaller
    ]

    for word_a, word_b, word_c in analogies:
        if all(w in vocab for w in [word_a, word_b, word_c]):
            results = word_analogy(
                model, word_a, word_b, word_c,
                vocab, inverse_vocab, top_k=3, device=device
            )
            print(f"\n{word_a} - {word_b} + {word_c} =")
            for word, score in results:
                print(f"  {word}: {score:.4f}")
        else:
            print(f"\nSkipping analogy (words not in vocabulary)")

    # Example 3: Visualize embeddings
    print("\n=== Visualizing Embeddings ===")
    try:
        # Visualize top 100 most frequent words
        visualize_embeddings(
            model,
            vocab,
            inverse_vocab,
            num_words=100,
            method='tsne',
            device=device,
            save_path='embeddings_visualization.png'
        )
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == '__main__':
    main()
