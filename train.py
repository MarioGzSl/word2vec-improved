"""
Training script for Word2Vec models
"""
import argparse
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import SkipGramModel, CBOWModel
from src.dataset import Word2VecDataset
from src.preprocessing import prepare_training_data
from src.utils import save_embeddings, find_similar_words


def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    print("\n=== Preparing Data ===")
    if args.dataset == 'wikipedia':
        try:
            from datasets import load_dataset
            print("Loading Wikipedia dataset...")
            dataset_wikipedia = load_dataset("wikipedia", "20220301.en", split='train')

            # Sample articles
            text_tokens = []
            for i in range(min(args.num_articles, len(dataset_wikipedia))):
                article = dataset_wikipedia[i]['text']
                from src.preprocessing import preprocess_text
                text_tokens.extend(preprocess_text(
                    article,
                    lowercase=not args.no_lowercase,
                    remove_punctuation=not args.keep_punctuation
                ))

            # Build vocabulary
            from src.preprocessing import build_vocabulary
            vocab, word_counts = build_vocabulary(
                text_tokens,
                min_count=args.min_count,
                max_vocab_size=args.max_vocab_size
            )

            # Filter tokens
            tokens = [token for token in text_tokens if token in vocab]

        except ImportError:
            print("datasets library not found. Please install it or use a text file.")
            return
    else:
        # Load from file
        tokens, vocab, word_counts = prepare_training_data(
            args.dataset,
            is_file=True,
            min_count=args.min_count,
            max_vocab_size=args.max_vocab_size,
            lowercase=not args.no_lowercase,
            remove_punctuation=not args.keep_punctuation
        )

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total tokens: {len(tokens)}")

    # Create dataset
    print("\n=== Creating Dataset ===")
    dataset = Word2VecDataset(
        text=tokens,
        vocab=vocab,
        word_counts=word_counts,
        window_size=args.window_size,
        num_negative_samples=args.num_negative_samples,
        subsample_threshold=args.subsample_threshold,
        mode=args.model
    )

    print(f"Training pairs: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Create model
    print("\n=== Creating Model ===")
    vocab_size = len(vocab)

    if args.model == 'skipgram':
        model = SkipGramModel(vocab_size, args.embedding_dim).to(device)
    else:
        model = CBOWModel(vocab_size, args.embedding_dim).to(device)

    print(f"Model: {args.model}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("\n=== Training ===")
    model.train()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            target = batch['target'].to(device)
            context = batch['context'].to(device)
            negatives = batch['negatives'].to(device)

            optimizer.zero_grad()

            if args.model == 'skipgram':
                loss = model(target, context, negatives)
            else:
                loss = model(context, target, negatives)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab': vocab,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    print("\n=== Saving Model ===")
    final_model_path = os.path.join(args.output_dir, "word2vec_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'embedding_dim': args.embedding_dim,
        'model_type': args.model
    }, final_model_path)
    print(f"Model saved to {final_model_path}")

    # Save embeddings in text format
    embeddings_path = os.path.join(args.output_dir, "embeddings.txt")
    save_embeddings(model, vocab, embeddings_path, device)

    # Test with example words
    if args.test_words:
        print("\n=== Testing ===")
        inverse_vocab = {idx: word for word, idx in vocab.items()}

        for word in args.test_words:
            similar = find_similar_words(
                model, word, vocab, inverse_vocab, top_k=5, device=device
            )
            if similar:
                print(f"\nMost similar to '{word}':")
                for similar_word, score in similar:
                    print(f"  {similar_word}: {score:.4f}")
            else:
                print(f"\nWord '{word}' not in vocabulary")

    print("\n=== Training Complete ===")


def main():
    parser = argparse.ArgumentParser(description='Train Word2Vec model')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='wikipedia',
                        help='Dataset to use (wikipedia or path to text file)')
    parser.add_argument('--num-articles', type=int, default=1000,
                        help='Number of Wikipedia articles to use')
    parser.add_argument('--min-count', type=int, default=5,
                        help='Minimum word frequency')
    parser.add_argument('--max-vocab-size', type=int, default=None,
                        help='Maximum vocabulary size')
    parser.add_argument('--no-lowercase', action='store_true',
                        help='Do not convert to lowercase')
    parser.add_argument('--keep-punctuation', action='store_true',
                        help='Keep punctuation marks')

    # Model arguments
    parser.add_argument('--model', type=str, default='skipgram',
                        choices=['skipgram', 'cbow'],
                        help='Model architecture')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Context window size')
    parser.add_argument('--num-negative-samples', type=int, default=5,
                        help='Number of negative samples')
    parser.add_argument('--subsample-threshold', type=float, default=1e-3,
                        help='Subsampling threshold for frequent words')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Do not use CUDA even if available')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--test-words', nargs='+', default=['king', 'woman', 'computer'],
                        help='Words to test similarity with')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    train(args)


if __name__ == '__main__':
    main()
