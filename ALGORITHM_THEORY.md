# Word2Vec Theory and Fundamentals

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Architectures](#architectures)
4. [Optimization Techniques](#optimization-techniques)
5. [Technical Implementation](#technical-implementation)
6. [Detailed Mathematics](#detailed-mathematics)
7. [References](#references)

---

## Introduction

Word2Vec is a family of deep learning algorithms designed to produce **dense vector representations of words** (word embeddings). These vectors capture semantic and syntactic relationships between words, enabling vector arithmetic operations to reflect linguistic analogies.

### Motivation

Traditional word representations (one-hot encoding) present two fundamental problems:

1. **High dimensionality**: A vocabulary of 100,000 words requires 100,000-dimensional vectors
2. **Lack of semantics**: They don't capture relationships between words (e.g., "king" and "queen" are as distant as "king" and "apple")

Word2Vec solves these problems through dense embeddings of reduced dimensionality (typically 100-300) that capture semantic meaning.

---

## Theoretical Foundations

### Distributional Hypothesis

> "You shall know a word by the company it keeps" - J.R. Firth (1957)

Word2Vec is based on the **distributional hypothesis**: words that appear in similar contexts tend to have similar meanings.

**Example:**
- "The **cat** sleeps on the couch"
- "The **dog** sleeps on the couch"

The words "cat" and "dog" share similar contexts, so their embeddings will be close in vector space.

### Learning Objective

Word2Vec learns representations by optimizing the ability to predict:
- **Skip-gram**: Predict context words given a target word
- **CBOW**: Predict target word given its context

---

## Architectures

### 1. Skip-gram

#### Concept

The Skip-gram model predicts surrounding context words given a central target word.

**Structure:**
```
Input: target word
    ↓
Embedding Layer
    ↓
Output: context word probabilities
```

#### How It Works

Given a text corpus, for each target word `w_t`:
1. Define a context window of size `c` (e.g., c=2)
2. For each context word `w_{t+j}` where `-c ≤ j ≤ c, j ≠ 0`:
   - Maximize `P(w_{t+j} | w_t)`

**Example with window c=2:**
```
Text: "The cat sleeps on the couch"
Target word: "sleeps"
Context: ["The", "cat", "on", "the"]

Training pairs:
(sleeps → The)
(sleeps → cat)
(sleeps → on)
(sleeps → the)
```

#### Objective Function

Maximize the average log probability:

```
L = (1/T) * Σ_{t=1}^{T} Σ_{-c≤j≤c, j≠0} log P(w_{t+j} | w_t)
```

Where:
- `T` = total number of words
- `c` = context window size

#### Advantages

- **Better for rare words**: Each target word generates multiple training examples
- **Captures complex syntactic and semantic relationships**
- **Asymmetric data**: Each (target, context) pair is treated independently

#### Disadvantages

- **Slower training**: Generates more training pairs
- **Requires more data**: Needs larger corpora to converge

---

### 2. CBOW (Continuous Bag of Words)

#### Concept

CBOW predicts the target word given the set of surrounding context words.

**Structure:**
```
Input: context words
    ↓
Embedding Layer (averaged)
    ↓
Output: target word probability
```

#### How It Works

For each target word `w_t`:
1. Collect all words in the context window
2. Average their embeddings
3. Predict `w_t` from the averaged embedding

**Example with window c=2:**
```
Text: "The cat sleeps on the couch"
Context: ["The", "cat", "on", "the"]
Target word: "sleeps"

Training pair:
([The, cat, on, the] → sleeps)
```

#### Objective Function

```
L = (1/T) * Σ_{t=1}^{T} log P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})
```

The prediction is based on the average of context embeddings:

```
h = (1/2c) * Σ_{-c≤j≤c, j≠0} v(w_{t+j})
```

Where `v(w)` is the embedding of word `w`.

#### Advantages

- **Faster training**: Fewer training pairs
- **Better for frequent words**: Smooths distributions by averaging context
- **Lower memory**: Fewer gradient updates

#### Disadvantages

- **Loses order information**: Averaging eliminates sequential structure
- **Less effective for rare words**

---

## Optimization Techniques

### 1. Negative Sampling

#### Original Problem

Computing softmax over the entire vocabulary is computationally prohibitive:

```
P(w_O | w_I) = exp(v'_{w_O}^T v_{w_I}) / Σ_{w=1}^{W} exp(v'_w^T v_{w_I})
```

For a vocabulary of 100,000 words, each update requires 100,000 exponential calculations.

#### Solution: Negative Sampling

Instead of normalizing over the entire vocabulary, the problem is reformulated as **binary classification**:
- Distinguish between real context words (positive) and randomly sampled words (negative)

#### Objective Function

For each (target, context) pair:

```
L = log σ(v'_{w_O}^T v_{w_I}) + Σ_{i=1}^{k} E_{w_i ~ P_n(w)} [log σ(-v'_{w_i}^T v_{w_I})]
```

Where:
- `σ(x) = 1/(1 + exp(-x))` is the sigmoid function
- `k` is the number of negative samples (typically 5-20)
- `P_n(w)` is the negative sampling distribution

#### Sampling Distribution

Word2Vec uses a **unigram distribution raised to the 3/4 power**:

```
P(w_i) = f(w_i)^{3/4} / Σ_j f(w_j)^{3/4}
```

Where `f(w_i)` is the word frequency.

**Justification:** The 3/4 exponent reduces bias towards very frequent words, giving more probability to less common words.

**Implementation (dataset.py:57-66):**
```python
def _calculate_negative_sampling_probs(self):
    vocab_size = len(self.vocab)
    self.negative_probs = np.zeros(vocab_size)

    for word, idx in self.vocab.items():
        count = self.word_counts[word]
        self.negative_probs[idx] = count ** 0.75

    self.negative_probs /= self.negative_probs.sum()
```

#### Advantages

- **Complexity reduction**: From O(W) to O(k) per example
- **Faster training**: 100-1000x faster than full softmax
- **Higher quality embeddings**: Focus on important distinctions

---

### 2. Subsampling of Frequent Words

#### Problem

Very frequent words ("the", "of", "that") appear in many contexts but provide little semantic information. Training with all their occurrences:
1. Wastes computational resources
2. Dilutes the signal from informative words

#### Solution

Discard frequent words with probability:

```
P(w_i) = 1 - sqrt(t / f(w_i))
```

Where:
- `f(w_i)` is the relative frequency of the word
- `t` is a threshold (typically 10^{-3} to 10^{-5})

**Interpretation:** Words with `f(w_i) > t` are discarded with higher probability.

**Implementation (dataset.py:68-77):**
```python
def _calculate_subsample_probs(self):
    self.subsample_probs = {}

    for word, count in self.word_counts.items():
        freq = count / self.total_words
        keep_prob = (np.sqrt(freq / self.subsample_threshold) + 1) * (
            self.subsample_threshold / freq
        )
        self.subsample_probs[word] = min(keep_prob, 1.0)
```

#### Effects

1. **Training acceleration**: 2-10x faster
2. **Better embeddings**: Greater emphasis on informative words
3. **Balancing**: Reduces dominance of frequent words

---

## Technical Implementation

### Embedding Architecture

#### Two Embedding Matrices

Unlike naive implementations, Word2Vec uses **two separate matrices**:

1. **Input Embeddings** (`embeddings`): Representations of target words
2. **Context Embeddings** (`context_embeddings`): Representations of context words

**Implementation (model.py:24-36):**
```python
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
```

**Justification:**
- **Optimization flexibility**: Different learning rates
- **Model capacity**: Doubles learnable parameters
- **Faster convergence**: Avoids conflicts between input/output roles

**Final vectors:** Typically only `embeddings` is used, or the average of both matrices.

---

### Weight Initialization

Embeddings are initialized uniformly in the range `[-r, r]`:

```
r = 0.5 / embedding_dim
```

**Implementation (model.py:41-45):**
```python
def _init_weights(self):
    init_range = 0.5 / self.embedding_dim
    self.embeddings.weight.data.uniform_(-init_range, init_range)
    self.context_embeddings.weight.data.uniform_(-init_range, init_range)
```

**Reason:** Small values prevent saturation of sigmoids/tanh during initialization.

---

### Forward Pass

#### Skip-gram (model.py:47-78)

```python
def forward(self, target_words, context_words, negative_samples=None):
    # Get embeddings
    target_embeds = self.embeddings(target_words)  # (batch, embed_dim)
    context_embeds = self.context_embeddings(context_words)  # (batch, embed_dim)

    # Positive samples score
    pos_score = torch.sum(target_embeds * context_embeds, dim=1)
    pos_loss = F.logsigmoid(pos_score)

    # Negative sampling loss
    if negative_samples is not None:
        neg_embeds = self.context_embeddings(negative_samples)  # (batch, k, embed_dim)
        neg_score = torch.bmm(
            neg_embeds,
            target_embeds.unsqueeze(2)
        ).squeeze(2)  # (batch, k)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

    return -pos_loss.mean()
```

**Steps:**
1. Get embeddings for target and context
2. Compute dot product: `score = v_target · v_context`
3. Apply log-sigmoid for positive samples: `log σ(score)`
4. For negative samples: `Σ log σ(-score_i)`
5. Final loss: `-mean(pos_loss + neg_loss)`

---

#### CBOW (model.py:122-156)

```python
def forward(self, context_words, target_word, negative_samples=None):
    # Average context embeddings
    context_embeds = self.embeddings(context_words)  # (batch, context_size, embed_dim)
    context_mean = context_embeds.mean(dim=1)  # (batch, embed_dim)

    # Target embedding
    target_embeds = self.context_embeddings(target_word)  # (batch, embed_dim)

    # Positive score
    pos_score = torch.sum(context_mean * target_embeds, dim=1)
    pos_loss = F.logsigmoid(pos_score)

    # Negative sampling loss
    if negative_samples is not None:
        neg_embeds = self.context_embeddings(negative_samples)
        neg_score = torch.bmm(
            neg_embeds,
            context_mean.unsqueeze(2)
        ).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

    return -pos_loss.mean()
```

**Key difference:** Averages context embeddings before dot product.

---

### Training Pair Generation

#### Dynamic Window

Instead of a fixed window, a **random window** of size `[1, window_size]` is used:

**Implementation (dataset.py:94-95):**
```python
# Random window size
window = np.random.randint(1, self.window_size + 1)
```

**Advantage:** Gives more weight to nearby words, better capturing local syntactic dependencies.

#### Skip-gram Pairs (dataset.py:87-113)

```python
def _generate_pairs(self):
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
```

#### CBOW Pairs (dataset.py:118-143)

For CBOW, **all context words** are collected and padded if necessary:

```python
def _generate_cbow_pairs(self):
    self.pairs = []

    for idx, target_word in enumerate(self.subsampled_text):
        target_idx = self.vocab[target_word]
        window = np.random.randint(1, self.window_size + 1)

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
```

---

## Detailed Mathematics

### Negative Sampling Derivation

#### Original Objective

Maximize:
```
log P(w_O | w_I) = log (exp(v'_{w_O}^T v_{w_I}) / Σ_{w=1}^{W} exp(v'_w^T v_{w_I}))
```

**Problem:** Sum over W words in denominator.

#### Reformulation

We consider the problem as binary classification:
- Random variable `D = 1` if the pair (w_I, w_O) comes from the corpus
- `D = 0` if it comes from the noise distribution

**Objective:** Maximize `P(D=1 | w_O, w_I)` for real pairs.

Using logistic regression:
```
P(D=1 | w_O, w_I) = σ(v'_{w_O}^T v_{w_I})
```

#### Final Loss Function

For one positive pair and k negatives:

```
L = log σ(v'_{w_O}^T v_{w_I}) + Σ_{i=1}^{k} log σ(-v'_{w_i}^T v_{w_I})
```

Where `w_i ~ P_n(w)` are negative samples.

**In PyTorch:**
```python
pos_loss = F.logsigmoid(pos_score)  # log σ(score)
neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # Σ log σ(-score)
total_loss = -(pos_loss + neg_loss).mean()
```

---

### Gradients

#### Gradient for Positive Sample

```
∂L/∂v_{w_I} = (σ(v'_{w_O}^T v_{w_I}) - 1) · v'_{w_O}
∂L/∂v'_{w_O} = (σ(v'_{w_O}^T v_{w_I}) - 1) · v_{w_I}
```

#### Gradient for Negative Samples

For each negative sample `w_i`:
```
∂L/∂v_{w_I} += σ(v'_{w_i}^T v_{w_I}) · v'_{w_i}
∂L/∂v'_{w_i} = σ(v'_{w_i}^T v_{w_I}) · v_{w_I}
```

**Interpretation:**
- If model assigns high probability to negative sample: large gradient
- Update pushes embeddings to separate positives from negatives

---

### Computational Complexity

#### Skip-gram

Per target word:
- **Without negative sampling:** O(W · d) [W = vocabulary size, d = dimension]
- **With negative sampling:** O(k · d) [k = negative samples]

For corpus of T tokens and window c:
- **Total:** O(T · c · k · d)

#### CBOW

Per target word:
- **Without negative sampling:** O(W · d)
- **With negative sampling:** O(k · d + 2c · d) [2c for context averaging]

For corpus of T tokens:
- **Total:** O(T · k · d)

**CBOW advantage:** Factor ~c fewer operations (window doesn't multiply)

---

## Emergent Properties

### Vector Analogies

Word2Vec embeddings capture analogical relationships:

```
v(king) - v(man) + v(woman) ≈ v(queen)
v(Madrid) - v(Spain) + v(France) ≈ v(Paris)
```

**Implementation (utils.py - word_analogy):**
```python
def word_analogy(model, word_a, word_b, word_c, vocab, inverse_vocab):
    # king - man + woman = ?
    vec_a = model.get_embedding(torch.tensor([vocab[word_a]]))
    vec_b = model.get_embedding(torch.tensor([vocab[word_b]]))
    vec_c = model.get_embedding(torch.tensor([vocab[word_c]]))

    # Vector operation
    result_vec = vec_a - vec_b + vec_c

    # Find nearest word
    # ...
```

### Semantic Clusters

Semantically similar words form clusters in embedding space:
- Colors: {red, blue, green, yellow}
- Countries: {Spain, France, Italy, Germany}
- Motion verbs: {run, walk, jump}

**Measure:** Cosine similarity
```
sim(w1, w2) = (v(w1) · v(w2)) / (||v(w1)|| · ||v(w2)||)
```

---

## Skip-gram vs CBOW Comparison

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| **Prediction** | Context given target | Target given context |
| **Speed** | Slower | Faster |
| **Data needed** | More | Less |
| **Rare words** | Better | Worse |
| **Frequent words** | Worse | Better |
| **Training pairs** | Many (T·c) | Few (T) |
| **Recommended use** | Small corpora, syntax | Large corpora, semantics |

---

## Best Practices

### Hyperparameters

#### Embedding Dimension
- **Small (50-100):** Specific tasks, limited vocabulary
- **Medium (128-300):** General use, good balance
- **Large (500-1000):** Complex tasks, massive vocabulary

#### Context Window
- **Small (2-3):** Syntactic relationships (POS tagging)
- **Medium (5-7):** Syntax-semantics balance
- **Large (10-15):** Broad semantic relationships

#### Negative Samples
- **Small corpus:** k = 5-10
- **Large corpus:** k = 2-5
- More samples → better quality but slower

#### Learning Rate
- **Skip-gram:** 0.025 (with decay)
- **CBOW:** 0.05 (with decay)
- Linear decay: `lr = lr_initial * (1 - epoch/total_epochs)`

### Preprocessing

1. **Lowercasing:** Reduces vocabulary, merges variants
2. **Remove punctuation:** Focus on meaningful words
3. **Min frequency:** Filter very rare words (min_count=5)
4. **Max vocab:** Limit to most frequent words (50k-100k)
5. **Subsampling:** t = 1e-3 to 1e-5 for balance

---

## Limitations and Extensions

### Limitations

1. **Fixed context:** Doesn't capture long-range dependencies
2. **Polysemy:** One word = one vector (doesn't handle multiple meanings)
3. **OOV (Out of Vocabulary):** Cannot represent unseen words
4. **Order information:** CBOW loses order when averaging

### Extensions

1. **FastText (Meta):** Represents words as bag of character n-grams
   - Handles OOV via subwords
   - Better for rich morphology

2. **GloVe (Stanford):** Matrix factorization of co-occurrence counts
   - Leverages global statistics
   - More stable training

3. **ELMo, BERT:** Contextual embeddings
   - Different embeddings per context
   - Solves polysemy

4. **Paragraph Vector:** Extends to documents
   - Learns paragraph/document representations

---

## Practical Applications

### 1. Semantic Search
```python
query_embedding = model.get_embedding(query)
doc_embeddings = [model.get_embedding(doc) for doc in documents]
similarities = [cosine_similarity(query_embedding, doc_emb)
                for doc_emb in doc_embeddings]
```

### 2. Text Classification
Use pre-trained embeddings as features for classifiers:
```python
text_embedding = np.mean([model.get_embedding(word) for word in text], axis=0)
prediction = classifier.predict(text_embedding)
```

### 3. Recommendation
Recommend similar items based on embeddings:
```python
item_embedding = model.get_embedding(item)
similar_items = find_k_nearest(item_embedding, all_item_embeddings, k=10)
```

### 4. Machine Translation
Align embedding spaces between languages for translation.

---

## References

### Foundational Papers

1. **Mikolov et al. (2013)** - "Efficient Estimation of Word Representations in Vector Space"
   - Introduces Skip-gram and CBOW
   - [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)

2. **Mikolov et al. (2013)** - "Distributed Representations of Words and Phrases and their Compositionality"
   - Negative sampling and subsampling
   - [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)

3. **Mikolov et al. (2013)** - "Linguistic Regularities in Continuous Space Word Representations"
   - Vector analogies
   - [NAACL-HLT 2013](https://aclanthology.org/N13-1090/)

### Additional Resources

- **TensorFlow Tutorial:** [Word2Vec Tutorial](https://www.tensorflow.org/tutorials/text/word2vec)
- **Paper Review:** [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- **Original Google Code:** [word2vec C implementation](https://code.google.com/archive/p/word2vec/)

---

## Appendix: Reference Code

### Similarity Calculation

```python
import torch.nn.functional as F

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return F.cosine_similarity(vec1, vec2, dim=0).item()

def find_similar_words(model, word, vocab, inverse_vocab, top_k=5):
    """Find k most similar words"""
    if word not in vocab:
        return []

    word_idx = torch.tensor([vocab[word]])
    word_embedding = model.get_embedding(word_idx)

    all_embeddings = model.embeddings.weight.data

    # Cosine similarity with all words
    similarities = F.cosine_similarity(
        word_embedding,
        all_embeddings,
        dim=1
    )

    # Top-k (excluding the word itself)
    top_k_indices = similarities.topk(top_k + 1)[1][1:]

    results = []
    for idx in top_k_indices:
        similar_word = inverse_vocab[idx.item()]
        similarity = similarities[idx].item()
        results.append((similar_word, similarity))

    return results
```

### Visualization with t-SNE

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(model, vocab, num_words=100):
    """Visualize embeddings using t-SNE"""
    embeddings = model.embeddings.weight.data.cpu().numpy()[:num_words]

    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

    # Annotate words
    words = list(vocab.keys())[:num_words]
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.title('Word Embeddings Visualization (t-SNE)')
    plt.show()
```

---

**Document created for the Word2Vec PyTorch Implementation project**
**Author:** Technical documentation generated from source code
**Last updated:** 2025
