# Teoría y Fundamentos de Word2Vec

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Arquitecturas](#arquitecturas)
4. [Técnicas de Optimización](#técnicas-de-optimización)
5. [Implementación Técnica](#implementación-técnica)
6. [Matemáticas Detalladas](#matemáticas-detalladas)
7. [Referencias](#referencias)

---

## Introducción

Word2Vec es una familia de algoritmos de aprendizaje profundo diseñados para producir **representaciones vectoriales densas de palabras** (word embeddings). Estos vectores capturan relaciones semánticas y sintácticas entre palabras, permitiendo que operaciones aritméticas vectoriales reflejen analogías lingüísticas.

### Motivación

Las representaciones tradicionales de palabras (one-hot encoding) presentan dos problemas fundamentales:

1. **Alta dimensionalidad**: Un vocabulario de 100,000 palabras requiere vectores de 100,000 dimensiones
2. **Falta de semántica**: No capturan relaciones entre palabras (e.g., "rey" y "reina" son igualmente distantes que "rey" y "manzana")

Word2Vec resuelve estos problemas mediante embeddings densos de dimensionalidad reducida (típicamente 100-300) que capturan significado semántico.

---

## Fundamentos Teóricos

### Hipótesis Distribucional

> "Una palabra se caracteriza por la compañía que mantiene" - J.R. Firth (1957)

Word2Vec se basa en la **hipótesis distribucional**: palabras que aparecen en contextos similares tienden a tener significados similares.

**Ejemplo:**
- "El **gato** duerme en el sofá"
- "El **perro** duerme en el sofá"

Las palabras "gato" y "perro" comparten contextos similares, por lo que sus embeddings serán cercanos en el espacio vectorial.

### Objetivo de Aprendizaje

Word2Vec aprende representaciones optimizando la capacidad de predecir:
- **Skip-gram**: Predecir palabras de contexto dada una palabra objetivo
- **CBOW**: Predecir palabra objetivo dado su contexto

---

## Arquitecturas

### 1. Skip-gram

#### Concepto

El modelo Skip-gram predice las palabras del contexto circundante dada una palabra objetivo central.

**Estructura:**
```
Entrada: palabra objetivo (target)
    ↓
Embedding Layer
    ↓
Salida: probabilidades de palabras de contexto
```

#### Funcionamiento

Dado un corpus de texto, para cada palabra objetivo `w_t`:
1. Definir ventana de contexto de tamaño `c` (e.g., c=2)
2. Para cada palabra de contexto `w_{t+j}` donde `-c ≤ j ≤ c, j ≠ 0`:
   - Maximizar `P(w_{t+j} | w_t)`

**Ejemplo con ventana c=2:**
```
Texto: "El gato duerme en el sofá"
Palabra objetivo: "duerme"
Contexto: ["El", "gato", "en", "el"]

Pares de entrenamiento:
(duerme → El)
(duerme → gato)
(duerme → en)
(duerme → el)
```

#### Función Objetivo

Maximizar la probabilidad logarítmica promedio:

```
L = (1/T) * Σ_{t=1}^{T} Σ_{-c≤j≤c, j≠0} log P(w_{t+j} | w_t)
```

Donde:
- `T` = número total de palabras
- `c` = tamaño de ventana de contexto

#### Ventajas

- **Mejor para palabras raras**: Cada palabra objetivo genera múltiples ejemplos de entrenamiento
- **Captura relaciones sintácticas y semánticas complejas**
- **Datos asimétricos**: Cada par (target, context) se trata independientemente

#### Desventajas

- **Entrenamiento más lento**: Genera más pares de entrenamiento
- **Requiere más datos**: Necesita corpus más grandes para convergir

---

### 2. CBOW (Continuous Bag of Words)

#### Concepto

CBOW predice la palabra objetivo dado el conjunto de palabras del contexto circundante.

**Estructura:**
```
Entrada: palabras de contexto
    ↓
Embedding Layer (promediado)
    ↓
Salida: probabilidad de palabra objetivo
```

#### Funcionamiento

Para cada palabra objetivo `w_t`:
1. Recolectar todas las palabras en la ventana de contexto
2. Promediar sus embeddings
3. Predecir `w_t` desde el embedding promedio

**Ejemplo con ventana c=2:**
```
Texto: "El gato duerme en el sofá"
Contexto: ["El", "gato", "en", "el"]
Palabra objetivo: "duerme"

Par de entrenamiento:
([El, gato, en, el] → duerme)
```

#### Función Objetivo

```
L = (1/T) * Σ_{t=1}^{T} log P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})
```

La predicción se basa en el promedio de embeddings del contexto:

```
h = (1/2c) * Σ_{-c≤j≤c, j≠0} v(w_{t+j})
```

Donde `v(w)` es el embedding de la palabra `w`.

#### Ventajas

- **Entrenamiento más rápido**: Menos pares de entrenamiento
- **Mejor para palabras frecuentes**: Suaviza distribuciones al promediar contexto
- **Menor memoria**: Menos actualizaciones de gradiente

#### Desventajas

- **Pierde información de orden**: El promedio elimina la estructura secuencial
- **Menos efectivo para palabras raras**

---

## Técnicas de Optimización

### 1. Negative Sampling

#### Problema Original

Calcular la softmax sobre todo el vocabulario es computacionalmente prohibitivo:

```
P(w_O | w_I) = exp(v'_{w_O}^T v_{w_I}) / Σ_{w=1}^{W} exp(v'_w^T v_{w_I})
```

Para un vocabulario de 100,000 palabras, cada actualización requiere 100,000 cálculos exponenciales.

#### Solución: Negative Sampling

En lugar de normalizar sobre todo el vocabulario, el problema se reformula como **clasificación binaria**:
- Distinguir entre palabras del contexto real (positivas) y palabras muestreadas aleatoriamente (negativas)

#### Función Objetivo

Para cada par (target, context):

```
L = log σ(v'_{w_O}^T v_{w_I}) + Σ_{i=1}^{k} E_{w_i ~ P_n(w)} [log σ(-v'_{w_i}^T v_{w_I})]
```

Donde:
- `σ(x) = 1/(1 + exp(-x))` es la función sigmoide
- `k` es el número de muestras negativas (típicamente 5-20)
- `P_n(w)` es la distribución de muestreo negativo

#### Distribución de Muestreo

Word2Vec usa una distribución **unigram elevada a 3/4**:

```
P(w_i) = f(w_i)^{3/4} / Σ_j f(w_j)^{3/4}
```

Donde `f(w_i)` es la frecuencia de la palabra.

**Justificación:** El exponente 3/4 reduce el sesgo hacia palabras muy frecuentes, dando más probabilidad a palabras menos comunes.

**Implementación (dataset.py:57-66):**
```python
def _calculate_negative_sampling_probs(self):
    vocab_size = len(self.vocab)
    self.negative_probs = np.zeros(vocab_size)

    for word, idx in self.vocab.items():
        count = self.word_counts[word]
        self.negative_probs[idx] = count ** 0.75

    self.negative_probs /= self.negative_probs.sum()
```

#### Ventajas

- **Reducción de complejidad**: De O(W) a O(k) por ejemplo
- **Entrenamiento más rápido**: 100-1000x más rápido que softmax completo
- **Embeddings de mayor calidad**: Enfoque en distinciones importantes

---

### 2. Subsampling de Palabras Frecuentes

#### Problema

Palabras muy frecuentes ("el", "de", "que") aparecen en muchos contextos pero aportan poca información semántica. Entrenar con todas sus ocurrencias:
1. Consume recursos computacionales
2. Diluye la señal de palabras informativas

#### Solución

Descartar palabras frecuentes con probabilidad:

```
P(w_i) = 1 - sqrt(t / f(w_i))
```

Donde:
- `f(w_i)` es la frecuencia relativa de la palabra
- `t` es un umbral (típicamente 10^{-3} a 10^{-5})

**Interpretación:** Palabras con `f(w_i) > t` se descartan con mayor probabilidad.

**Implementación (dataset.py:68-77):**
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

#### Efectos

1. **Aceleración del entrenamiento**: 2-10x más rápido
2. **Mejores embeddings**: Mayor énfasis en palabras informativas
3. **Balanceo**: Reduce dominancia de palabras frecuentes

---

## Implementación Técnica

### Arquitectura de Embeddings

#### Dos Matrices de Embeddings

A diferencia de implementaciones ingenuas, Word2Vec usa **dos matrices separadas**:

1. **Input Embeddings** (`embeddings`): Representaciones de palabras objetivo
2. **Context Embeddings** (`context_embeddings`): Representaciones de palabras de contexto

**Implementación (model.py:24-36):**
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

**Justificación:**
- **Flexibilidad de optimización**: Diferentes tasas de aprendizaje
- **Capacidad del modelo**: Duplica parámetros aprendibles
- **Convergencia más rápida**: Evita conflictos entre roles input/output

**Vectores finales:** Típicamente se usa solo `embeddings`, o el promedio de ambas matrices.

---

### Inicialización de Pesos

Los embeddings se inicializan uniformemente en el rango `[-r, r]`:

```
r = 0.5 / embedding_dim
```

**Implementación (model.py:41-45):**
```python
def _init_weights(self):
    init_range = 0.5 / self.embedding_dim
    self.embeddings.weight.data.uniform_(-init_range, init_range)
    self.context_embeddings.weight.data.uniform_(-init_range, init_range)
```

**Razón:** Valores pequeños previenen saturación de sigmoides/tanh durante inicialización.

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

**Pasos:**
1. Obtener embeddings de target y context
2. Calcular producto punto: `score = v_target · v_context`
3. Aplicar log-sigmoide para muestras positivas: `log σ(score)`
4. Para muestras negativas: `Σ log σ(-score_i)`
5. Loss final: `-mean(pos_loss + neg_loss)`

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

**Diferencia clave:** Promedia embeddings de contexto antes del producto punto.

---

### Generación de Pares de Entrenamiento

#### Ventana Dinámica

En lugar de una ventana fija, se usa **ventana aleatoria** de tamaño `[1, window_size]`:

**Implementación (dataset.py:94-95):**
```python
# Random window size
window = np.random.randint(1, self.window_size + 1)
```

**Ventaja:** Da más peso a palabras cercanas, capturando mejor dependencias sintácticas locales.

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

Para CBOW, se recolectan **todas las palabras de contexto** y se padding si necesario:

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

## Matemáticas Detalladas

### Derivación de Negative Sampling

#### Objetivo Original

Maximizar:
```
log P(w_O | w_I) = log (exp(v'_{w_O}^T v_{w_I}) / Σ_{w=1}^{W} exp(v'_w^T v_{w_I}))
```

**Problema:** Suma sobre W palabras en denominador.

#### Reformulación

Consideramos el problema como clasificación binaria:
- Variable aleatoria `D = 1` si el par (w_I, w_O) viene del corpus
- `D = 0` si viene de la distribución de ruido

**Objetivo:** Maximizar `P(D=1 | w_O, w_I)` para pares reales.

Usando regresión logística:
```
P(D=1 | w_O, w_I) = σ(v'_{w_O}^T v_{w_I})
```

#### Función de Pérdida Final

Para un par positivo y k negativos:

```
L = log σ(v'_{w_O}^T v_{w_I}) + Σ_{i=1}^{k} log σ(-v'_{w_i}^T v_{w_I})
```

Donde `w_i ~ P_n(w)` son muestras negativas.

**En PyTorch:**
```python
pos_loss = F.logsigmoid(pos_score)  # log σ(score)
neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # Σ log σ(-score)
total_loss = -(pos_loss + neg_loss).mean()
```

---

### Gradientes

#### Gradiente para Muestra Positiva

```
∂L/∂v_{w_I} = (σ(v'_{w_O}^T v_{w_I}) - 1) · v'_{w_O}
∂L/∂v'_{w_O} = (σ(v'_{w_O}^T v_{w_I}) - 1) · v_{w_I}
```

#### Gradiente para Muestras Negativas

Para cada muestra negativa `w_i`:
```
∂L/∂v_{w_I} += σ(v'_{w_i}^T v_{w_I}) · v'_{w_i}
∂L/∂v'_{w_i} = σ(v'_{w_i}^T v_{w_I}) · v_{w_I}
```

**Interpretación:**
- Si el modelo asigna alta probabilidad a muestra negativa: gradiente grande
- Actualización empuja embeddings para separar positivos de negativos

---

### Complejidad Computacional

#### Skip-gram

Por palabra objetivo:
- **Sin negative sampling:** O(W · d) [W = tamaño de vocabulario, d = dimensión]
- **Con negative sampling:** O(k · d) [k = muestras negativas]

Para corpus de T tokens y ventana c:
- **Total:** O(T · c · k · d)

#### CBOW

Por palabra objetivo:
- **Sin negative sampling:** O(W · d)
- **Con negative sampling:** O(k · d + 2c · d) [2c por promediado de contexto]

Para corpus de T tokens:
- **Total:** O(T · k · d)

**Ventaja de CBOW:** Factor ~c menos operaciones (ventana no multiplica)

---

## Propiedades Emergentes

### Analogías Vectoriales

Los embeddings de Word2Vec capturan relaciones analógicas:

```
v(rey) - v(hombre) + v(mujer) ≈ v(reina)
v(Madrid) - v(España) + v(Francia) ≈ v(París)
```

**Implementación (utils.py - word_analogy):**
```python
def word_analogy(model, word_a, word_b, word_c, vocab, inverse_vocab):
    # rey - hombre + mujer = ?
    vec_a = model.get_embedding(torch.tensor([vocab[word_a]]))
    vec_b = model.get_embedding(torch.tensor([vocab[word_b]]))
    vec_c = model.get_embedding(torch.tensor([vocab[word_c]]))

    # Operación vectorial
    result_vec = vec_a - vec_b + vec_c

    # Buscar palabra más cercana
    # ...
```

### Clusters Semánticos

Palabras semánticamente similares forman clusters en el espacio de embeddings:
- Colores: {rojo, azul, verde, amarillo}
- Países: {España, Francia, Italia, Alemania}
- Verbos de movimiento: {correr, caminar, saltar}

**Medida:** Similitud coseno
```
sim(w1, w2) = (v(w1) · v(w2)) / (||v(w1)|| · ||v(w2)||)
```

---

## Comparación Skip-gram vs CBOW

| Aspecto | Skip-gram | CBOW |
|---------|-----------|------|
| **Predicción** | Contexto dado target | Target dado contexto |
| **Velocidad** | Más lento | Más rápido |
| **Datos necesarios** | Más | Menos |
| **Palabras raras** | Mejor | Peor |
| **Palabras frecuentes** | Peor | Mejor |
| **Pares de entrenamiento** | Muchos (T·c) | Pocos (T) |
| **Uso recomendado** | Corpus pequeños, sintaxis | Corpus grandes, semántica |

---

## Mejores Prácticas

### Hiperparámetros

#### Dimensión de Embeddings
- **Pequeña (50-100):** Tareas específicas, vocabulario limitado
- **Media (128-300):** Uso general, buen balance
- **Grande (500-1000):** Tareas complejas, vocabulario masivo

#### Ventana de Contexto
- **Pequeña (2-3):** Relaciones sintácticas (POS tagging)
- **Media (5-7):** Balance sintaxis-semántica
- **Grande (10-15):** Relaciones semánticas amplias

#### Negative Samples
- **Pequeño corpus:** k = 5-10
- **Gran corpus:** k = 2-5
- Más samples → mejor calidad pero más lento

#### Learning Rate
- **Skip-gram:** 0.025 (con decay)
- **CBOW:** 0.05 (con decay)
- Decay lineal: `lr = lr_initial * (1 - epoch/total_epochs)`

### Preprocesamiento

1. **Lowercasing:** Reduce vocabulario, fusiona variantes
2. **Eliminar puntuación:** Enfoca en palabras significativas
3. **Min frequency:** Filtrar palabras muy raras (min_count=5)
4. **Max vocab:** Limitar a palabras más frecuentes (50k-100k)
5. **Subsampling:** t = 1e-3 a 1e-5 para balance

---

## Limitaciones y Extensiones

### Limitaciones

1. **Contexto fijo:** No captura dependencias a larga distancia
2. **Polisemia:** Una palabra = un vector (no maneja múltiples significados)
3. **OOV (Out of Vocabulary):** No puede representar palabras no vistas
4. **Información de orden:** CBOW pierde orden al promediar

### Extensiones

1. **FastText (Meta):** Representa palabras como bag of character n-grams
   - Maneja OOV mediante subwords
   - Mejor para morfología rica

2. **GloVe (Stanford):** Factorización de matriz de co-ocurrencias
   - Aprovecha estadísticas globales
   - Entrenamiento más estable

3. **ELMo, BERT:** Embeddings contextuales
   - Diferentes embeddings según contexto
   - Resuelve polisemia

4. **Paragraph Vector:** Extiende a documentos
   - Aprende representaciones de párrafos/documentos

---

## Aplicaciones Prácticas

### 1. Búsqueda Semántica
```python
query_embedding = model.get_embedding(query)
doc_embeddings = [model.get_embedding(doc) for doc in documents]
similarities = [cosine_similarity(query_embedding, doc_emb)
                for doc_emb in doc_embeddings]
```

### 2. Clasificación de Texto
Usar embeddings pre-entrenados como features para clasificadores:
```python
text_embedding = np.mean([model.get_embedding(word) for word in text], axis=0)
prediction = classifier.predict(text_embedding)
```

### 3. Recomendación
Recomendar items similares basándose en embeddings:
```python
item_embedding = model.get_embedding(item)
similar_items = find_k_nearest(item_embedding, all_item_embeddings, k=10)
```

### 4. Traducción Automática
Alinear espacios de embeddings entre idiomas para traducción.

---

## Referencias

### Papers Fundamentales

1. **Mikolov et al. (2013)** - "Efficient Estimation of Word Representations in Vector Space"
   - Introduce Skip-gram y CBOW
   - [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)

2. **Mikolov et al. (2013)** - "Distributed Representations of Words and Phrases and their Compositionality"
   - Negative sampling y subsampling
   - [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)

3. **Mikolov et al. (2013)** - "Linguistic Regularities in Continuous Space Word Representations"
   - Analogías vectoriales
   - [NAACL-HLT 2013](https://aclanthology.org/N13-1090/)

### Recursos Adicionales

- **Tutorial de TensorFlow:** [Word2Vec Tutorial](https://www.tensorflow.org/tutorials/text/word2vec)
- **Paper Review:** [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- **Original Google Code:** [word2vec C implementation](https://code.google.com/archive/p/word2vec/)

---

## Apéndice: Código de Referencia

### Cálculo de Similitud

```python
import torch.nn.functional as F

def cosine_similarity(vec1, vec2):
    """Calcula similitud coseno entre dos vectores"""
    return F.cosine_similarity(vec1, vec2, dim=0).item()

def find_similar_words(model, word, vocab, inverse_vocab, top_k=5):
    """Encuentra las k palabras más similares"""
    if word not in vocab:
        return []

    word_idx = torch.tensor([vocab[word]])
    word_embedding = model.get_embedding(word_idx)

    all_embeddings = model.embeddings.weight.data

    # Similitud coseno con todas las palabras
    similarities = F.cosine_similarity(
        word_embedding,
        all_embeddings,
        dim=1
    )

    # Top-k (excluyendo la palabra misma)
    top_k_indices = similarities.topk(top_k + 1)[1][1:]

    results = []
    for idx in top_k_indices:
        similar_word = inverse_vocab[idx.item()]
        similarity = similarities[idx].item()
        results.append((similar_word, similarity))

    return results
```

### Visualización con t-SNE

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(model, vocab, num_words=100):
    """Visualiza embeddings usando t-SNE"""
    embeddings = model.embeddings.weight.data.cpu().numpy()[:num_words]

    # Reducir a 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

    # Anotar palabras
    words = list(vocab.keys())[:num_words]
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.title('Word Embeddings Visualization (t-SNE)')
    plt.show()
```

---

**Documento creado para el proyecto Word2Vec PyTorch Implementation**
**Autor:** Documentación técnica generada a partir del código fuente
**Última actualización:** 2025
