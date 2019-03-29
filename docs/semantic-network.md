
# semantic_network API

1. [Constructor](#constructor)
2. [Embedding Matrix and Aligned Keys](#matrixandkeys)
3. [Key to Index](#keytoindex)
4. [Index to Key](#indextokey)
5. [Graph](#graph)
6. [Demo: Semantic Network Learning](#learning)


```python
import numpy as np
import os
os.chdir('..')
from src.graph.semantic_network import SemanticNetwork
from src.text.text_wrangler import Corpus
from gensim.models import Word2Vec
```

    C:\Users\carso\Anaconda3\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    


```python
shakespeare = Corpus("data/input/shakespeare.txt")
model = Word2Vec(shakespeare.sentence_matrix, size = 120,
                 window = 5, min_count=5, workers=8, sg=1)
```

## 1 Constructor <a name="constructor"></a>
Pass in the embeddings matrix (VxE) and keys (Vx1) that are to be used in the semantic_network.

Note: the keys vector MUST be aligned so that the index of each key corresponds to its row in the embeddings matrix.

* For Word2Vec: embeddings are the word vectors and aligned_keys are the keys for the word vectors.


```python
network = SemanticNetwork(embeddings=model.wv.vectors, aligned_keys=model.wv.index2word)
```

## 2 Embedding Matrix and Aligned Keys <a name="matrixandkeys"></a>
The embedding matrix is a matrix of all embedding vectors. 
The aligned keys are the keys that correspond to rows in the embedding matrix.


```python
for i in range(10):
    print("{}: {}...".format(network.aligned_keys[i], network.embedding_matrix[i, 0:5]))
```

    the: [-0.00280978 -0.33994433 -0.22588709  0.13702348  0.1990086 ]...
    and: [ 0.14502114  0.01234245 -0.13245517 -0.14034772  0.12314954]...
    i: [-0.0431974  -0.12119325  0.08583751  0.08318941  0.171759  ]...
    a: [ 0.24622974 -0.01455849 -0.36390817 -0.08250172  0.30967784]...
    to: [ 0.22367853  0.2579566   0.00666375 -0.06436289 -0.04638908]...
    of: [-0.02497098 -0.05907164 -0.41035387 -0.23772378 -0.00653878]...
    you: [-0.0706336   0.12692171 -0.02447871  0.15697604  0.05940285]...
    my: [-0.08880568 -0.04827406  0.07021175  0.1239695   0.23525085]...
    in: [ 0.16767137  0.09106478 -0.18268701 -0.39845642  0.15184204]...
    that: [ 0.11740334 -0.06691196 -0.14201538 -0.08600818  0.08484195]...
    

## 3 Key to Index <a name="keytoindex"></a>
Retrieve the index of an embedding via a key


```python
print(network.embedding_matrix[network.key_to_index["romeo"], :])
```

    [-0.08698846 -0.26235968 -0.26969332  0.05089405  0.15857889  0.24010134
      0.35792786 -0.47392368 -0.5088288   0.09867685 -0.17164436  0.05717923
     -0.11224584  0.46068627 -0.42624918 -0.19675364  0.20519069 -0.06418575
      0.00552568 -0.29590446  0.18216302  0.2589483   0.13376693  0.00700089
      0.17066754  0.25589782 -0.14134236  0.17260018 -0.12932506  0.37334207
     -0.1600645   0.03352617 -0.13623567 -0.07444574  0.16503198  0.0446138
      0.3393892  -0.0975441  -0.32340658  0.11510324  0.04890903  0.57631826
     -0.29897442  0.14389968 -0.11519077 -0.3163161   0.46940663 -0.0771839
      0.22386383  0.17656524 -0.2509969  -0.16984825  0.05441857  0.19751723
     -0.37420136 -0.11913896 -0.52730334 -0.30328637  0.06067149  0.05582255
      0.08092062  0.40430903  0.1431255   0.02778628 -0.34552318 -0.18416773
     -0.09545767  0.20395552  0.00749557  0.33116296 -0.30359653 -0.15646084
     -0.39150745  0.39895764  0.2427295  -0.03765616 -0.0100162  -0.44775602
     -0.2023313  -0.39661965 -0.15501924 -0.06159584  0.19407149 -0.26925865
     -0.15548486  0.00708158  0.20396248  0.38953647 -0.13853522  0.06578679
      0.11617649 -0.15046456  0.17771612 -0.06989928 -0.05050361 -0.12588148
      0.0888787   0.39925686  0.24629065  0.15699667 -0.3816204  -0.40458515
     -0.18548444  0.37377375  0.27805638  0.18530765 -0.50066394 -0.05220122
      0.00749597 -0.18861982 -0.21458456 -0.03480453 -0.07056356  0.46907103
      0.16996111  0.24567446 -0.19755246  0.18760915  0.27138415 -0.0855778 ]
    

## 4 Index to Key <a name="indextokey"></a>
Retrieve the key of an embedding via an index


```python
romeo_index = network.key_to_index["romeo"]
print(network.index_to_key[romeo_index])
```

    romeo
    

## 5 Graph <a name="graph"></a>
View the graph associated with a semantic network


```python
print(network.graph.nodes[:10])
```

    ['the', 'and', 'i', 'a', 'to', 'of', 'you', 'my', 'in', 'that']
    

## 6 Semantic Network Learning <a name="learning"></a>
Incrementally learn embeddings using Word2Vec and update the semantic network's graph using update()
* em_proportion: the proportion of the embeddings to be sampled
* g_proportion: the proportion of nodes in the graph to be sampled
* include_set: what to update (i.e. nouns, or some other set of valid keys)
* stop_set: what to ignore (i.e. stopwords). This is only really needed if you want to ignore some of what is in your include_set
* thresh: threshold for inclusion in the graph. With the library being used, higher = more related, lower = less related. If the similarity between two keys is below the threshold, we do not include/update it in the graph


```python
for i in range(5):
    model.train(shakespeare.sentence_matrix, total_examples=len(shakespeare.sentence_matrix),
                epochs=1, compute_loss=True)
    print("Round {} ==================".format(i))
    network.update(em_proportion=0.1, g_proportion=0.1, include_set=shakespeare.nouns, stop_set=shakespeare.stopwords, thresh=0.8, verbose=True)
```

    Round 0 ==================
    Updated 40611 edges
    Round 1 ==================
    Updated 27625 edges
    Round 2 ==================
    Updated 12154 edges
    Round 3 ==================
    Updated 5469 edges
    Round 4 ==================
    Updated 1957 edges
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
