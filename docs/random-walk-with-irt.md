
# Random Walk with IRT

### Setup
Import necessary modules, create corpus and model


```python
import os
os.chdir('..')
```


```python
import numpy as np
from src.graph.Graph import UndirectedGraph
from src.algorithms.random_walk import RandomWalker
from src.algorithms.irt import IRT
from src.graph.semantic_network import SemanticNetwork
from src.text.text_wrangler import Corpus
from gensim.models import Word2Vec

np.random.seed(1)
```

    C:\Users\jacob\Anaconda3\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    


```python
shakespeare = Corpus("data/input/shakespeare.txt")
model = Word2Vec(shakespeare.sentence_matrix, size = 120,
                 window = 5, min_count=5, workers=8, sg=1)
network = SemanticNetwork(embeddings=model.wv.vectors, aligned_keys=model.wv.index2word)
```

### Train


```python
for i in range(5):
    model.train(shakespeare.sentence_matrix, total_examples=len(shakespeare.sentence_matrix),
                epochs=1, compute_loss=True)
    print("Round {} ==================".format(i))
    network.update(em_proportion=1, g_proportion=1, include_set=shakespeare.nouns, stop_set=shakespeare.stopwords, thresh=0.8, verbose=True)
```

    Round 0 ==================
    Updated 4277785 edges
    Round 1 ==================
    Updated 2475417 edges
    Round 2 ==================
    Updated 1237055 edges
    Round 3 ==================
    Updated 561099 edges
    Round 4 ==================
    Updated 245587 edges
    

### Create and run random walker


```python
walker = RandomWalker(network.graph)
path = walker.run(10)
path
```




    ['lass',
     'brine',
     'smoky',
     'throe',
     'fun',
     'cicero',
     'peacock',
     'goal',
     'construe',
     'spice']



### Calculate irt


```python
irt = IRT.calculate(path)
```

### Get total count of IRT


```python
tot = 0
for x in irt:
    tot += x[2]
tot
```




    0



### You can also specify a start node for random walk


```python
walker = RandomWalker(network.graph, start='tybalt')
path = walker.run(10)
path
```




    ['tybalt',
     'mercutio',
     'andromache',
     'maw',
     'jar',
     'stillness',
     'glutton',
     'brand',
     'descends',
     'maim']



(NOTE: If the path only consists of the start node, that means the node has no connections)


```python
walker = RandomWalker(network.graph, start='romeo')
path = walker.run(10)
path
```




    ['romeo',
     'romeo',
     'romeo',
     'romeo',
     'romeo',
     'romeo',
     'romeo',
     'romeo',
     'romeo',
     'romeo']




```python

```
