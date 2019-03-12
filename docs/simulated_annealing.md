# Simulated Annealing Walk

This module uses the simulated annealing algorithm as a walk through a semantic network.  The cooling schedule was taken from Yaghout Nourani and Bjarne Andresen's 1998 paper "A comparison of simulated annealing cooling strategies" found [here.](https://www.fys.ku.dk/~andresen/BAhome/ownpapers/permanents/annealSched.pdf)

### Setup
Import necessary modules, create corpus and model


```python
import os
os.chdir('..')
```


```python
from src.algorithms.simulated_annealing import SimulatedAnnealer
from src.graph.semantic_network import SemanticNetwork
from src.text.text_wrangler import Corpus
from gensim.models import Word2Vec
```

```python
shakespeare = Corpus("docs/shakespeare.txt")
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
    

### Create and run the simulated annealing walker

Create a SimulatedAnnealer object, passing in the formed semantic network as an UndirectedGraph.  By default, the initial temperature is set to 1000, and the start node will be chosen at random.

The run() method takes a count parameter, which is the number of steps the walker should take in the network, or the length of the desired output path.   

```python
walker = SimulatedAnnealer(network.graph)
path = walker.run(10)
path
```




    ['division', 
    'operation', 
    'nourishment', 
    'botch', 
    'holder', 
    'sty', 
    'sweetheart', 
    'talke', 
    'thatch', 
    'worshipp', 
    'mortall']




 You can also specify a different starting temperature and start node for the simulated annealing walk


```python
walker = SimulatedAnnealer(network.graph, initial_temp=2000, start='tybalt')
path = walker.run(10)
path
```


    ['tybalt', 
    'bassianus', 
    'wiltshire', 
    'germany', 
    'heresy', 
    'foster', 
    'interpreter', 
    'turne', 
    'drowns', 
    'damnation', 
    'beautify']

If the output path only consists of the starting node, that means the starting node has no other connections. 

    ['tybalt']
