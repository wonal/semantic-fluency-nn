# Networkx-Plot

This module creates four types of plots based on a semantic network.   
1. Most similar word pairs
2. Most connected words or words that have the most similarities to other words
3. Custom relational plots 
4. Path coloring for a given walk algorithm

#### Most similar word pairs:  `plot_most_similar_pairs (n: int, title = "<n>_most_similar", expanded: bool, epoch = 0)`
Creates a relational plot of the `n` most similar words in the corpus based on their cosine-similarities.  `n` should be a positive value greater than zero.  By default, `expanded` is `False` and represents whether or not the plot will be more filled in. If set to `True`, each node in the plot will be expanded to show its neighbors.  

#### Most connected words: `plot_most_connected (n: int, title = "<n>_most_connected", epoch = 0)`
Creates a relational plot of the top `n` words that have the most connections/similarities to other words.  `n` should be greater than zero and less than the total number of unique words in the corpus. A low value is recommended for `n`.

#### Custom relational plots: `create_custom_plot (words: [str], title = "custom_plot", epoch = 0)`
Creates a relational plot based on a list of words.  The method expands each word and plots both the word and its neighbors.   

#### Path coloring for a given walk algorithm: `create_colored_path_plot (path: [str], title = "path_colored_plot", epoch = 0)`
Creates a relational plot based on a path taken by a network walker.  The method plots the path in a color specified in `constants.py` and then expands each word in the plot.  


### Setup
Import necessary modules


```python
import os
os.chdir('..')
```


```python
from src.graph.semantic_network import SemanticNetwork
from src.text.text_wrangler import Corpus
from gensim.models import Word2Vec
from src.visualization.networkx_plot import NetworkxPlot
```

Import a network walker (HillClimber, RandomWalker, SimulatedAnnealer) if creating the path coloring plot:

```python
from src.algorithm.random_walk import RandomWalker
```

Create corpus, model, and network:

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
    

### Create NetworkxPlot and choose desired plot   

```python
nx_plot = NetworkxPlot(network)
```

##### Most similar word pairs
```python
nx_plot.plot_most_similar_pairs(100)
```

##### Most connected words 
```python
nx_plot.plot_most_connected(10)
```

##### Custom relational plots 
```python
some_hamlet_characters = ['claudius', 'cornelius', 'fortinbras', 'francisco', 'guildenstern', 'hamlet', 'horatio', 'laertes', 'marcellus', 'ophelia', 'osric', 'polonius', 'reynaldo', 'rosencrantz', 'servant']
nx_plot.create_custom_plot(some_hamlet_characters)
```

##### Path coloring for a given walk algorithm
```python
walker = RandomWalker(network.graph)
path = walker.run(15)
nx_plot.create_colored_path_plot(path)
```

No plots will be shown for paths of length zero or one. 

### Plot titles

Custom plot titles can be given for any of the plots:
```python
walker = RandomWalker(network.graph)
path = walker.run(15)
nx_plot.create_colored_path_plot(path, title="random_walk_path_plot")
```

It may also be useful to plot over epochs, in which case the epoch can be noted within the title:

```python
for i in range(5):
    model.train(shakespeare.sentence_matrix, total_examples=len(shakespeare.sentence_matrix),
                epochs=1, compute_loss=True)
    print("Round {} ==================".format(i))
    network.update(em_proportion=1, g_proportion=1, include_set=shakespeare.nouns, stop_set=shakespeare.stopwords, thresh=0.8, verbose=True)
    nx_plot.plot_most_similar_pairs(100, title="shakespeare_100_most_similar_thresh80", epoch=i)

```

 
