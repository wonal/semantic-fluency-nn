# Experiments and IRT Results

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Corpus Preparation](#corpus)
4. [Run Algorithms](#algorithms)
5. [Generate Plots](#generate_plots)
6. [Experiments](#experiments)
   * [Simulated Annealing](#sim_anneal): Temperature hyperparameter fine-tuned 
   * [Random Start/Restart Hill Climbing](#hill_climb): Repeat parameter tested
   * [Visualizing Training Effects on Algorithm Performance](#training)
   * [Multiple Test Runs](#tests)
7. [Conclusion](#conclusion)
8. [Resources](#resources)


## 1 Introduction  <a name="introduction"></a>
Inter-item retrieval times (IRT) in psychological human fluency tests measure the time it takes a human to think of related words after an initial prompt. For example, an initial prompt may be "animal". A human may respond with "elephant, rhinoceros, hippo, ..." with short IRT times, but take longer as the test progresses.

The inspiring source for our project compared and used human IRTs as a proxy for their semantic network's fluency measurements [[1](#cite1)]. We used a slightly modified measure that counts IRTs based on repeated visits to a node. In other words, if a node is visited three times, the IRT measurement would be 2, not counting the initial visit.

Experiments were run to compare the IRT results from our three algorithms: *weighted random walk*, *simulated annealing* and *random start/restart hill climbing*.

The [IRTPlot module](https://github.com/mkduer/semantic-fluency-nn/blob/master/src/visualization/irt_plot.py) provides methods for generating barplots for the total IRT measurements for an algorithm and line plots to provide an overview of IRT levels throughout an algorithm.

Experiments were run to modify algorithm hyperparameters to better simulate human semantic fluency. To best simulate human behavior, an algorithm would need to start with small IRT values (simulating an easier time thinking of unique words when first prompted) and end with higher IRT values (simulating a more difficult time to think of unique words as time progresses). Random spikes and drops in IRT values would be expected to better simulate a human's ability to recall related words.



### 2 Setup  <a name="setup"></a>
Import necessary modules.

```python
import os
os.chdir('..')
```

```python
from src.text.text_wrangler import Corpus
from gensim.models import Word2Vec
from src.graph.semantic_network import SemanticNetwork

from src.algorithms.simulated_annealing import SimulatedAnnealer
from src.algorithms.random_walk import RandomWalker
from src.algorithms.hill_climbing import HillClimber
from src.algorithms.irt import IRT

from src.visualization.irt_plot import IRTPlot
import numpy as np
import src.visualization.constants as C
```


### 3 Corpus Preparation  <a name="corpus"></a>
Clean the textual corpus (tokenize and lematize), generate a Word2Vec model from the resulting corpus, and create a semantic network (graph data structure), which is adjusted with each epoch.

```python
corpus = 'shakespeare.txt'
corpus_name = 'Shakespeare Corpus'
save_name = 'shakespeare'

# Clean corpus
clean_corpus = Corpus('data/input/' + corpus)
model = Word2Vec(clean_corpus.sentence_matrix, size=120, window=5, min_count=2, workers=8, sg=1)

# Train the model for a set number of epochs (e.g. C.W2V_ETA)
for i in range(C.W2V_ETA):
     model.train(clean_corpus.sentence_matrix, total_examples=len(clean_corpus.sentence_matrix), epochs=1, compute_loss=True)
     network.update(em_proportion=1, g_proportion=1, include_set=clean_corpus.nouns, stop_set=clean_corpus.stopwords, thresh=0.8, verbose=True)
```


### 4 Run Algorithms  <a name="algorithms"></a>
The algorithms are run on the same network and with the same starting node to provide a similar testing environment. Resulting IRT values for each step of the algorithms are stored.

```python
# Simulated Annealing Algorithm
sim_annealing = SimulatedAnnealer(network.graph, initial_temp=C.SA_TEMP)
sa_path = sim_annealing.run(C.MAX_ITERATIONS)
sa_irts = IRT.calculate(sa_path)
start_node = sa_path[0]

# Weighted Random Walk Algorithm
walker = RandomWalker(network.graph, start_node)
walker_path = walker.run(C.MAX_ITERATIONS)
walker_irts = IRT.calculate(walker_path)

# Random Start/Restart Hill Climbing Algorithm
climber = HillClimber(network.graph, start_node, C.MAX_ITERATIONS, C.HC_REPEAT)
climber_path = climber.run()
climber_irts = IRT.calculate(climber_path)

# Gather IRT data
sa_line, rw_line, hc_line = [], [], []
for sa, walker, climber in zip(sa_irts, walker_irts, climber_irts):
       sa_line.append(sa[2])
       rw_line.append(walker[2])
       hc_line.append(climber[2])

irt_totals = [np.sum(sa_line), np.sum(rw_line), np.sum(hc_line)]
```


### 5 Generate Plots  <a name="generate_plots"></a>
Note that the hyperparameters have been set as constants in the ```constants.py``` file. The following examples will title each plot specifying which plot is being generated. In addition, the plots will be saved with their hyperparameter details to allow for easier hyperparameter fine-tuning, which is recommended for the best visualizations.

```python
plot = IRTPlot()
algorithms = ['Simulated Annealing', 'Random Walk', 'Hill Climbing']
plot.generate_plots(algorithms,  str(test_run) + 'total_irt', irt_totals, 
					str(test_run) + 'line_irt', [sa_line, rw_line, hc_line])
```


### 6 Experiments  <a name="experiments"></a>

Various experiments were run to ensure that hyperparameters were well-tuned to better simulate human fluency. The weighted random walk did not have any hyperparameters and is used as a default that is the closest in representing the original research with modifications to the IRT [[1](#cite1)].

##### Simulated Annealing  <a name="sim_anneal"></a>

The temperature hyperparameter, T, was tested with the values 0.01, 1, 2, . . . , 500, 1000, 2000. For values in [5, 2000], the simulated annealing IRTs remained low throughout the algorithm's run. Lower values of [0.01, 3] resulted in a low IRT to start and a higher IRT towards the end of the algorithm's run, which better simulated human fluency. The hyperparameter was set to T=3 because lower values tended to have extreme spikes in IRT that were too long to realistically mimic human behavior. 

![](/data/output/test_results/sa_line_irt0.png "Simulated Annealing IRTs")

##### Random Start/Restart Hill Climbing  <a name="hill_climb"></a>

A repeat parameter was added to the random start/restart hill climbing algorithm that can mimic a human lingering on a word. This parameter was set to ```repeat = 1```, which resulted in an overal increase in IRT levels throughout the algorithm's run. These increases were too consistently high to realistically simulate human fluency, however, the hyperparameter could be modified to have a repeating effect at random, periodic stages, which could better simulate human fluency.

![](/data/output/test_results/hc_line_irt0.png "Hill Climbing IRTs")

##### Visualizing Training Effects on Algorithm Performance  <a name="training"></a>

Plots were taken of the algorithmic effects while the model and network were trained for 20 epochs. A pattern emerged that was reproduced in several trials where the simulated annealing begins with a very high IRT usually caused by an extreme spike towards the end of the algorithm's run. The total IRT tends to fall to approximately half its original values. The hill climbing and random walk remain very low, but around epoch 5 in training, the random start/restart hill climbing algorithm noticeably increases in total IRTs until it consistently remains at a similar level to the simulated annealing results. The random walk remained low throughout and did not simulate human behavior very well as its IRT levels remained low through the overall algorithm and had no consistent pattern that emerged during training.

![](/data/output/test_results/0total_irt.png "Training Epoch 0")

![](/data/output/test_results/8total_irt.png "Training Epoch 8")

![](/data/output/test_results/19total_irt.png "Training Epoch 19")

##### Multiple Test Runs  <a name="tests"></a>

After the hyperparameters were adjusted and patterns noted, twenty test runs were made to give a final comparison of the algorithms' results and decide which one best simulated human semantic fluency. Following are some random test results::

![](/data/output/test_results/test1_line_irt.png "Test 1")

![](/data/output/test_results/test7_line_irt.png "Test 7")

![](/data/output/test_results/test12_line_irt.png "Test 12")


### 7 Conclusion  <a name="conclusion"></a>

Based on the multiple test runs, the hill climbing and simulated annealing algorithms best simulated the overall expected IRT runs starting low to start and increasing towards the end. The random walk, surprisingly, had low IRTs throughout and its levels were random, but without any pattern that suggested human semantic fluency IRT levels when looking at the line plots.

Between the hill climbing and simulated annealing algorithms, the random start/restart hill climbing algorithm best simulated human semantic fluency due to its periodic random spikes, but a more gentle increase in the IRT levesl compared to the simulated annealing that would have more extreme hikes towards the end.

Considering these results, we thought that the original research could afford more visual representations of the actual walks from the random walk algorithm and how that may compare to the actual human semantic fluency tests rather than focusing on the overall IRT averages. Both the simulated annealing and hill climbing algorithms had similar averages in our tests, however, the actual IRT levels throughout the algorithm runs were considerably different and these seem to be important details to consider.


### 8 Resources  <a name="resources"></a>
[1]<a name="cite1"></a> F. Miscevic, A. Nematzadeh and S. Stevenson, "Predicting and Explaining Human Semantic Search in a Cognitive Model," Nov. 29, 2017. [Online]. Available:  https://arxiv.org/pdf/1711.11125.pdf.  
[2] [Shakespeare Corpus](http://www.gutenberg.org/files/100/100-h/100-h.htm)  
[3] [Fairy Tale Corpus](https://www.gutenberg.org/files/19734/19734-h/19734-h.htm)  