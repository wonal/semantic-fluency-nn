t-SNE Visualizations for Word Embeddings

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Corpus Preparation](#corpus)
4. [Generate Plots](#generate_plots)
5. [Examples of tSNE Visualizations](#examples)
   * [2-Dimensional t-SNE Plot](#2D)
   * [3-Dimensional t-SNE Plot](#3D)
   * [t-SNE Cluster Plot](#cluster)
6. [Additional Resources](#resources)


## 1 Introduction  <a name="introduction"></a>
The [TsnePlot module](https://github.com/mkduer/semantic-fluency-nn/blob/master/src/visualization/tsne_plot.py) provides methods for 2-dimensional and 3-dimensional visualizations of Word2Vec embeddings generated from a text corpus. In addition, subsets of similar words can be clustered and visualized based on specific corpus keywords that are passed in.

Note that t-SNE, t-distributed Stochastic Neighbor Embedding, is not meant to give perfect mathematical representations of data. Rather, it is a visualization tool for reducing high-dimensional data into smaller dimensions to allow for any type of comprehensible visualization. More about this module can be found here: [sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

That being said, t-SNE can be a very fun way to cut through dense data and allow for human-readable representations of high-dimensional data. The module does tend to require an extensive amount of hyperparameter fine-tuning. Slight tweaks in the hyperparameters can result in very different visualizations, so knowing what you expect from your data and what you're trying to relay is quite important prior to creating these visualizations. Examples are provided below.

Note: the Shakespeare corpus was generated from available content in Project Gutenberg ([Shakespeare Corpus Source](http://www.gutenberg.org/files/100/100-h/100-h.htm)). 

### 2 Setup  <a name="setup"></a>
Import necessary modules.

```python
import os
os.chdir('..')
```

```python
from src.text.text_wrangler import Corpus
from gensim.models import Word2Vec
from src.visualization.tsne_plot import TsnePlot
import src.visualization.constants as C
```


### 3 Corpus Preparation  <a name="corpus"></a>
Clean the textual corpus (tokenize and lematize) and generate a Word2Vec model from the resulting corpus.

```python
corpus = 'shakespeare.txt'
corpus_name = 'Shakespeare Corpus'
save_name = 'shakespeare'

clean_corpus = Corpus('docs/' + corpus)
model = Word2Vec(clean_corpus.sentence_matrix, size=120, window=5, min_count=2, workers=8, sg=1)

# Train the model for a set number of epochs (e.g. C.W2V_ETA)
for i in range(C.W2V_ETA):
     model.train(clean_corpus.sentence_matrix, total_examples=len(clean_corpus.sentence_matrix), epochs=1, compute_loss=True)
```


### 4 Generate Plots  <a name="generate_plots"></a>
Note that the hyperparameters have been set as constants in the ```constants.py``` file. The following examples will title each plot specifying which plot is being generated. In addition, the plots will be saved with their hyperparameter details to allow for easier hyperparameter fine-tuning, which is recommended for the best visualizations.

```python
plot = TsnePlot()

# Create Cluster Plot
words, clusters = plot.similarity_clusters(model.wv, C.CLUSTER_KEYS)
reduced_clusters = plot.reduce_clusters(clusters)
img_title = plot.save_title(save_name[name], C.C_PERPLEXITY, C.C_COMPONENTS, C.C_ITER, C.C_ETA, '_top ' + str(C.TOP_N))
plot_title = corpus_names[name] + ': Word Embeddings (Top ' + str(C.TOP_N) + ' Most Similar)'
plot.visualize_clusters(reduced_clusters, words, C.CLUSTER_KEYS, plot_title, img_title)

# Process data for full embedding plots
words, tokens = plot.process_vectors(model.wv)

# Generate 2D Embeddings Plot
reduced_model_2D = plot.reduce_model(tokens, C.C_PERPLEXITY, C.C_COMPONENTS, C.C_ITER, C.C_ETA)
img_title = plot.save_title(save_name[name], C.C_PERPLEXITY, C.C_COMPONENTS, C.C_ITER, C.C_ETA)
plot_title = corpus_names[name] + ': 2-Dimensional Word Embeddings'
plot.visualize_embeddings_2D(reduced_model_2D, words, plot_title, img_title, name)

# Generate 3D Embeddings Plot
reduced_model_3D = plot.reduce_model(tokens, C.DIM3_PERPLEXITY, C.DIM3_COMPONENTS, C.DIM3_ITER, C.DIM3_ETA)
img_title = plot.save_title(save_name[name], C.DIM3_PERPLEXITY, C.DIM3_COMPONENTS, C.DIM3_ITER, C.DIM3_ETA)
plot_title = corpus_names[name] + ': 3-Dimensional Word Embeddings'
plot.visualize_embeddings_3D(reduced_model_3D, words, plot_title, img_title, name)
```


### 5 Examples of tSNE Visualizations  <a name="examples"></a>

The 2-dimensional embeddings could be saved in the following format displaying what hyperparemter settings were applied: *shakespeare_tSNE2D_perplexity5_components2_iter2400_eta450*. 

This allows for more efficient hyperparmeter fine-tuning with the learning rate, perplexity, iterations and more. Adjusting these hyperparameter values can result in drastically different results with t-SNE. The following examples all use the Shakespeare corpus to generate visualizations of the embeddings.

##### t-SNE Cluster Plot  <a name="cluster"></a>


![tSNE cluster plot](https://github.com/mkduer/semantic-fluency-nn/tree/master/docs/example_images/shakespeare_tSNE2D_perplexity5_components2_iter2400_eta450_top100.png "tSNE cluster plot" | width=100

![tSNE cluster plot](https://github.com/mkduer/semantic-fluency-nn/tree/master/docs/example_images/shakespeare_tSNE2D_perplexity5_components2_iter2400_eta450_top100.png "tSNE cluster plot" =100x75)

![tSNE cluster plot](https://github.com/mkduer/semantic-fluency-nn/tree/master/docs/example_images/shakespeare_tSNE2D_perplexity5_components2_iter2400_eta450_top100.png "tSNE cluster plot")

##### 2-Dimensional t-SNE Plot  <a name="2D"></a>

![2D tSNE plot](https://github.com/mkduer/semantic-fluency-nn/tree/master/docs/example_images/shakespeare_tSNE2D_perplexity5_components2_iter2400_eta450.png "2D tSNE plot")

##### 3-Dimensional t-SNE Plot  <a name="3D"></a>

![3D tSNE plot](https://github.com/mkduer/semantic-fluency-nn/tree/master/docs/example_images/shakespeare_tSNE3D_perplexity10_components3_iter2400_eta400.png "3D tSNE plot")


### 6 Additional Resources  <a name="resources"></a>
These resources were used to further understand how to implement t-SNE, its restrictions and how to better visualize with hyperparameter fine-tuning.

[How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
[Google News and Leo Tolstoy: Visualizing Word2Vec Word Embeddings using t-SNE](https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d)
[sklearn.manifold.TSNE module](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)