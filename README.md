## Replicating Human Semantic Fluency
Artificial Intelligence
Winter 2019
CS 441/541

## Authors
Zach Bohley ([zbohley](https://github.com/zbohley))
Michelle Duer ([mkduer](https://github.com/mkduer))
Alli Wong ([wonal](https://github.com/wonal))
Carson Cook ([cjc77](https://github.com/cjc77))
Jacob Collins ([jacobmcollins](https://github.com/jacobmcollins))

## Project Overview
Inspired by the work done in "Predicting and Explaining Human Semantic Search in a Cognitive Model" [1], our team decided to create a semantic network off of word embeddings to simulate human semantic fluency. The various steps in our project included:
  * NLP Pre-Processing: cleaning the corpus through tokenization and lemmatization
  * Generating word embeddings using Word2Vec's skip-gram method
  * Learning the embeddings and building a graph network where the edges exist based on cosine similarities above a specific threshold
  * Algorithms for traversing the network and attempting to simulate human fluency
    * Weighted Random Walk
    * Simulated Annealing
    * Random Start/Re-start Hill Climbing
  * Visualizations
    * Networkx plots for visualizing the network and similarities
    * t-SNE plot for starting corpus word embeddings and similar clusters
    * SVD and t-SNE plot for comparing image subspaces and corpus subspaces
    * Seaborn lineplot for algorithm walks
    * Seaborn barplot for total quantitative measurements 
  * Experimentation through parameter fine-tuning
  * Conclusion that the random start/re-start hill climbing algorithm best simulated human semantic fluency
  * Additional projects for visualizing and comparing subspaces of data


## Test Datasets
* Shakespeare: http://www.gutenberg.org/files/100/100-h/100-h.htm
* Fairy Tales: https://www.gutenberg.org/files/19734/19734-h/19734-h.htm
* Wine: https://www.kaggle.com/zynicide/wine-reviews

## Resource
[1]<a name="cite1"></a> F. Miscevic, A. Nematzadeh and S. Stevenson, "Predicting and Explaining Human Semantic Search in a Cognitive Model," Nov. 29, 2017. [Online]. Available:  https://arxiv.org/pdf/1711.11125.pdf.