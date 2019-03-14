from gensim.models import Word2Vec
from nltk import sent_tokenize, RegexpTokenizer, WordNetLemmatizer

from sklearn.manifold import TSNE  # imports t-SNE to visualize embeddings
import matplotlib.pyplot as plt  # allows for plotting
from mpl_toolkits.mplot3d import Axes3D  # allows for 3D plotting
import matplotlib.cm as cm  # allows for coloring data points
import numpy as np  # imports numpy for matrix operations

from os.path import exists  # check that path exists
from os import mkdir, getcwd  # directory operations
from shutil import rmtree  # remove specific directory contents
import constants as C  # import constants

from datetime import datetime  # measure time for plotting to complete


class TsnePlot:

    def plot_clusters(self, words: [str], clusters: [[str]], cluster_keys: [str]):
        """
        Creates the tSNE model for visualizing clusters of similar words
        :param words: string of words
        :param clusters: a vector of similarity scores
        :param cluster_keys: the keys containing initial words for each cluster
        """
        # hyperparameters
        perplexity, components = 5, 2
        iterations, learning_rate = 2500, 150

        # turn clusters into numpy array
        clusters = np.array(clusters)
        n, m, k = clusters.shape

        # Setup for visualizing 2-Dimensional cluster model
        model = TSNE(perplexity=perplexity, n_components=components, init='pca', n_iter=iterations)
        embeddings_2D = np.array(model.fit_transform(clusters.reshape(n * m, k))).reshape(n, m, 2)

        # Create specific image title and create visualization
        img_title = self.save_title(perplexity, components, iterations, '_top ' + str(C.TOP_N))
        plot_title = 'Word Embeddings (Top ' + str(C.TOP_N) + ' Most Similar)'
        self.visualize_clusters(embeddings_2D, words, cluster_keys, plot_title, img_title)

    def visualize_clusters(self, embeddings: [[int, int]], words: [str], keys: [str],
                              plot_title: str, img_title: str):
        """
        Create a 2-dimensional visual of the embeddings
        :param embeddings: the embedding coordinates
        :param words: the word labels
        :param keys: the original cluster keys
        :param plot_title: the plot title
        :param img_title: the title for saving the image
        """
        plt.figure(figsize=(16, 9))

        cmap = cm.rainbow(np.linspace(0.0, 1.0, len(keys)))
        word_count = 0

        for key, coords, color in zip(keys, embeddings, cmap):
            x, y = coords[:, 0], coords[:, 1]
            plt.scatter(x, y, color=color, alpha=0.9, label=key)

            cluster = len(x)
            for i in range(cluster):
                plt.annotate(words[word_count],
                             xy=(x[i], y[i]),
                             alpha=0.8,
                             size=9.5,
                             textcoords='offset pixels', xytext=(-15, 7),
                             ha='right', va='top')
                word_count += 1

        plt.title(plot_title)
        plt.grid(True)
        plt.legend(loc=4)
        self.save_plot(img_title)
        plt.show()

    def tsne_plot(self, word_vectors):
        """
        Create the tSNE model for visually representing word embeddings
        :param word_vectors: contains vectors of words generated from Word2Vec (aka model.wv)
        """
        # hyperparameters
        perplexity, components = 5, 2
        iterations, learning_rate = 2400, 200

        tokens, words = [], []

        # TODO TESTING: temporarily reduce the number of words that are graphed
        if C.TEST:
            count = 0
            test_amount = 100  # set the number of embeddings that are used
            for word in word_vectors.vocab:
                count += 1
                if count == test_amount:
                    break
                tokens.append(word_vectors[word])
                words.append(word)
        # TODO: remove this else statement for the final commit
        else:
            # tokenize the word embeddings
            for word in word_vectors.vocab:
                tokens.append(word_vectors[word])
                words.append(word)

        # Setup for visualizing 2-Dimensional model
        model = TSNE(perplexity=perplexity, n_components=components, init='pca', n_iter=iterations)
        embeddings_2D = model.fit_transform(tokens)

        # Create specific title and create visualization
        img_title = self.save_title(perplexity, components, iterations)
        plot_title = '2-Dimensional Word Embeddings'
        self.visualize_embeddings_2D(embeddings_2D, words, plot_title, img_title)

        # Setup for visualizing 3-Dimensional model
        perplexity, components = 10, 3
        iterations, learning_rate = 2400, 200
        model_3D = TSNE(perplexity=perplexity, n_components=components, init='pca', n_iter=iterations)
        embeddings_3D = model_3D.fit_transform(tokens)

        # Create specific title and create visualization
        img_title = self.save_title(perplexity, components, iterations)
        plot_title = '3-Dimensional Word Embeddings'
        self.visualize_embeddings_3D(embeddings_3D, words, plot_title, img_title)

    def visualize_embeddings_3D(self, embeddings: [[int, int]], labels: [str], plot_title: str, img_title: str):
        """
        Plots the parameterized data into a visual plot
        :param embeddings: the x, y and z datapoints for the embeddings
        :param labels: word labels associated with each xy data point
        :param plot_title: plot title
        :param img_title: title for saved image
        note: help for this plotting was found on
        https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-
        embeddings-with-t-sne-11558d8bd4d
        """
        xs, ys, zs = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
        total = len(xs)

        plt.figure(figsize=(16, 9))
        fig = plt.figure()
        ax = Axes3D(fig)

        print(f'3D Word Embeddings generating for {total} embeddings...')
        ax.scatter3D(xs, ys, zs, color='#00b4be', edgecolors='#00a3ac')

        total_count = np.arange(0, total, dtype=int)
        for x, y, z, count in zip(xs, ys, zs, total_count):
            if count % C.RESTRICT_3D_TEXT == 0:  # restricts number of annotations
                print(f'count: {count} with label: {labels[count]}')
                ax.text(x, y, z, labels[count], 'y', fontsize=8, color='#000000')

        plt.title(plot_title, y=1.02)
        self.save_plot(img_title)
        plt.show()

    def visualize_embeddings_2D(self, embeddings: [[int, int]], labels: [str], plot_title: str, img_title: str):
        """
        Plots the parameterized data into a visual plot
        :param embeddings: the x and y datapoints for the embeddings
        :param labels: word labels associated with each xy data point
        :param plot_title: plot title
        :param img_title: title for saved image
        """
        x, y = embeddings[:, 0], embeddings[:, 1]
        total = len(x)
        x_min, x_max = min(x) - 10, max(x) + 10

        plt.figure(figsize=(16, 9))
        fig, ax = plt.subplots()

        ax.scatter(x, y, alpha=0.4, color='#0375fa', edgecolors='#0256bc')
        for coord in range(total):
            ax.annotate(labels[coord],
                        xy=(x[coord], y[coord]),
                        size=6,
                        alpha=1,
                        textcoords='offset pixels', xytext=(-8, 5),
                        ha='right', va='top')

        plt.grid(True)
        plt.title(plot_title)
        self.save_plot(img_title)
        plt.show()

    @staticmethod
    def similarity_clusters(self, word_vectors, cluster_keys: [str]) -> ([str], [[str]]):
        """
        Provide specific cluster keys around which to find similarity clusters
        :param model: the Word2Vec model containing all of the word embeddings
        :param cluster_keys: the key words for finding similar word clusters
        :return: the words and embedding clusters
        """
        words, embedding_clusters = [], []

        for word in cluster_keys:
            embeddings = []
            for similar_word, similarity in word_vectors.most_similar(word, topn=C.TOP_N):
                words.append(similar_word)
                embeddings.append(word_vectors[word])
            embedding_clusters.append(embeddings)

        return words, embedding_clusters

    @staticmethod
    def save_title(self, perplexity: int, components: int, iterations: int, extra: str = ''):
        """
        Returns string title for image to be saved with parameter details
        :param perplexity: hyperparameter detailing nearest neighbors
        :param components: reduced dimensions
        :param iterations: total itereations
        :param extra: any extra details in string type
        :return: save image name
        """
        return 'tSNE2D_perplexity' + str(perplexity) + \
               '_components' + str(components) + \
               '_iter' + str(iterations) + extra

    @staticmethod
    def save_plot(self, title: str):
        """
        Saves plot to specified directory.
        :param title: title of plot
        """
        path = C.GRAPH_DIR

        if C.DELETE_GRAPHS:
            # clear directory if specified by function call
            if exists(path):
                rmtree(path)

        # create directory if it doesn't exist
        if not exists(path):
            mkdir(path)

        # save plot to specified directory
        img = path + title + '.png'
        plt.savefig(img)
