import os
os.chdir('..')

from src.text.text_wrangler import Corpus
from gensim.models import Word2Vec
from src.visualization.tsne_plot import TsnePlot
import src.visualization.constants as C


def prototype():

    # Define corpus details
    multi_corpus = ['fairy_tales.txt', 'shakespeare.txt']
    corpus_names = ['Fairy Tales Corpus', 'Shakespeare Corpus']
    save_name = ['fairytale', 'shakespeare']
    name = 0

    # Grab cleaned corpus and run through word2vec
    for corpus in multi_corpus:
        clean_corpus = Corpus('docs/' + corpus)
        model = Word2Vec(clean_corpus.sentence_matrix, size=120, window=5, min_count=2, workers=8, sg=1)

        # Train the model
        for i in range(5):
            model.train(clean_corpus.sentence_matrix, total_examples=len(clean_corpus.sentence_matrix),
                        epochs=1, compute_loss=True)

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

        # Iterate to next corpus
        name += 1


if __name__ == "__main__":
    prototype()
