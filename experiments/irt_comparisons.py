import os
os.chdir('..')

import numpy as np
from src.graph.Graph import UndirectedGraph  # TODO remove if not used
from src.algorithms.simulated_annealing import SimulatedAnnealer
from src.algorithms.random_walk import RandomWalker
from src.algorithms.hill_climbing import HillClimber
from src.algorithms.irt import IRT
from src.graph.semantic_network import SemanticNetwork
from src.text.text_wrangler import Corpus
from gensim.models import Word2Vec
import src.visualization.constants as C


def measure_algorithm_irts():
    """
    Runs the different algorithms using the same starting node and total iterations:
    - random walk
    - simulated annealing
    - discrete space random-start hill climbing

    The IRT of each algorithms resulting path is calculated and plotted
    """

    irt_totals = [0, 0, 0]

    print(f'cleaning corpus...')
    corpus = ['shakespeare.txt', 'fairy_tales.txt']
    clean_corpus = Corpus('docs/' + corpus[1])
    print(f'passing through W2V...')
    model = Word2Vec(clean_corpus.sentence_matrix, size=120,
                     window=5, min_count=5, workers=8, sg=1)
    print(f'semantic network...')
    network = SemanticNetwork(embeddings=model.wv.vectors, aligned_keys=model.wv.index2word)

    print(f'training model...')
    for i in range(C.W2V_ETA):
        model.train(clean_corpus.sentence_matrix, total_examples=len(clean_corpus.sentence_matrix),
                    epochs=1, compute_loss=True)
        network.update(em_proportion=1, g_proportion=1, include_set=clean_corpus.nouns,
                       stop_set=clean_corpus.stopwords, thresh=0.8, verbose=True)

    print(f'iterate {C.MAX_ITERATIONS} times')
    print(f'\nSimulated Annealing...')
    sim_annealing = SimulatedAnnealer(network.graph)
    path = sim_annealing.run(C.MAX_ITERATIONS)
    print(f'simulated annealing path: {path}')

    start_node = path[0]
    print(f'START NODE: {start_node}')

    print('\nRandom Walker...')
    walker = RandomWalker(network.graph, start_node)
    path = walker.run(C.MAX_ITERATIONS)
    print(f'random walker path: {path}')

    print('\nHill Climbing...')
    climber = HillClimber(network.graph, start_node, C.MAX_ITERATIONS, 0)
    path = climber.run()
    print(f'random walker path: {path}')


    # TODO: implement IRT counts
    """
    print(f'calculating IRT')
    irts = IRT.calculate(path)  # TODO: curious, would walker.calculate(path) work because it's a @classmethod?
    print(f'irts: {irts}')

    for irt in irts:
        irt_totals += irt[2]   # TODO adjust so that each index of irt_totals represents a different algorithms results
    print(f'total for random walk: {irt_totals}')
    """


if __name__ == '__main__':
    np.random.seed(1)   # TODO: probably not needed unless testing
    measure_algorithm_irts()
