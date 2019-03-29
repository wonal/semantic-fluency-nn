
# text_wrangler API

1. [Constructor](#constructor)
2. [Sentence Matrix](#sentence_matrix)
3. [Vocabulary](#vocabulary)
4. [Stopwords](#stopwords)
5. [Filtered Vocabulary](#filtered_vocabulary)
6. [Demo with Word2Vec](#word2vec_demo)


```python
import os
os.chdir("..")
```


```python
from src.graph.Graph import UndirectedGraph
from src.text.text_wrangler import Corpus
```

## 1 Constructor <a name="constructor"></a>
Loads a corpus into a text_wrangler.Corpus object


```python
shakespeare = Corpus("data/input/shakespeare.txt")
```

## 2 Sentence Matrix <a name="sentence_matrix"></a>
A matrix where the rows are tokenized sentences. This is the format that the Word2Vec model expects to receive the corpus in.


```python
shakespeare.sentence_matrix[200]
```




    ['ah',
     'wherefore',
     'with',
     'infection',
     'should',
     'he',
     'live',
     'and',
     'with',
     'his',
     'presence',
     'grace',
     'impiety',
     'that',
     'sin',
     'by',
     'him',
     'advantage',
     'should',
     'achieve',
     'and',
     'lace',
     'it',
     'self',
     'with',
     'his',
     'society']



## 3 Vocabulary <a name="vocabulary"></a>
A set containing the unique vocabulary in the corpus.


```python
list(shakespeare.vocab)[:10]
```




    ['philomels',
     'using',
     'bid',
     'stagger',
     'metamorphisd',
     'loathed',
     'fitment',
     'producing',
     'lingered',
     'plashy']




```python
len(shakespeare.vocab)
```




    22444



## 4 Stopwords <a name="stopwords"></a>
A set containing the words that may be unimportant.


```python
list(shakespeare.stopwords)[:10]
```




    ['about',
     'hasn',
     'i',
     "mustn't",
     'then',
     'couldn',
     'ours',
     'here',
     'won',
     'shouldn']



## 5 Filtered Vocabulary <a name="filtered_vocabulary"></a>
A set containing the corpus' vocab, with stopwords filtered out.


```python
list(shakespeare.filtered_vocab)[:10]
```




    ['philomels',
     'using',
     'bid',
     'stagger',
     'metamorphisd',
     'loathed',
     'fitment',
     'producing',
     'lingered',
     'plashy']



## 6 Word2Vec Demo with text_wrangler.Corpus <a name="word2vec_demo"></a>


```python
from gensim.models import Word2Vec

model = Word2Vec(shakespeare.sentence_matrix, size = 120,
                 window = 5, min_count=5, workers=8, sg=1)
for i in range(5):
    model.train(shakespeare.sentence_matrix, total_examples=len(shakespeare.sentence_matrix),
                epochs=1, compute_loss=True)
    loss = model.get_latest_training_loss()
    # Quick glimpse at what Word2Vec finds to be the most similar
    sim = model.wv.most_similar("romeo")
    print("Round {} ==================".format(i))
    for s in sim:
        print(s)
    print("\n\n")
```

    Round 0 ==================
    ('tybalt', 0.864901065826416)
    ('juliet', 0.7948235273361206)
    ('mercutio', 0.7702454328536987)
    ('nurse', 0.7538067102432251)
    ('imogen', 0.7395459413528442)
    ('rutland', 0.7356479167938232)
    ('troilus', 0.7340472936630249)
    ('arthur', 0.7301068902015686)
    ('lavinia', 0.7268906235694885)
    ('percy', 0.7246760129928589)
    
    
    
    Round 1 ==================
    ('tybalt', 0.8521748185157776)
    ('juliet', 0.7760286331176758)
    ('mercutio', 0.760892391204834)
    ('rutland', 0.7247177362442017)
    ('nurse', 0.7158874869346619)
    ('percy', 0.7120639681816101)
    ('imogen', 0.7080466747283936)
    ('lavinia', 0.7065379023551941)
    ('troilus', 0.7044037580490112)
    ('arthur', 0.7034294605255127)
    
    
    
    Round 2 ==================
    ('tybalt', 0.8476352095603943)
    ('mercutio', 0.7645272016525269)
    ('juliet', 0.7600152492523193)
    ('arthur', 0.7063480615615845)
    ('cato', 0.7031240463256836)
    ('nurse', 0.70293790102005)
    ('troilus', 0.7006533145904541)
    ('imogen', 0.6991547346115112)
    ('richmond', 0.6943120956420898)
    ('lavinia', 0.6922498345375061)
    
    
    
    Round 3 ==================
    ('tybalt', 0.8407829999923706)
    ('mercutio', 0.7501236200332642)
    ('juliet', 0.7363349199295044)
    ('cato', 0.6898269653320312)
    ('benvolio', 0.686690092086792)
    ('arthur', 0.6824208498001099)
    ('montague', 0.6800373792648315)
    ('aeneas', 0.6776838898658752)
    ('troilus', 0.675659716129303)
    ('richmond', 0.6729200482368469)
    
    
    
    Round 4 ==================
    ('tybalt', 0.8286418914794922)
    ('mercutio', 0.7347580790519714)
    ('juliet', 0.7073820233345032)
    ('benvolio', 0.6708059906959534)
    ('arthur', 0.6629817485809326)
    ('cato', 0.6616808176040649)
    ('aeneas', 0.6560759544372559)
    ('troilus', 0.6495552062988281)
    ('richmond', 0.6470909118652344)
    ('bassianus', 0.6407738924026489)
    
    
    
    
