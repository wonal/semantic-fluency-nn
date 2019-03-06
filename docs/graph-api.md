
# Graph API

# Table of contents
1. [Introduction](#introduction)
1. [Import](#graph_import)
2. [Create a graph](#graph_create)
3. [Add an edge](#graph_add_edge)
4. [Get an edge weight given two nodes](#graph_add_edge)
5. [Expand a node](#graph_expand)
6. [Example](#graph_example)

## Introduction <a name="introduction"></a>
The UndirectedGraph class implements a graph using an adjacency matrix. This graph cannot contain self-edges, or repeated edges. Each column and row represents a node in the graph. The node represented by the ith column is the same node represented by the ith row. An edge weight of 0 means there is no edge, but the existence of a non-zero float implies an edge between the nodes.

For the context of this API, a node can be of any type. In the examples I use both a string and an integer to demonstrate this. The UndirectedGraph class uses a dictionary to store the node -> row/column integer pairs. This class provides no way to delete a node after it has been added (this can be changed if someone needs that functionality).

## Import <a name="graph_import"></a>


```python
import os
os.chdir('..')

from src.graph.Graph import UndirectedGraph
```

### 2.2 Create a graph: <a name="graph_create"></a>


```python
g = UndirectedGraph()
```

### 2.3 Add an edge: <a name="graph_add_edge"></a>
    Input: Two nodes and a float representing the edge weight
    Output: None


```python
first_node = 'Gold'
second_node = 'Silver'
edge_weight = 0.01017856140726281

g.add_edge(first_node, second_node, edge_weight)
```

### 2.4 Get an edge weight given two nodes: <a name="graph_get_weight"></a>
    Input: Two nodes
    Output: An edge weight


```python
g.edge(first_node, second_node)
```




    0.01017856140726281



## Expand a node: <a name="graph_expand"></a>
    Input: A node
    Output: A list of tuples. The 0th element is an adjacent node, and the 1st element is an edge.


```python
g.expand('Gold', sort=True)
```




    [('Silver', 0.01017856140726281)]



### 2.6 Get a list of the nodes: <a name="graph_nodes"></a>


```python
g.nodes
```




    ['Gold', 'Silver']



## Example: <a name="graph_example"></a>
This example places the numbers 0-9 into a graph with some random number edge weight.


```python
import numpy as np

np.random.seed(8)

g = UndirectedGraph()
for i in range(0, 10):
    for j in range(0, 10):
        g.add_edge(i, j, np.random.randint(0, 100))
```

The expand function gives the list of (node, edge) tuples in the order that they are in the adjacency matrix.


```python
g.expand(4)
```




    [(0, 86.0),
     (1, 31.0),
     (2, 20.0),
     (3, 45.0),
     (5, 2.0),
     (6, 66.0),
     (7, 73.0),
     (8, 16.0),
     (9, 28.0)]



The expand function has an option for sorting the edge weights. The default sort is ascending order.


```python
g.expand(4, sort=True)
```




    [(5, 2.0),
     (8, 16.0),
     (2, 20.0),
     (9, 28.0),
     (1, 31.0),
     (3, 45.0),
     (6, 66.0),
     (7, 73.0),
     (0, 86.0)]



The expand function also has an option for reversing the order of the list, whether it is sorted or not. This is what happens with a sorted list.


```python
g.expand(4, sort=True, reverse=True)
```




    [(0, 86.0),
     (7, 73.0),
     (6, 66.0),
     (3, 45.0),
     (1, 31.0),
     (9, 28.0),
     (2, 20.0),
     (8, 16.0),
     (5, 2.0)]



Once the graph is created, we could do some something like this to walk through it.


```python
start = np.random.choice(g.nodes)
path = [start]
curr = start

for _ in range(0, 10):
    expansion = g.expand(curr)
    nodes = np.array([tup[0] for tup in expansion])
    edges = np.array([tup[1] for tup in expansion])
    prob = [edge/edges.sum() for edge in edges]
    curr = np.random.choice(nodes, p=prob)
    path.append(curr)
    
print('Path through graph: {}'.format(path))
```

    Path through graph: [8, 2, 8, 9, 0, 7, 9, 7, 4, 0, 8]
    
