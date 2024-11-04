
## Lecture 2-2.5
### Feature Engineering for Machine Learning in Graphs

- When training traditional ML pipleine, a lot of times we are just obtain features to make a prediction. 



# Missing 







## Lecture 3

Assumptions:
- Homogenous Graph, All nodes are the same
- unweighted edges
- undirected

### Naive Approach
Add Adjancency matrix and features to the neural network
- sensitive to node ordering
- not applicable different sizes 
- a lot of parameters

### Desigining GNNs from CNNs


- Like sets, graphs don't have inherent ordering
**Permutation Invariance**: a property of mathematical function or algorithm that reamains unchanged even when the order of its inputs is altered. 

- So when inputting a graph in to the function $G = (A, X) \rightarrow \mathbb{R}^d$, we want $f(A_1, X_1) = f(A_2, X_2) $

Why we want permutation Invaraince?

_____

**Permutation Equivaraince**: Order of vector output changes respectivelty to the input of the Graph

Why do we want permutation equivaraince?

_____

Because most neural networks, switching order of input of vector will result in different answers, we don't want that to happen for GNNS because there is no inherent order in graphs!!!1


**KEY IDEA** DEsign node embeddings based on local network neighborhoods, sorta like a CNN but instead of looking at a specific chunk of an image or matrix, you look at specific parts of a graph


How do we aggregate each node embedding for the graph? How do we combine each node in previous layers??

- Average messeages from each neighbors:


$$
h_{v}^{0} = x_v
$$

$$
z_v = h_{v}^{(k)}
$$

$$
h_{v}^{(k+1)} = \sigma(W_k \sum_{u \in N(v)} \frac{h_{u}^{k}}{|N(v)|} + B_kh_{v}^{(k)}) , \forall k \in \{0, \cdots, k - 1\}
$$

Thus:

For as single GNN layer

Each Layer is consider a hopx


$$ 
A --> Adjancy Matrix
$$
$$
H^{k} = [(h_1^{(k)}) \cdots (h_{|V|}^{(k)})] \rightarrow 
H^{(k + 1)} = \sigma(D^{-1}AH^{(k)}W_k^T + H^{(k)} B_k^T)
$$


This formula has node invariance as summation aggregates all the nodes and no matter what oder of of summation, you will get same result


Considering all nodes in a graph, GCN computation is permutation equivariant due to the fact that the rows of the input node features and output embeddings are aligned (the inner computations are invariant). 

### Trainning GNNs

For unsupervised learing, there is node labels so must use the graph structure as the superevision

For supervised, we use loss function
#### Unsupervised

Using the sum of cross entropy loss of two nodes u, v and the dot product of embedding u and embedding v

Node simularity can be based on Random walks or even matrix factorization

THE FEATURES ARE USED TO FIND NODE SIMILARITy



#### Supervised training:

Directly train the model for a superivsed task, such as drug safety


#### Model Design
1. For each node, you have a small mini neural network consisting by finding the local nodes connected to it.

2. Define a loss function for all nodes to find similarity? perhaps randomwalk etc

3. Train in batches

4. Generate embeddings for nodes as needed and for thos that aren't trained (how do they generate the never trained on, is by using the parameters of the nodes that are trained, if so how do you know you got ever combination for each node)



https://rish-16.github.io/posts/gnn-math/

## Lecture 4

Basic Framework for GNN

Message $m_u^{(L)}$   + Aggregation $AGG^{(l)} (\{m_u^{( l )}, u \in N(v) \}, m_v^{(l)})$ = One layer of GNN






# things to look at


- LSTM Aggregation

- $l_2$ normalization

- $a_{vu}$ Attention weights/mechanism

- 



## Capstone Lecture 2

- Node Classification, Lnik prediction, community detection, network similarity



- NOde to latent space 




LEARNed aboug graph

bench mark 
atleast one graph benchmark


Transformers 














 





