# A generalization of transformer networks to graphs

## Abstract and Introduction
General Ideas Between NLP Transformers and Graph Tranformers

- Firstly, since graphs are not fully connected, the attention mechanism is a function of the neighborhood of connections between nodes and edges.
- Secondly, the positional encoding is laplacian instead of sinusoidal. 
-  Normalization in NLP is batch normalized which improves faster training and better generalization in performance

- Lastly, edge features are included which are crucial for some graph networks

## Proposed Architecture

### Sparcity of Graph Transformers

In NLP, it is *difficult* to find meaningful sparse interactions between words in a sentence/context, thus making it necessary for all words to attend to each other i.e fully connected graph.

However, in a Graph Transformer Architecture, graphs are arbitrarly connected which often depends on the domain, allowing for a rich source of inductive bias in the network, which makes it ideal and practical for nodes to attend to local node neighbors, the same as GNNs. 

### Positional Encodings

In NLP, the positinal encodings are done by sinosodial functions to preserve distance information.

$$
PE_{pos, 2i } = sin(pos/10000^{\frac{2i}{d_{model}}})
$$

$$
PE_{pos, 2i + 1} = con(pos/10000^{\frac{2i}{d_{model}}})
$$

This is a challenge for graphs because gnns should learn structural node information invariant to the node position

In some recent works in 2020, we use the graph structure to pre-compute the Laplaican eigenvectors and use them as node positional information. 

Positional Encoding layers are done only before the transformer step. 

### Input (Postional Encoding/ Vector Representation)

node features = $a_i \in \mathbb{R}^{d_n x 1}$
edge features = $B_{ij} \in \mathbb{R}^{d_e x 1} $

project onto $d$- dimensional hidden features $h_i^0$ and $e_{ij}^0$ respectively

$$
\hat{h_i^0} = A^0a_i + a^0 ; e_{ij}^0 = B^0\beta_{ij} + b^0
$$
where $A^0 \in \mathbb{R}^{dxd_n}, B^0 \in \mathbb{R}^{dxd_e}$ and $a^0, b^0 \in \mathbb{R}^d$ are the parameters of the linear projection layers

We then embed these parameters with Laplacian PE

$$
\lambda_i^0 = C^0 \lambda_i + c^0 ; h_i^0 = \hat{h_i^0} + \lambda_i^0
$$
where $C^0 \in \mathbb{R}^{dxk}$ and $c^0 \in \mathbb{R}^d$. Note that the Laplacian pe are only added to hte node features input layers and not during intermediate graph transformer layers

### Graph Transformer Layer without edge features


$$
h_i^{\ell+1} = O_h^{\ell} \, \Big\|_{k=1}^{H} \left( \sum_{j \in \mathcal{N}_i} w_{ij}^{k, \ell} V^{k, \ell} h_j^{\ell} \right)

$$
where 

$$
w_{ij}^{k, \ell} = \text{softmax}_j \left( \frac{Q^{k, \ell} h_i^{\ell} \cdot K^{k, \ell} h_j^{\ell}}{\sqrt{d_k}} \right)
$$

$
Q^{k,l}, K^{k,l}, V^{k,l} \in \mathbb{R}^{d_k x d}, O_h^l \in \mathbb{R}^{dxd}, k= 1
$ to H denote the number of attention heads and $\Big\|$ denotes concatenation

Q and K are queried and dot product to find similarity, afterwards and softmaxing -> used as attention weight for $V^{k,l}h_j^l$ and sum for all neighbors in attention block which are then concatenated for all the multi-head attentions into a vector

Afterwards, put $\hat{h}_i^{\ell + 1}$ through FFN preceded and succeded by residual connections and normalization layers

1. 
$$
\hat{\hat{h}}_i^{\ell+1} = \text{Norm} \left( h_i^{\ell} + \hat{h}_i^{\ell+1} \right)
$$
2.
$$
\hat{\hat{\hat{h}}}_i^{\ell+1} = W_2^{\ell} \, \text{ReLU} \left( W_1^{\ell} \hat{\hat{h}}_i^{\ell+1} \right)
$$
3.
$$
h_i^{\ell+1} = \text{Norm} \left( \hat{\hat{h}}_i^{\ell+1} + \hat{\hat{\hat{h}}}_i^{\ell+1} \right)
$$


### With Edges

JUST SEE THE PAPER MAN ITS LIKE THE SAME PLUS DOT PRODUCT OF E^K and eij in layer \ell







