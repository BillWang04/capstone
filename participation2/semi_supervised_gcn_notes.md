# Notes on Semi-Supervised Classificaiton askdfjals kdjflas

## 1. Introduction

$$
\mathcal{L} = \mathcal{L}_0 + \lambda \mathcal{L}_{reg}, \text{with } \mathcal{L}_{reg} = \sum_{i,j} A_{ij} ||f(X_i) - f(X_j)||^2 = = F(X)^T \delta f(X). 
$$


#### Regularization Idea

- $A_{ij}$ attributes that to whether or not the to verticies $v_i$ and $v_j$ are connected, or else the sum of the component will be 0 for $i,j$

- $||f(X_i) - f(X_j)||^2 $ describes the difference between the two features of nodes $i,j$. If features of $i,j$ are large, they will contribute a lot to the loss, 

The loss function + regularization allows for better smoothness, the smoothness happens as the  

X_i and X_j don't seem to change, how does regularization actually happen if really $\mathcal{L}$ is actually just a scalar for a specific graph


When looking at $\lambda$,  we know the larger $\lambda$ is, the more weight the regularization factor has. 


#### Premise
1. 
    - GCN (propegation directly on graphs directly)
    - show that there is first-oder approximation on spectral graph convolutions?????
2. 
    - Show that semi-superivsed classification of nodes in a graph

## 2. Fast approximate Convolutions on Graphs

Assuming that GCN is described as a function $f(X, A)$

**Discussion**

Single Layer:

$$
H^{(l+1)} = \sigma (\tilde{D}^{- \frac{1}{2}} \tilde{A} \tilde{D}^{- \frac{1}{2}} H^{(l)} W^{(l)}) . 
$$

- $\tilde{A} = A + I_N$ --> adjacency matrix undirected graph $G$ with added self connections is a degree matrix 

- $\tilde{D}$ is a degree matrix of summing across the rows of $\tilde{A}$

- $W^{l}$ --> trainable weights

- $H^{(l)}$ is considered the matrix of activations (input into the the fed layer, .i.e $H^{(0)}$ = X)

### 2.1 Spectral Graph Convolutions

$$
g_\theta * x = Ug_\theta U^Tx 
$$

**Direct Spectral Graph Convolution**

$U$ is the matrix of eigenvectors of the normalized graph Laplacian $L = I_N - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}= U \Lambda U^T$

- $\Lambda$ diagonal matrix of its eigenvalues

- $x$ is a graph signal

**Decomposition of above equation**

*Step 1: Change x to the Fourier Domain (Laplacian Basis)*
- $\tilde{x} = U^T x$, 

*Step 2: Do the Convolution in the Fourier Domain using*

- $\tilde{y} = g_\theta (\Lambda) * \tilde{x}$

*Step 3: Convert back to normal basis*

- $y = U * \tilde{y}$

**Interpretation**

#### Describe What a Graph signal exactly is 

- A graph signal is essentially the same information about a specific node on the graph, ie  we had three features for each node, a signal would be all nodes across one feature

- if the graph has $N$ nodes, the signal is a vector $x \in \mathbb{R}^N$, where $x_i$ is the value of the signal at node i

- we can represent $X$ as a $[x_1, x_2, \cdots, x_n]$ for all signals represent nodes of $A$ respectively

#### Describe what a Laplacian expresses 

- Laplacian is $D - A$, so the diagonal of the matrix is degree while the offsets are the strength of each connection for a specific node (negative)

- We can think of Laplacian Matrix as a function for graph smoothness for some feature by the expression
$$
x^T L x = \sum_{i,j}A_{i,j}(x_i - x_j)^2
$$

where $x$ is a graph signal

- The sum quantifies how much the signal values $x_i, x_j$ and 
$x_i, x_j$​ differ for connected nodes i and j. If neighboring nodes have very different signal values, the quadratic form will be large, indicating that the signal is not smooth. If the signal is constant or nearly the same for connected nodes, the value will be small, indicating smoothness.

- When the expression $x_i - x_j == 0$, that means that the feature is the same for those two neighbors 

- When the $A_{i,j}$ is 0, the nodes are not connected 


#### Reason why we want to express Laplacian in decomposed eigen- signals and running a filter

$$Lu_i = \lambda_i u_i$$

- Thus, if we see that $x^T L x = x^T \lambda_i x$ where $\lambda_i$ is large, we know that there is a lack of smoothness

- The Eigenvectors and their corresponding eigenvalues represent signals and the weight of each signal respectively like classical fourier transform

- Consider that inputting a graph signal $x$ into the Laplacian matrix, we can express $x$ in the fourier domain (laplacian basis) to decomposes its elements into the eigenvectors and eigenvalues of $L$ 


**Reasons to change basis**

- The Laplacian in the standard basis is not interpretable and may not reveal useful patterns or structure in the signal

    - Eigenvectors corresponding to small eigenvalues (low frequencies) capture smooth, global variations in the graph signal—where neighboring nodes have similar signal values.

    - Eigenvectors corresponding to large eigenvalues (high frequencies) capture local variations and noise, where the signal changes rapidly between neighboring nodes.

- We can filter eigenvectors, allowing for better us to focus smoother or shaper signals between neighbors

**Types of filters perhas**
- A low-pass filter might suppress the high eigenvalues (high frequencies) and keep the low ones, resulting in a smoother signal.

- A high-pass filter would do the opposite, enhancing sharp changes between neighboring nodes.


#### Describe Chebyshev polynomials

It is computationally difficult to calculate large spectral convolutions on graphs due to the fact that multiplying eigenvector matrix U is $O(N^2)$ (Multiply x (N x 1) with U^T (N x N))

Moreover, the direct convolution is global, meaning the entire graph strucutre is influenced by every node. We want localized filters that focus on the node's neighborhood.

Thus, $g_\theta(\Lambda)$ can be well approximated by truncated expansion in terms of Chebyshev polynomials $T_k(x)$ up to the K^th order. Chebyshev polynomials $T_k(x)$ are a sequence of orthogonal polynomials. Thus, we approximate the filter on the Laplician Eigenvalues as a way for faster computation and so it avoids the need for eigenvector decomposition by approximating the filter in a localized manner up to K-th order:

**Convolution in the Spectral Domain**

$$
g_{\theta '}(\Lambda) \approx \sum_{k = 0}^{K} \theta_k ' T_k (\tilde{\Lambda})
$$

with a rescaled $\tilde{\Lambda} = \frac{2}{\lambda_{max}} \lambda - I_N $. $\lambda_{max} $ denotes the largest eigenvalue of L. $\theta ' \in \mathbb{R}^K$ is now a vector of Chebyshev coefficients. (Reference to Wavelets on graphs via spectral graph theory Hammond et al. 2011)

Regularization is done for $\tilde{\Lambda} = \frac{2}{\lambda_{max}} \lambda - I_N$ 
- numerical stability (exploding and vanishing value) and rounding errors implemented by computational errors implemented by computer.
- Rescaling for convergence 
    - There are apparent convergence properties of the Chebyshev polynomials but the range must be between [-1,1]


**Convolution in the Node (spatial Domain)**

$$
g_{\theta'} \star x \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{L}) x
$$

By operating directly on the Laplacian operation, we are able to convolve without U, avoiding eigenvector decomposition and instead doing basic matrix multiplication, so we lose the inherent graph frequencies of the Laplacian Eigenvalues


with $\tilde{L} = \frac{2}{\lambda_{\text{max}}}L - I_N$ ; as can easily be verified by noticing that $(U \Lambda U^\top)^k = U \Lambda^k U^\top$. Note that this expression is now K-localized since it is a K-th order polynomial in the Laplacian, i.e., it depends only on nodes that are at maximum steps away from the central node (K-th order neighborhood). The complexity of evaluating Eq. 5 is $\mathcal{O}(|\mathcal{E}|)$, i.e., linear in the number of edges. **Defferrard et al. (2016)** use this K-localized convolution to define a convolutional neural network on graphs.


*What is the point talking about about the spectral domain when we use the chebyshev polynomials directly on the Laplacian?*

Perhaps its a way to for the model to learn their own shit?

### 2.2 Layer-Wise Linear Model

- When k = 1 for the approximation for the graph convolution is further simplified by setting K = 1, which means only the first two terms of the Chebyshev polynomial expansion are used (Linear approximation for the filter). 

- Because of this, this allows for the problem of overfitting as now we won't have to worry about graphs with very wide node degree distributions due to the normalization (high degree nodes would dominate the graph convolution operation) imagine a graph with a node with 1000000 degrees and a node with 2 degrees

- We can also have $\lambda_{max} \approx 2$ as we expect the neural network paramets will adapt to this change in scale during training.

With all these approximations

$$
g_{\theta '} * x \approx \theta_0 'x + \theta_1 ' (L- I-N)x = \theta_0 ' - \theta_1 ' D^{- \frac{1}{2}}AD^{-\frac{1}{2}}x
$$


- with two free parameters $\theta_0 '$ and $\theta_1 '$

- Successive iterations convolve the $k^{th}$ - order neighborhood of a node
- We minimize the parameters further to address overfitting????? and to decrease the number of operations per layer

$$
g_{\theta '} * x \approx \theta(I_N + D^{- \frac{1}{2}}AD^{-\frac{1}{2}})x
$$

- single parameter $\theta = \theta_0 ' = - \theta_1 '$

- because repeated application of this operator will create exploding/vanishing gradients when used in deep neural network, we have to d a *renormalization trick* $I_N + \tilde{D}^{- \frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$ with $\tilde{A} = A + I_N$ and $\tilde{D}_{ii} = \sum_{j}\tilde{A}_{ij} $

- Thus:
    - Consider that a signal is $X \in \mathbb{R}^{N x C}$ with $C$ input channels(C dimensional feature vector for every node) and $F$ filters of feature maps:
    $$
    Z = \theta(I_N + D^{- \frac{1}{2}}AD^{-\frac{1}{2}})X\Theta
    $$    
    - $\Theta \in \mathbb{R}^{CxF} $ is a filter parameter
    - Efficient as complexity $O(|\epsilon|FC)$ 


## 3. Semi-Supervised Node Classification









