---
layout:     post
title:      Sparse Matrix Multiplication in JAX
date:       2020-06-03 22:05:00
comments:   true
summary:    "In this post I’ll share my sparse matrix multiplication implementation in JAX, which could be useful for other problems besides implementing graph neural networks."
categories: deep-learning
---

# Sparse Matrix Multiplication in JAX

In my recent post I showed my [implementation of two graph convolutional network models using JAX](http://gcucurull.github.io/deep-learning/2020/04/20/jax-graph-neural-networks/). For GCNs the convolution operator can be defined using a matrix multiplication with the adjacency matrix to propagate node features across the graph. I didn't mention it in the post, but Thomas Kipf correctly [pointed out in a comment](http://gcucurull.github.io/deep-learning/2020/04/20/jax-graph-neural-networks/#comment-4902236586) that a graph adjacency matrix is usually very sparse and that a standard matrix multiplication is a waste of memory and computation, since most multiplications will be with elements that are zero.

Unfortunately, JAX doesn't yet support sparse matrices as well as other libraries like Pytorch, Tensorflow or Numpy (via Scipy), so in my implementation I used a standard matrix multiplication and decided that a sparse implementation was a problem for the future. It turned out that that future was not very far, so I ended up implementing a very basic sparse matrix multiplication in JAX, based on the [Flax implementation of Message Passing Neural Networks](https://github.com/google/flax/blob/master/examples/graph/models.py) (a generalization of Graph Convolutional Networks).

In this post I'll share my sparse matrix multiplication implementation in JAX, which could be useful for other problems besides implementing graph neural networks. 

## 1. Sparse Matrices

Sparse matrices are just like normal matrices, but most of their entries are zero. This means that when doing a matrix multiplication with a sparse matrix, most of the computation is wasted by multiplying by zero. To see why, remember that the result of a multiplication \\( c = Ab \\) between a matrix \\( A \\) and a vector \\( b \\) is defined as:

$$ c_i = \sum^{N}_{j=0} A_{i,j} * b_j $$

If most elements \\( A_{i,j} \\) are zero we don't need to compute the product \\(A_{i,j}*b_j\\) because we already now the answer. It is zero. Moreover, we don't even need to store all these zeros if we don't need them to compute the multiplication.

### 1.1 Sparse Representations

There are several ways to represent sparse matrices without having to store all those zeros. A very intuitive one is to store which positions contain a value that is not zero, along with it's value. For example, the following matrix:

$$ \begin{pmatrix} 
0 & 2 & 0 & 4 \\ 
1 & 0 & 0 & 0 \\
0 & 3 & 0 & 2
\end{pmatrix}  $$

Could be stored with some structure like this one:

```
values =  [2, 4, 1, 3, 2]
rows =    [0, 0, 1, 2, 2]
columns = [1, 3, 0, 1, 3]
```

### 1.2 Sparse Multiplications

The key to implementing a simple sparse matrix multiplication is that we only care about the elements that are not zero. Since the product of two matrices \\( AB \\) is the dot product of the rows of \\(A \\) with the columns of \\(B \\), we can easily see that the non-zero columns of a row of \\(A \\) indicate which rows of a column of \\(B \\) we need to compute the result. Assuming that \\(B \\) has only one column (it is a vector), it looks like this:

<img src="https://i.imgur.com/hIzzBNU.png"></img>


We can go one step further and with a simple trick extend it to all rows at the same time:

![](https://i.imgur.com/T1gYwDj.png)


What I show in the figure above is that we can use the flat array `values` that holds the non-zero values of `A` and takes the rows that we need from `B` in the same order. Then, we do an elementwise product between the two, and we add together the values from the result that correspond to the same row in `A`. Let's see how to implement that in JAX.


## 2. Implementation in JAX

We need two functions to implement the approach described above:
* `jax.numpy.take`: An equivalent of `numpy.take` that takes elements from an array along a given axis. This is what we need to take the rows from `B` that we need, according to the columns of `A` that have non-zero elements.
* `jax.ops.segment_sum`: Similar to `tf.math.segment_sum` from Tensorflow. This handles the second step, where we need to sum the results grouped by the rows of `A`.

With this two functions, we can implement the sparse matrix multiplication as:

```python
@jax.partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    assert B.ndim == 2
    indexes, values = A
    rows, cols = indexes
    in_ = B.take(cols, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res
```

The use of `jax.partial(jax.jit, static_argnums=(2))` as a decorator is used to tell JAX that the 3rd argument `shape` is static. Therefore, it will compile the function for each different value of `shape` that it encounters.

## 3. Performance

In my [GCN implementation in JAX](https://github.com/gcucurull/jax-gcn) I tested the same model on Cora and Citeseer using dense and sparse matrix multiplications for the adjacency propagation (both with `jax.jit`). With the sparse implementation the models were between 2 to 3 times faster both in CPU and GPU, so there is a clear improvement on performance.

## 4. Conclusion

This won't be the most efficient way to do sparse matrix multiplications in JAX, since other sparse representations like CSR (Compressed Sparse Row) offer faster multiplications, but as I have shown above it gives a significant boost in performance and the code is very simple to implement.
I hope the explanation and this little snippet of code are useful to you, feel free to leave a comment if you are using a different approach to sparse matmul with JAX.