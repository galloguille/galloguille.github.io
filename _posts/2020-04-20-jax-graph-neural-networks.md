---
layout:     post
title:      Implementing Graph Neural Networks with JAX 
date:       2020-04-20 20:05:00
comments:   true
summary:    "In this post I explain how to implement two different Graph Neural Networks using JAX. I start with an introduction to JAX and move to the implemententations of Graph Convolutional Neural Networks and Graph Attention Networks."
categories: deep-learning
---

I had wanted to do something with JAX for a while, so I started by checking the [examples in the main repository](https://github.com/google/jax/tree/master/examples) and tried doing a couple of changes. The examples are easy to follow, but I wanted to get a deeper understanding of it, so after a choppy attempt with some RL algorithms, I decided to work on something I had implemented before and went for two different Graph Neural Networks papers. I have to say that for an early stages library (version 0.1.59 as I started writing this), the process was reasonably smooth. 
In this post I'll talk about my experience on how to build and train Graph Neural Networks (GNNs) with JAX.

I focus on the implementation of Graph Convolutional Networks and Graph Attention Networks and I assume familiarity with the topic, not with JAX. For details about the models, check their papers ([GCN](https://arxiv.org/abs/1609.02907), [GAT](https://arxiv.org/abs/1710.10903)) or their accompanying posts ([GCN](https://tkipf.github.io/graph-convolutional-networks/), [GAT](https://petar-v.com/GAT/)). For those not familiar with JAX, I start with an introduction and a simple linear regression example.

## 1. What is JAX

[JAX](https://github.com/google/jax) is a Machine Learning library which I would describe (very vaguely) as Numpy with auto differentiation that you can execute on GPUs (and TPUs too!). Additionally, it has XLA compilation and built-in vectorization and parallelization capabilities.

Before jumping straight to the Graph Neural Network specifics, let me review the basics of JAX. Feel free to skip this section if you already know it.

The most important JAX feature to implement neural networks is autodiff, which lets us easily compute derivatives of our functions.


First of all, let's see how to use it for a simple function like ReLU.

```python
import jax.numpy as np
from jax import grad

def relu(x):
    return np.maximum(0, x)

relu_grad = grad(relu)
```

By calling `grad()` on a function, JAX returns another function that computes the gradient of the first function, which we can use for any input value.

For example:

```python
x = 5.0
print(relu(x))
print(relu_grad(x))
```

Will print `5.0` and `1.0`, which are the values of `relu(5.0)` and the derivative of ReLU at x=5.0.
And not just that, but we can also easily compute higher order derivatives by chaining multiple calls of `grad()`. For example, we can compute the 2nd derivative like this:

```python
relu_2nd = grad(grad(relu))
print(relu_2nd(x))
```

As we would expect, `relu_2nd(x)` will evaluate to `0.` for any value of `x`, as ReLU is a piecewise linear function without curvature.

In the same way, with `jax.grad()` we can compute derivatives of a function with respect to its parameters, which is a building block for training neural networks. For example, let's take a look at the following simple linear model and see how to compute the derivatives w.r.t its parameters for an input value of `x = 5.0`:

```python
def linear(params, x):
    w,b = params
    return w*x + b

grad_linear = grad(linear)

# initial parameters
w, b = 1.5, 2.
params_grad = grad_linear((w, b), 5.0)
```

What we are doing in the previous code snippet is the following: first we define the function `linear(params, x)` which gets as arguments the parameters of the linear model `(w, b)` and the data point `x` to get a prediction on. Then, calling `grad(linear)` gives us a function that computes the gradient of `linear` w.r.t its first argument (the linear model parameters.) Thus, `params_grad` will be a vector with two values, the first one is the derivative w.r.t `w` and the second one is the derivative w.r.t `b`. 
We can go one step further and compute the gradient of a loss function w.r.t the linear model parameters:

```python
def loss(params, dataset):
    x, y = dataset
    pred = linear(params, x)
    return np.square(pred - y).mean()

loss_grad = grad(loss)
# (5.0, 2.0) are the made up values for x and y
params_grad = loss_grad((w, b), (5.0, 2.0))
```

For the example I've set `x=5.0` and `y=2.0` arbitrarily, and then compute the gradient of the loss function for that arbitrary label. Notice how `loss` is a function of `linear` as well, and `loss_grad()` will compute the gradient w.r.t to the parameters, chaining the gradient of `loss` and the gradient of `linear`.

With all these pieces, we can write a small piece of code that trains a linear model:

```python
import numpy.random as npr
import jax.numpy as np
from jax import grad

# first generate some random data
X = npr.uniform(0, 1, 300)
true_w, true_b = 2, 1
# add some noise to the labels
Y = X*true_w + true_b + 0.4*npr.randn(300)

# the linear model
def linear(params, x):
    w,b = params
    return w*x + b
    
def loss(params, dataset):
    x, y = dataset
    pred = linear(params, x)
    return np.square(pred - y).mean()

# gradient function
loss_grad = grad(loss)

iterations = 500
step_size = 0.01
dataset = (X, Y)
w, b = 1.5, 2. # initial values for the parameters
for i in range(iterations):
    params = (w, b)
    loss_ = loss(params, dataset)
    # compute gradient w.r.t model parameters
    params_grad = loss_grad(params, dataset)
    # update parameters
    w -= step_size * params_grad[0]
    b -= step_size * params_grad[1]
    print(loss_)
```

We can visualize the final model (orange line) and compare it to the true data generating model (red line), and we see that we didn't get too far:

![](https://i.imgur.com/ejj3sid.png)

Well, I guess that's enough of JAX basics, in the next sections you'll see that the GNN implementations are not that different from this simple example.

## 2. Graph Neural Networks

Graph Neural Networks (GNNs) are neural networks that we can apply to graph structured data in a similar way that we apply Convolutional Neural Networks (CNNs) to images or Recurrent Neural Networks (RNNs) to sequences.

In this post I'll focus on Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs), but there are several more models. Check [this paper](https://arxiv.org/abs/1806.01261) for a detailed explanation of different kinds of models. It is a nice overview and very useful to get an understanding of them.

### 2.1 Graph Convolutional Networks
First defined in the paper [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) these models are very fast, and also easy to implement, using the normalized adjacency matrix of the graph to propagate information between neighbouring nodes. You can check [this post](https://tkipf.github.io/graph-convolutional-networks/) for more details.

Basically, a graph convolutional layer is defined as:

$$H^{(l+1)} = ReLU \left( \hat{A}H^{(l)}\Theta^{(l)} \right) $$

where \\( H^{(L)} \\) is the input to the layer, a \\( N \times C \\) matrix with as many rows and columns as nodes and input features respectively. \\( \hat{A} \\) is the degree normalized adjacency matrix and \\( \Theta^{(l)} \\) is a \\( C \times F \\) matrix of learnable parameters.

### 2.2 Graph Attention Networks
This is another family of GNNs that we proposed in the paper [Graph Attention Networks](https://arxiv.org/abs/1710.10903). Instead of using the values in the normalized adjacency matrix to propagate information, these models compute attention coefficients between neighbouring nodes, and use those coefficients when propagating the nodes' features. Check [this post](https://petar-v.com/GAT/) for more details.

A layer can still be defined as:

$$ H^{(l+1)} = ReLU \left( \hat{A}H^{(l)}\Theta^{(l)} \right) $$

but this time, the propagation matrix \\( \hat{A} \\) is not the normalized adjacency matrix, but a matrix whose elements are attention coefficients computed between neighbouring nodes:

$$ \hat{A}_{ij} = \frac{exp(e_{ij})}{\sum_{k \in \mathcal{N}} exp(e_{ik}) } $$

with the unnormalized coefficient \\( e_{ij} \\) between each pair of nodes \\( (i,j) \\) being a function of their features:

$$ e_{ij} = a(\vec{h_i}, \vec{h_j}) $$

The other difference between GAT layers and GCN layers is that GAT layers use a MultiHead approach, similar to the attention proposed in the [original Transformer paper](https://arxiv.org/abs/1706.03762). With this approach each layer consists on several independent attention heads whose output is concatenated. 

## 3. JAX implementation

For my GNNs implementations, I based my code structure on the [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py) package in `jax.experimental`, which implements several neural network layers. The whole idea is that a layer decouples its parameters from its forward computation, and the parameters are always passed as an argument to the forward function. If you are familiar with Pytorch, this would be like calling `layer(params, x)` instead of `layer(x)` to compute the forward pass of a layer.

Therefore, to define a layer in JAX we have to define two functions:
1. `init_fun`: initializes the parameters of the layer.
2. `apply_fun`: defines the forward computation function.

Something like this:
```python
def Layer():
    def init_fun(*args, **kwargs):
        # initialize parameters and compute output shape
        return output_shape, params
    def apply_fun(params, x, *args, **kwargs):
        # process the input and return the output
        return out
```

A model is nothing more than a collection of layers, so it is defined in the same way, an `init_fun` that will call the initializers of the layers, and an `apply_fun` that will define the forward computation of the model from input to output by chaining different layers and activation functions.

Let's see how to do that for the first model, GCNs.


### 3.1 Graph Convolutional Networks

Let's start by defining a graph convolutional layer, which is the building block of a GCN. As I said before, we need to define an initilization function and a forward function for the layer.
The initialization function takes care of initializing the parameters of the layer, which in this case are a linear projection matrix and a vector of biases.

```python
out_dim = 64
def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    k1, k2 = random.split(rng)
    W_init, b_init = glorot_uniform(), zeros
    W = W_init(k1, (input_shape[-1], out_dim))
    b = b_init(k2, (out_dim,)) if bias else None
    return output_shape, (W, b)
```
The initialization function only needs two arguments:
 * `rng`: a random key used to generate random numbers
 * `input_shape`: a tuple indicating the input shape, necessary to know the shape of the layer parameters.

And returns two values:
* `output_shape`: the shape of the output computed by the layer.
* `(W, b)`: the initialized parameters.

The reason for receiving and returning input and output shapes is that we will chain these initialization functions when creating a model with multiple layers, so each layer will know exactly which input shape to expect.

Finally, the forward function can be easily defined as:

```python
def apply_fun(params, x, adj, **kwargs):
    W, b = params
    support = np.dot(x, W)
    out = np.matmul(adj, support)
    if bias:
        out += b
    return out
```

Notice how the parameters are an argument to the function? This makes it a pure function since it only depends on its inputs, instead of using values stored in global variables or class attributes. `params` is a tuple of parameters, like the one returned by `init_fun`. Additionally, a GCN layer needs the adjacency matrix of the graph to propagate information in the graph with `np.matmul(adj, support)`.

After defining these two functions, we can put them together to form a Graph Convolutional Layer:

```python
def GraphConvolution(out_dim, bias=False):
    """
    Layer constructor function for a Graph Convolution layer 
    as the one in https://arxiv.org/abs/1609.02907
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W_init, b_init = glorot_uniform(), zeros
        W = W_init(k1, (input_shape[-1], out_dim))
        b = b_init(k2, (out_dim,)) if bias else None
        return output_shape, (W, b)

    def apply_fun(params, x, adj, **kwargs):
        W, b = params
        support = np.dot(x, W)
        out = np.matmul(adj, support)
        if bias:
            out += b
        return out

    return init_fun, apply_fu
```
On the forward step, this layer will project the input nodes' features using a learned projection defined by `W` and `b` and then propagate them according to the normalized adjacency matrix.

To use these layers to create a Graph Convolutional Network, we follow the same approach for the model, defining an `init_fun` and an `apply_fun` functions for the model.

First, we call the layer functions like this:

```python
gc1_init, gc1_fun = GraphConvolution(nhid)
_, drop_fun = Dropout(dropout)
gc2_init, gc2_fun = GraphConvolution(nclass)
```

The `GraphConvolution` function is called twice to generate two initialization and forward functions, one of each for each layer. You can also see that we are using a `Dropout` layer too. I'm not going to explain it here since it follows the exact same pattern as the `GraphConvolution` layer, but I want you to notice that I discard the initializer of the layer, since it doesn't have parameters to initialize. You can check the exact implementation of the Dropout layer [here](https://github.com/gcucurull/jax-gcn/blob/master/models.py#L11).
Keep in mind that up to this point we haven't initialized or used these layers yet, we have just instantiated their initialization and forward functions.

With that, we have all we need to define the `init_fun` for the GCN model:

```python
def init_fun(rng, input_shape):
    init_funs = [gc1_init, gc2_init]
    params = []
    for init_fun in init_funs:
        rng, layer_rng = random.split(rng)
        input_shape, param = init_fun(layer_rng, input_shape)
        params.append(param)
    return input_shape, params
```

This function is initializing the model layers and storing their parameters in `params`. Then, following the same pattern as the layers' `init_fun` function, it returns the output shape and the parameters of the model, which are stored in a list with one item per layer.

The other function that we have to define is the `apply_fun` for the GCN model:

```python
def apply_fun(params, x, adj, is_training=False, **kwargs):
    rng = kwargs.pop('rng', None)
    k1, k2, k3, k4 = random.split(rng, 4)
    x = drop_fun(None, x, is_training=is_training, rng=k1)
    x = gc1_fun(params[0], x, adj, rng=k2)
    x = nn.relu(x)
    x = drop_fun(None, x, is_training=is_training, rng=k3)
    x = gc2_fun(params[1], x, adj, rng=k4)
    x = nn.log_softmax(x)
    return x
```

This function uses the `gc1_fun`, `drop_fun` and `gc2_fun` that we have obtained before, and basically defines the forward pass of the full model. Easy, right? Also notice how I used an additional argument called `is_training`. This is a boolean flag used by Dropout to change its behaviour at eval time.

Putting all this pieces together, we can build a GCN model like this:

```python
def GCN(nhid: int, nclass: int, dropout: float):
    """
    This function implements the GCN model that uses 2 Graph Convolutional layers.
    """
    gc1_init, gc1_fun = GraphConvolution(nhid)
    _, drop_fun = Dropout(dropout)
    gc2_init, gc2_fun = GraphConvolution(nclass)

    init_funs = [gc1_init, gc2_init]

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)
        k1, k2, k3, k4 = random.split(rng, 4)
        x = drop_fun(None, x, is_training=is_training, rng=k1)
        x = gc1_fun(params[0], x, adj, rng=k2)
        x = nn.relu(x)
        x = drop_fun(None, x, is_training=is_training, rng=k3)
        x = gc2_fun(params[1], x, adj, rng=k4)
        x = nn.log_softmax(x)
        return x
    
    return init_fun, apply_fun
```

Let's see how to use the same pattern for Graph Attention Networks now.

### 3.2 Graph Attention Networks

For Graph Attention Networks we follow the exact same pattern, but the layer and model definitions are slightly more complex, since a Graph Attention Layer requires a few more operations and parameters. This time, similar to Pytorch implementation of Attention and MultiHeaded Attention layers, the layer definitions are split into two: 

1. `GraphAttentionLayer`: implements a single attention layer, equivalent to one head.
2. `MultiHeadLayer`: implementes the multi-head logic, using several `GraphAttentionLayer`.
 
Let's start with the Graph Attention Layer definition:

```python
def GraphAttentionLayer(out_dim, dropout):
    """
    Layer constructor function for a Graph Attention layer.
    """
    _, drop_fun = Dropout(dropout)
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2, k3, k4 = random.split(rng, 4)
        W_init = glorot_uniform()
        # projection
        W = W_init(k1, (input_shape[-1], out_dim))

        a_init = glorot_uniform()
        a1 = a_init(k2, (out_dim, 1))
        a2 = a_init(k3, (out_dim, 1))

        return output_shape, (W, a1, a2)
       
    def apply_fun(params, x, adj, rng, activation=nn.elu, is_training=False, 
                  **kwargs):
        W, a1, a2 = params
        k1, k2, k3 = random.split(rng, 3) 
        x = drop_fun(None, x, is_training=is_training, rng=k1)
        x = np.dot(x, W)

        f_1 = np.dot(x, a1) 
        f_2 = np.dot(x, a2)
        logits = f_1 + f_2.T
        coefs = nn.softmax(
            nn.leaky_relu(logits, negative_slope=0.2) + np.where(adj, 0., -1e9))

        coefs = drop_fun(None, coefs, is_training=is_training, rng=k2)
        x = drop_fun(None, x, is_training=is_training, rng=k3)

        ret = np.matmul(coefs, x)

        return activation(ret)

    return init_fun, apply_fun
```
The layer looks a bit more complicated, but it isn't much different from the previous `GraphConvolution` layer. First, it projects the input nodes to a new space with `W` and then it propagates the nodes' features with `np.matmul(coefs, x)`, as we did before. The main difference is that the values in `coefs` are attention coefficients computed from the node features, instead of coming from the adjacency matrix.
`coefs` is build by computing an attention coefficient between each pair of nodes, and then using softmax over all the attention coefficients for each node, to normalize them. The input to the softmax is masked out to consider only the direct neighbours to each node.

With this layer, we can easily build the multi-head mechanism following the same pattern of writing and `init_fun` and an `apply_fun`:

```python
def MultiHeadLayer(nheads: int, nhid: int, dropout: float, last_layer: bool=False):
    layer_funs, layer_inits = [], []
    for head_i in range(nheads):
        att_init, att_fun = GraphAttentionLayer(nhid, dropout=dropout)
        layer_inits.append(att_init)
        layer_funs.append(att_fun)
    
    def init_fun(rng, input_shape):
        params = []
        for att_init_fun in layer_inits:
            rng, layer_rng = random.split(rng)
            layer_shape, param = att_init_fun(layer_rng, input_shape)
            params.append(param)
        input_shape = layer_shape
        if not last_layer:
            # multiply by the number of heads
            input_shape = input_shape[:-1] + (input_shape[-1]*len(layer_inits),)
        return input_shape, params
    
    def apply_fun(params, x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)
        layer_outs = []
        assert len(params) == nheads
        for head_i in range(nheads):
            layer_params = params[head_i]
            rng, _ = random.split(rng)
            layer_outs.append(layer_funs[head_i](
                    layer_params, x, adj, rng=rng, is_training=is_training))
        if not last_layer:
            x = np.concatenate(layer_outs, axis=1)
        else:
            # average last layer heads
            x = np.mean(np.stack(layer_outs), axis=0)

        return x

    return init_fun, apply_fun
```

As you can see, we instantiate as many `GraphAttentionLayer` layers as the number of heads, and store each of their `att_init` and `att_fun` functions into two lists. Then, we write an `init_fun` for the `MultiHeadLayer` which will initialize each head and compute the appropriate output shape.
The `apply_fun` is not very different from the others, in this case we run the input through each head and then concatenate their outputs (or average them for the last layer).

With this two pieces, the Graph Attention Model definition is quite straightforward:

```python
def GAT(nheads: List[int], nhid: List[int], nclass: int, dropout: float):
    """
    Graph Attention Network model definition.
    """

    init_funs = []
    attn_funs = []

    nhid += [nclass]
    for layer_i in range(len(nhid)):
        last = layer_i == len(nhid) - 1
        layer_init, layer_fun = MultiHeadLayer(nheads[layer_i], nhid[layer_i],
                                    dropout=dropout, last_layer=last)
        attn_funs.append(layer_fun)
        init_funs.append(layer_init)

    def init_fun(rng, input_shape):
        params = []
        for i, init_fun in enumerate(init_funs):
            rng, layer_rng = random.split(rng)
            layer_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
            input_shape = layer_shape
        return input_shape, params

    def apply_fun(params, x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, len(attn_funs))

        for i, layer_fun in enumerate(attn_funs):
            x = layer_fun(params[i], x, adj, rng=rngs[i], is_training=is_training)
        
        return nn.log_softmax(x)

    return init_fun, apply_fun
```

Something worth mentioning here is the difference between the `params` argument for the `GAT` model and the `params` for the `GCN` model. Did you notice anything different?

The difference is their structure. For `GCN`s, `params` is a `List` of `Tuple`, where each `Tuple` holds the `(W, b)` values for each layer. Therefore, it looks something like this:

```python
params = [
    (W_0, b_0), # first layer
    (W_1, b_1)  # output layer
]
```

For `GAT`s, the structure is a bit different, because we have different heads at each layer. Therefore, the structure of the `params` argument is a `List` of `List` of `Tuple` like this:

```python
params = [
    [ # first layer
        (W_0, a1_0, a2_0), # first head
        (W_1, a1_1, a2_1), # second head
        ...
        (W_k, a1_k, a2_k)  # k-th head
    ],
    [ # output layer
        ...
    ]
]
```

You could think that this change in the structure would mean that we will have to adapt our code to support both of them. However, that is not the case, because `jax.grad()` will return the gradients of the loss w.r.t `params` with the same structure as `params`, and the optimizers only need `params` and `grads` to have the same structure, but can work with arbitrary structures, so we don't have to worry about that.

### 3.3 Main loop

If you made it to this point, you should have an idea on how to define the models: via an `init_fun` and an `apply_fun`. Once we have defined the model functions, all we have to do to use them is to write the standard training loop, as well as the loss function and the optimizer.

Let's start by instantiating and initializing a model, for example a GCN.

```python
init_fun, predict_fun = GCN(nhid=hidden, 
                            nclass=labels.shape[1],
                            dropout=dropout)
input_shape = (-1, n_nodes, n_feats)
rng_key, init_key = random.split(rng_key)
_, init_params = init_fun(init_key, input_shape)
```

As we did individually for each layer, we first call the model function `GCN()` with the desired configuration, and we get the initilization function `init_fun` and the forward computation function `predict_fun`.
Next, we get the initial model parameters `init_params` as a List by using the `init_fun` function.

Now we only have a few things left: define the loss function, the optimizer, and the main loop.

The loss functions in JAX should also be pure functions. They get the model parameters and the input data as arguments, like this:
```python
def loss(params, batch):
    """
    The idxes of the batch indicate which nodes are used to compute the loss.
    """
    inputs, targets, adj, is_training, rng, idx = batch
    preds = predict_fun(params, inputs, adj, is_training=is_training, rng=rng)
    ce_loss = -np.mean(np.sum(preds[idx] * targets[idx], axis=1))
    l2_loss = 5e-4 * optimizers.l2_norm(params)
    return ce_loss + l2_loss
```
Here the input data consists on a list of the input nodes features and their true labels, along with the adjacency matrix, the `is_training` label, the random key and the set of node indexes that we want to compute the loss on. These indexes change at train and eval time, so they also have to be an input to the loss function. Notice how the parameters are the first argument of the loss function, while everything else is packed as a second argument. The reason is that when computing the gradient of the loss function w.r.t the parameters, JAX assumes by default that the first argument are the parameters. This works for most use cases, but it can always be changed to suit one's specific needs.

The reason for passing the random key around as an argument is that this way, the functions depend uniquely on their arguments, not on an external random key defined somewhere else, making them true pure functions.

For the optimizer I used the `optimizer` package from `jax.experimental` because I wanted to use ADAM to replicate the papers, but we could easily write our own SGD optimizer similarly to what I have shown for the linear regression model. The optimizer is defined as:

```python
opt_init, opt_update, get_params = optimizers.adam(0.001)
opt_state = opt_init(init_params)
```

See how we create the optimizer state from the initial values of the model parameters. Then, whenever we want to retrieve the parameters, to use them as an argument for example, we can use `get_params(opt_state)`. Finally, `opt_update` is the function that we will use to update the parameters based on the gradient of the loss, and is used like this:

```python
opt_state = opt_update(i, grad(loss)(params, batch), opt_state)
```

where `i` is the iteration number and `grad(loss)(params, batch)` are be the gradients of the loss function w.r.t the parameters of the model, as we have seen before in the linear regression example.

All that's left is to wrap the loss computation and parameter update into a single function:

```python
def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)
```

And with this we can write a simple training loop that will train our graph neural network models:

```python
print("\nStarting training...")
for epoch in range(num_epochs):
    start_time = time.time()
    # define training batch
    batch = (features, labels, adj, True, rng_key, idx_train)
    # update parameters
    opt_state = update(epoch, opt_state, batch)
    epoch_time = time.time() - start_time

    # validate
    params = get_params(opt_state)
    eval_batch = (features, labels, adj, False, rng_key, idx_val)
    val_acc = accuracy(params, eval_batch)
    val_loss = loss(params, eval_batch)
    print(f"Iter {epoch}/{num_epochs} ({epoch_time:.4f} s) val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    # new random key at each iteration, othwerwise dropout uses always the same mask 
    rng_key, _ = random.split(rng_key)
```

I'm not showing how to load the dataset and preprocess the data, since that is dataset specific, but you can check my full implementation of [Graph Convolutional Networkss in JAX](https://github.com/gcucurull/jax-gcn) to see the full training script using the Cora dataset, as well as the use of `@jax.jit` to speed up the training. Also, I haven't explained how to use GATs instead of GCNs because the way to use them is the same. If you want to see it, check also my repository implementing [Graph Attention Networks in JAX](https://github.com/gcucurull/jax-gat).

## 4. Other JAX resources
I hope this post has helped you understand JAX and how to use it. While I was doing my implementation of these two models a resource that I found very useful and educative was Sabrina Mielke's post "[From PyTorch to JAX: towards neural net frameworks that purify stateful code](https://sjmielke.com/jax-purify.htm)". I encourage everyone to give it a good read. If you want to check other JAX codebases, you can start with [Flax](https://github.com/google/flax) and [Haiku](https://github.com/deepmind/dm-haiku), two neural networks libraries that use JAX by Google and Deepming respectively. Additionally, if you are interested in Reinforcement Learning, [RLax](https://github.com/deepmind/rlax) uses JAX to implement some RL algorithms. 

### Get in Touch
I will gladly answer any questions or discuss anything about the code, you can contact me at   `gcucurull at gmail dot com`. 
