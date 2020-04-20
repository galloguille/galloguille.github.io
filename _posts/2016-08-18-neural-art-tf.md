---
layout:     post
title:      Neural art style transfer in Tensorflow
date:       2016-08-18 22:21:58
comments: true
summary:    "In order to practise a little bit with Tensorflow I have implemented the paper A Neural Algorithm of Artistic Style. I think it is an easy algorithm to code, and I encourage all the people who want to learn about Deep Learning or the ones who are trying to learn how to use a new framework to implement it themselves."
categories: tensorflow style-transfer
---

Since it was released last November, I have read and learned a little bit about Tensorflow. The Udacity's [exercises in their github repo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity) are very good and the book by Jordi Torres [First Contact With Tensorflow](http://www.jorditorres.org/first-contact-with-tensorflow/) is an amazing introduction to the usage of Tensorflow. However, as it is said, practice makes perfect, so in order to really learn how to use Tensorflow one must get his hands dirty and code some real stuff.

I decided to implement the paper "A Neural Algorithm of Artistic Style" by Gatys *et. al* [(link)](http://arxiv.org/abs/1508.06576) for several reasons. First of all, the idea of merging the content of a picture with the style of another seemed really cool to me, and I was eager to learn how the algorithm worked. Secondly, the algorithm is quite easy to implement and the model can be build using standard Tensorflow components.

# Neural Art Style Transfer

Let's see how to use a Convolutional Neural Network to merge the style and content of two images.
Essentially, what we want is to capture the content of an image and the artistic style of another image, and create a new image which has the content of the first image represented with the style of the latter image.

This can be achieved by generating a random noise image and then changing its pixels values until its content representation is similar to the content representation of the original image and its artistic representation is similar to the artistic style representation of the style reference image.

Then we need to define two important things: how to represent the style and content of an image and how to change the random image pixels values to make it similar to the reference representations.

The first one is easy, we can just use the network's activations to represent the content and style. The content of the image will be represented by the activations of a chosen layer, and the artistic style will be represented by the correlation of different filters activations.

The activations on layer \\( l \\) are represented by \\( F^l \\) which is a matrix of size \\( N_l \times M_l \\) where \\( N_l \\) is the number of filters in that layer and \\( M_l \\) is the size (height times the width) of each feature map.

The feature correlations are computed by the [Gram matrix](https://inst.eecs.berkeley.edu/~ee127a/book/login/def_Gram_matrix.html) \\( G^l \\) of \\( F^l \\) such as:

$$G^l_{ij} = \sum_{k}F^l_{ik}F^l_{jk}$$

The dimensions of \\( G^l \\) will be then \\( N_l \times N_l \\) because it is representing the correlation between each pair of filter activations.

Now we have a way to represent the content \\( F^l \\) and style \\( G^l \\) of an image based on the activations for a given layer \\( l \\). First problem solved, now let's see how to create an image that merge them.

As we are working with neural networks the solution seems quite obvious. Anyone said backprogation? Because that's what we are going to use, we will feed forward the noise image through the network and backprogate the error given by a loss function to the image pixels, changing them to new values which reduce that error. The trick is to know which loss function to use.

The **loss function** proposed by Gatys *et. al* does the following:

* Minimize the difference between the content representation of the source image \\( \vec{p} \\) and the generated image \\( \vec{x} \\) for a given layer \\( l \\). With \\( P^l \\) and \\( F^l \\) being the content representations of these images in layer \\( l \\), the **content loss** is defined as:

    $$\mathcal{L}_{content}(\vec{p}, \vec{x}, l) = \frac{1}{2} \sum_{i,j}(F^l_{i,j} - P^l_{i,j})^2$$

* Minimize the difference between the style representation of the source style image \\( \vec{a} \\) and the generated image \\( \vec{x} \\) for a set of layers \\( L \\). With \\( A^l \\) and \\( G^l \\) being the style representations of these images layer \\( l \\), the **style loss** for that layer is defined as:

	$$E_l = \frac{1}{4N^2_lM^2_l}\sum_{i,j}(A^l_{i,j}-G^l_{i,j})^2$$

	And the style loss for all \\( L \\) layers is:

	$$\mathcal{L}_{style}(\vec{a}, \vec{x}) = \sum^L_{l}w_lE_l$$

	where \\( w_l \\) is the weighting factor of layer's \\( l \\) contribution to the style loss.

Once we have defined how to measure the error between the original style and content images with respect to the gnerated image, we get the **total loss** defined as:

$$\mathcal{L}_{total}(\vec{p}, \vec{a} ,\vec{x}) = \alpha\mathcal{L}_{content}(\vec{p}, \vec{x}) + \beta\mathcal{L}_{style}(\vec{a}, \vec{x})$$

where \\( \alpha \\) and \\( \beta \\) are the weighting factors for the contribution of the content loss and style loss to the total loss.

# Implementation
Until now I have explained how to merge the content and style of two images according to the [paper](http://arxiv.org/abs/1508.06576) by Gatys *et. al*:

1. Generate a white noise image \\( \vec{x} \\).

2. Compute the content representation \\( P^l \\) of the content image \\( \vec{p} \\) and the style representation \\( A^l, l \in L \\) of the style image \\( \vec{a} \\).

3. Perform gradient descent on the pixels of \\( \vec{x} \\) to minimize the total loss \\( \mathcal{L}_{total}(\vec{p}, \vec{a} ,\vec{x}) \\).

Now let's see how to do that in Tensorflow.

## The model
First of all we need to define a model, the network through that the images will be propagated to get their representations and which will be used to generate the new image. In my case I've adapted both Alexnet [cite] and VGG [cite] architectures pre-trained with ImageNet [cite].

So let's see how to **load the weights of a model to use it in Tensorflow**. I downloaded the VGG weights provided by Davi Frossard [here](http://www.cs.toronto.edu/~frossard/post/vgg16/) and saved them to a file named `vgg16_weights.npz`. Then the weights are loaded by:

```python
net_data = np.load(os.path.dirname(__file__)+"/vgg16_weights.npz")
```

This loads a dictionary in `net_data` which has an entry for every weight layer:

```python
net_data.keys()
['conv4_3_W', 'conv5_1_b', 'conv1_2_b', 'conv5_2_b', 'conv1_1_W', 'conv5_3_b', 'conv5_2_W', 'conv5_3_W', 'conv1_1_b', 'fc7_b', 'conv5_1_W', 'conv1_2_W', 'conv3_2_W', 'conv4_2_b', 'conv4_1_b', 'conv3_3_W', 'conv2_1_b', 'conv3_1_b', 'conv2_2_W', 'fc6_b', 'fc8_b', 'conv4_3_b', 'conv2_2_b', 'fc6_W', 'fc8_W', 'fc7_W', 'conv3_2_b', 'conv4_2_W', 'conv3_3_b', 'conv3_1_W', 'conv2_1_W', 'conv4_1_W']
```

As you can see only weight layers are stored in the file, so in order the build the same reference model I first define a list of layers such as:

```python
layers = []
layers.extend(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1'])
layers.extend(['conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2'])
layers.extend(['conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'maxpool3'])
layers.extend(['conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'maxpool4'])
layers.extend(['conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'maxpool5'])
```

And then I loop through that list to create each layer with the right parameters and load the weights if it is a weight layer:

```python
for layer in layers:
    layer_type = get_type(layer)

    if layer_type == 'conv':
        W_conv = tf.constant(net_data[layer+'_W'])
        b_conv = tf.constant(net_data[layer+'_b'])
        conv_out = conv2d(current, W_conv, stride=1, padding='SAME')

        current = tf.nn.bias_add(conv_out, b_conv)

    elif layer_type == 'pool':
        current = max_pool(current, k_size=2, stride=2, padding="SAME")

    elif layer_type == 'relu':
        current = tf.nn.relu(current)

    model[layer] = current

return model
```

You can see the whole loading model process and the auxiliary functions `conv2d()`, `get_type()`, `max_pool()` [here](https://github.com/gcucurull/neural-art-transfer/blob/master/models/vgg.py), but the most important part is the one I just explained. This file defines the function `get_model()` which will be used later to build the computation graph of the CNN that we want to use.

## Content and Style representations
Once we have a CNN that will process the images it is time to get the content and style representations of the source images, which will be used to generate the resulting image.

First we have to load the images, I have used `scipy.misc.imread` function for that.

```python
import scipy

content = scipy.misc.imread('input/1-content.jpg').astype(np.float)
style = scipy.misc.imread('styles/1-style.jpg').astype(np.float)
```

And now we just have to run these images trough our model to get their content and style representations respectively:

```python
# compute layer activations for content
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    content_pre = np.array([network_model.preprocess(content)])

    image = tf.placeholder('float', shape=content_pre.shape)
    model = network_model.get_model(image)
    content_out = sess.run(model[C_LAYER], feed_dict = {image:content_pre})
```

This piece of code is creating a tensorflow graph which will be run on cpu, and pre-processes the content image according to the model that we are using. In this case it just converts the RGB image to BGR and subtracts the mean pixel value of each channel. Then a placeholder with the same size as the image is created, and the model is run.
Notice that it is run as `model[C_LAYER]`. `C_LAYER` is the name of the layer chosen to represent content, `'conv4_2'` in my case, and `model` is a dictionary that holds the computation graph for each layer. In `model['conv4_2']` there is stored the graph of operations needed to produce the output of the layer `conv4_2`. At the end a Tensor containing the activations of the given layer for the content image is stored in `content_out`.

The code for loading the style representation of the source style image is very similar, it does almost the same but instead of getting the activations of one layer it gets the activations of several layers. This is done because `sess.run()` accepts a single operation or dictionary of operations, being each operation the activations of a given layer for the style image.

```python
# compute layer activations for style
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    style_pre = np.array([network_model.preprocess(style)])
    image = tf.placeholder('float', shape=style_pre.shape)
    model = network_model.get_model(image)
    style_out = sess.run({s_l:model[s_l] for s_l in S_LAYERS}, feed_dict = {image:style_pre})
```

## Generating the new image

Now we have the content and style representations of our source images stored in `content_out` and `style_out` respectively, it's time to generate the new image combining both content and style.

The new image will be the result of optimizing the loss function described before by changing the pixel values of a random noise image.  First let's see how the loss is defined in **Tensorflow**.

### Content Loss

```python
def content_loss(cont_out, target_out, layer, content_weight):
    '''
        # content loss is just the mean square error between the outputs of a given layer
        # in the content image and the target image
    '''
    cont_loss = tf.reduce_sum(tf.square(tf.sub(target_out[layer], cont_out)))

    # multiply the loss by its weight
    cont_loss = tf.mul(cont_loss, content_weight, name="cont_loss")

    return cont_loss
```

The code itself is quite self explanatory, it computes the mean square distance between the content representations of `cont_out` and `target_out[layer]`. See how in the last line before return the **content loss** is weighted, to allow us control the relevance of the content in the resulting image.

### Style Loss

```python
def style_loss(style_out, target_out, layers, style_weight_layer):

    def style_layer_loss(style_out, target_out, layer):
        '''
            returns the style loss for a given layer between
            the style image and the target image
        '''
        def gram_matrix(activation):
            flat = tf.reshape(activation, [-1, get_shape(activation)[3]]) # shape[3] is the number of feature maps
            res = tf.matmul(flat, flat, transpose_a=True)
            return res

        N = get_shape(target_out[layer])[3] # number of feature maps
        M = get_shape(target_out[layer])[1] * get_shape(target_out[layer])[2] # dimension of each feature map

        # compute the gram matrices of the activations of the given layer
        style_gram = gram_matrix(style_out[layer])
        target_gram = gram_matrix(target_out[layer])

        st_loss = tf.mul(tf.reduce_sum(tf.square(tf.sub(target_gram, style_gram))), 1./((N**2) * (M**2)))

        # multiply the loss by it's weight
        st_loss = tf.mul(st_loss, style_weight_layer, name='style_loss')

        return st_loss

    losses = []
    for s_l in layers:
        loss = style_layer_loss(style_out, target_out, s_l)
        losses.append(loss)

    return losses
```
The style loss implementation is very easy too. For every layer passed as argument in `layers` the function `style_layer_loss()` is called and the loss for each layer is stored in a list, which is the function return value. For each layer, the gram matrices of the style source and target image are computed, and then the loss for that layer is calculated as stated in the formula with the operations defined in the line:

```python
st_loss = tf.mul(tf.reduce_sum(tf.square(tf.sub(target_gram, style_gram))), 1./((N**2) * (M**2)))
```

### Tensorflow implementation

```python
# create image merging content and style
g = tf.Graph()
with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
    # init randomly
    target = tf.random_normal((1,)+content.shape)
    target_pre_var = tf.Variable(target)

    # build model with empty layer activations for generated target image
    model = network_model.get_model(target_pre_var)

    # compute loss
    cont_cost = losses.content_loss(content_out, model, C_LAYER, content_weight)
    style_cost = losses.style_loss(style_out, model, S_LAYERS, style_weight_layer)

    total_loss = cont_cost + tf.add_n(style_cost)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    sess.run(tf.initialize_all_variables())
    min_loss = float("inf")
    best = None
    for i in range(options.iter):
        train_step.run()
        print('Iteration %d/%d' % (i + 1, options.iter))

        if (i%5 == 0):
            loss = total_loss.eval()
            print('    total loss: %g' % total_loss.eval())
            if(loss < min_loss):
                min_loss = loss
                best = target_pre_var.eval()

    print('  content loss: %g' % cont_cost.eval())
    print('    style loss: %g' % tf.add_n(style_cost).eval())
    print('    total loss: %g' % total_loss.eval())

    final = best
    final = final.squeeze()
    final = network_model.postprocess(final)

    final = np.clip(final, 0, 255).astype(np.uint8)

    scipy.misc.imsave(out, final)
```

Now that we have implemented the loss functions, we just have to set up the training procedure to generate the image that merges the content and style of two source images. We start by creating the variable that we will optimize, i.e the generated image. To do so, we create a `tf.Variable()` initialized with random noise and the shape of the content image.
The line `model = network_model.get_model(target_pre_var)` creates the operations graph which builds the VGG network we will use to generate the content and style representations.
Then we instantiate the two losses we will be minimize, the content loss and the style loss, in the variables `cont_cost` and `style_cost`. Both losses are minimized at the same time, so we unify them in one node by adding them `total_loss = cont_cost + tf.add_n(style_cost)`. Remember that the function `style_loss` returns a list with the style loss of each layer specified in `S_LAYERS`. Tensorflow allows us to add all these losses with just the method `tf.add_n()`, which receives a list of tensors (of the same shape) and [produces one tensor containing the sum](http://stackoverflow.com/a/34520066/1738214).

The optimizer that I use is [Adam](http://sebastianruder.com/optimizing-gradient-descent/index.html#adam). It dynamically changes the learning rate for each parameter depending on previous gradients. I chose it because it generally performs better than SGD without tunning hyper-parameters like momentum, learning rate and learning rate policy.

Once we have defined the optimization operation in `train_step` we have all we need to change the initial random image until it succesfully merges the content and style of two images. We just have to call the operation `train_step.run()` inside a `for` loop, which in my code is done in the GPU but can be switched to the CPU by changing `g.device('/gpu:0')` to `g.device('/cpu:0')`.

## Results
You can view all the code in my github repo [neural-art-transfer](https://github.com/gcucurull/neural-art-transfer), which produces the following results:

#### Input image:
![input image](https://github.com/gcucurull/neural-art-transfer/raw/master/input/1-content.jpg)

#### Style image:
![style image](https://github.com/gcucurull/neural-art-transfer/raw/master/styles/1-style.jpg)

#### Result image:
![out image](https://github.com/gcucurull/neural-art-transfer/raw/master/output/1-output-new.jpg)

That's all, I hope you find it useful, I will gladly answer any questions or discuss anything about the code, you can contact me through the Github repository or send me an email at `gcucurull at gmail dot com`.
