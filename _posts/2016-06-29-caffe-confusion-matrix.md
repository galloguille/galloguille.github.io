---
layout:     post
title:      Caffe python layer to print confusion matrix
date:       2016-06-29 12:31:19
summary:    To view the confusion matrix and the accuracy of a neural network while it is being trained, I coded this easy python layer to print the confusion matrix while training a model with Caffe. 
categories: caffe python deep-learning
---

While training a model with [Caffe deep learning framework](https://github.com/BVLC/caffe) it is very easy and useful to evaluate how the model performs on an independent test set. This is very important to test whether the model is overfitting or underfitting, and by adding an Accuracy Layer it lets us know how the model performs in a classification task.

But sometimes the accuracy metric is not enough, so I coded a very simple __python layer for Caffe__ that replaces the accuracy layer and prints a __confusion matrix__, to have a slightly deeper understanding of what our classification model is doing right (or wrong). You can find the [code in Github](https://github.com/gcucurull/caffe-conf-matrix).

The usage of this python layer is very easy. First of all Caffe has to be compiled to __support python layers__, check out [this post in @chrischoy blog](http://chrischoy.github.io/research/caffe-python-layer/) to learn how to compile Caffe with support for python layers and learn more about them.
Once Caffe is build with support for python layers, the usage of the layer is very simple, it just has to be used as an accuracy layer in the prototxt file like:
	
	layer {
	  type: 'Python'
	  name: 'py_accuracy'
	  top: 'py_accuracy'
	  bottom: 'ip2'
	  bottom: 'label'
	  python_param {
	    # the module name -- usually the filename -- that needs to be in $PYTHONPATH
	    module: 'python_confmat'
	    # the layer name -- the class name in the module
	    layer: 'PythonConfMat'
	    # this is the number of test iterations, it must be the same as defined in the solver.
	    param_str: '{"test_iter":100}'
	  }
	  include {
	    phase: TEST
	  }
	}

There is a working example in the `examples` folder of [the Github repo](https://github.com/gcucurull/caffe-conf-matrix), which must be copied in `caffe/examples` folder in order for the relative paths to work. The file `python_confmat.py` must be copied in `caffe/examples/mnist` to work for the example, but for your own usage you can place it anywhere as long as the path is included in your `$PYTHONPATH`.

The confusion matrix is printed to console and looks like this:

	Confusion Matrix                                                | Accuracy
	------------------------------------------------------------------------
	3438    166     191     16      45      9       136     0       | 85.93 % 
	191     3306    177     1       69      2       15      0       | 87.90 % 
	88      114     3205    34      431     46      80      3       | 80.10 % 
	30      12      98      3735    78      23      24      0       | 93.38 % 
	11      28      437     29      3196    65      45      11      | 83.62 % 
	3       0       64      7       38      3702    8       0       | 96.86 % 
	59      4       79      42      44      5       3234    1       | 93.25 % 
	2       0       29      3       113     9       6       2639    | 94.22 % 
	Number of test samples: 29676

Here you have a nice example on how to use a Python Layer for Caffe to create a confusion matrix during training, I hope it is useful and feel free to use anywhere you need it.
