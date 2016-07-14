---
layout:     post
title:      Reverse a Torch Tensor
date:       2016-07-12 16:48:00
summary:    Reversing a Tensor in Torch is not as easy as it is with Numpy, this is my one line code to reverse a Tensor in one dimension.
categories: torch deep-learning
---

I've recently started using Torch for Deep Learning, and besides the fact that Lua has 1-based array indexing and I used to forget that, I'm not having too much trouble.

However, when I needed to reverse a Torch Tensor in one dimension, I discovered that Torch slicing is not as expressive as the awesome Numpy slicing, so I had to figure out another way to reverse a Tensor.

In Numpy to reverse an array in a specific dimension I would do something like:
{% highlight python %}
x = x[:,::-1,:]
{% endhighlight %}

But in Torch I haven't figured out how to do so, and when I googled for a solution I didn't find many helpful links explaining a good way to reverse a Tensor, I found [this question on Torch's google group](https://groups.google.com/forum/#!topic/torch7/O1btOEDC0t8) for example, so I'm going to share what I've been using.

In Torch, for a case of a tensor with size (batch_size, dim, seq_len) which we want to reverse in the 3rd  dimension, the code is the following:

{% highlight lua %}
x = torch.Tensor(batch_size, dim, seq_len)
x = x:index(3 ,torch.linspace(seq_len,1,seq_len):long())
{% endhighlight %}

Easy, right? I hope it works for you too.
