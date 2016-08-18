---
layout:     post
title:      My implementation of neural art style transfer in Tensorflow
date:       2016-08-18 22:21:58
summary:    "In order to practise a little bit with Tensorflow I have implemented the paper A Neural Algorithm of Artistic Style. I think it is an easy algorithm to code, and I encourage all the people who want to learn about Deep Learning or the ones who are trying to learn how to use a new fraemeowrk to implement it themselves."
categories: tensorflow style-transfer
---

Since it was released last November, I have read and learned a little bit about Tensorflow. The Udacity's [exercises in their github repo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity) are very good and the book by Jordi Torres [First Contact With Tensorflow](http://www.jorditorres.org/first-contact-with-tensorflow/) is an amazing introduction to the usage of Tensorflow. However, as it is said, practice makes perfect, so in order to really learn how to use Tensorflow one must get his hands dirty and code some real stuff.

I decided to implement the paper "A Neural Algorithm of Artistic Style" by Gatys *et. al* [(link)](http://arxiv.org/abs/1508.06576) for several reasons. First of all, the idea of merging the content of a picture with the style of another seemed really cool to me, and I was eager to learn how the algorithm worked.
