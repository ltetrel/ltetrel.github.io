# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:38:02 2018

@author: ltetrel
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2
import time

# Hand on : Construct your own deep deedforward network

# You all heard about deep neural networks, or use it. In this post, I will try to 
# explain to the fundamental behind neural nets with some simple examples.
# Thanks to http://peterroelants.github.io/posts/neural_network_implementation_intermezzo01/
# who helped me to have better understanding behing on the subject.

# Basic example

#Our goal will be to predict two different classes $C = {c_a, c_b}$, given some data.
# From this data, we know that the features $x_0, x_1$ can indeed be used to discriminate the samples.
# So the input of the algorithm will be a $Nx2$ vector $X = [x_0, x_1]$
# In 2D computer vision, one feature usually correspond to one pixel in the image.

np.random.seed(0)

# Distribution of the classes
n = 30
c_a_mean = [-1.5,0]
c_b_mean = [1.5,0]
std_dev = 1.2

# Generate samples from classes
X_c_a = np.random.randn(n, 2) * std_dev + c_a_mean + np.random.rand(n,2)
X_c_b = np.random.randn(n, 2) * std_dev + c_b_mean + np.random.rand(n,2) 

# Merge samples in set of input variables x, and corresponding set of output variables t
X = np.vstack((X_c_a, X_c_b))
C = np.vstack((np.zeros((n,1)), np.ones((n,1))))

# Plot both classes on the x1, x2 plane
plt.plot(X_c_a[:,0], X_c_a[:,1], 'ro', label='class $c_a$')
plt.plot(X_c_b[:,0], X_c_b[:,1], 'bo', label='class $c_b$')
plt.grid()
plt.legend()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Class a vs. b in the space X')
plt.show()

# !!!! Add random permutations to data

# To have non linear discriminant, we need to incorporate a kernel
# First we have the linear input x.w, we apply some function f to it
# f = g(x.w)
# Some function are Rectified linear unit (ReLU), 
# Using a activation function, we convert this data to a probability.
# Some output can be linear, sigmoid (for two class), softmax (for many classes)


# Now that we have our data, we can begin to construct the classifier.
# Neural nets uses the assumption of logistic function. 
# Sigmoid functions are best fitted for gradient descent optimization because they
# are more smooth compare to other activation functions like ReLU (rectified linear function).
# 
