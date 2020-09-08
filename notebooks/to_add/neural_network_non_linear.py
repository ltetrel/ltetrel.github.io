# %% [markdown]
'''
# Introduction to neural networks (part 2)
'''
# %% [markdown]
'''
In the [previous post](*-neural_network.html), we saw an example of a linear neural network were the data was clearly linearly sperabale.
This time, we will try to classify non-linearly separable data.
Before showing how we could do that, it is important to introduce some important functions.
'''
# %% [markdown]
'''
## tl;dr
1. Importance of data analysis for modeling
2. Softmax is limited
3. Adding (hidden) layers with relu activation if data is non-linearly separable
4. Gradient checking
'''
# %% [markdown]
'''
## 1. Background
'''
# %% [markdown]
'''
### 1.1. Introducing non-linearity in a system
'''
# %% [markdown]
'''
Considering the basic McCulloch's model (with parameters $w$ and $b$), one would maybe have the idea that stacking multiple linear layers
could introduce non-linearity to the system.
But it is important to understand that stacking mutliple linear layers would not have any effect
on the linearity of the system. Indeed, imagine that you stack together $3$ linear layers, then the resulting output would be :

$$
\begin{equation}
        \begin{split}
            f_1\circ f_2\circ f_3(x) & = ((x\cdot w_3 + b_3)\cdot w_2 + b_2)\cdot w_1 + b_1 \\
                                     & = x\cdot w_3\cdot w_2\cdot w_1 + b_3\cdot w_2\cdot w_1 + b_2\cdot w_1 + b_1
        \end{split}
\end{equation}
$$

This is exactly the same a simple linear layer with $w = w_3\cdot w_2\cdot w_1$ and $b = b_3\cdot w_2\cdot w_1 + b_2\cdot w_1 + b_1$ !
Hence the importance of introducing activation function, that projects the data into a space where it will be linearly-separable.
'''
# %% [markdown]
'''
### 1.2. Limits of the softmax
'''
# %% [markdown]
'''
We already used the softmax popularized by Bridle et al. (Bridle, John S. "Probabilistic interpretation of feedforward classification
network outputs, with relationships to statistical pattern recognition." Neurocomputing. Springer, Berlin, Heidelberg, 1990. 227-236.).
If we would stack mutliple layers with softmax outputs, then our model would be non-linear.
The issue here is that the softmax (or sigmoid in $1D$) is closer to zero, when its inputs are less equal.
This can lead to the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) that can leads to zero gradients which would stop the training.
'''
#  %% [code]
### imports
class Neuron:
  def __init__(self, weights, bias):
    # first dim is the number of features
    # second dim is the number of neurons 
    self.weights = weights
    self.bias = bias

  def output(self, input):
    return input @ self.weights + self.bias

  def grad(self, input):
    D = np.kron(np.eye(self.weights.shape[1]), input)
    return [D, np.eye(self.weights.shape[1])]

def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))

# Derivative of the activation function
def dv_softmax(input):
    diag = softmax(input)*(1-softmax(input))
    xj, xi = np.meshgrid(softmax(input), softmax(input))
    jacob = (-1)*xi*xj
    np.fill_diagonal(jacob, diag)

    return jacob

def cost_function(input, t):
    return (-1)*np.sum(t * np.log(input))

# derivative of the cost function for gradient descent
def dv_cost_function(input, t):
    return (-1)*(t/input)

class LinearModel:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.n_neurons = weights.shape[1]

    def update_model(self, weights=[], bias=[]):
        if not isinstance(weights, list):
            self.weights = weights
        if not isinstance(bias, list):
            self.bias = bias

    def feed_forward(self, inputs):
        n_samples = inputs.shape[0]
        activations = np.zeros((n_samples, self.n_neurons))

        features = Neuron(self.weights, self.bias).output(inputs)
        for i in range(n_samples):
            activations[i, :] = softmax(features[i, :])
            
        return activations

    def back_propagation(self, inputs, t):
        n_samples = inputs.shape[0]
        g_w = np.zeros((self.weights.shape[0], self.weights.shape[1], n_samples))
        g_b = np.zeros((self.bias.shape[0], n_samples))

        feed_forwards = self.feed_forward(inputs)
        neuron = Neuron(self.weights, self.bias)
        for i in range(n_samples):
            grad_cost = dv_cost_function(feed_forwards[i, :], t[i, :])
            grad_activation = dv_softmax(neuron.output(inputs[i, :]))
            grad_neuron = neuron.grad(inputs[i, :])
            # here we resize the jacobian w.r.t. W so it can be easily substracted to W
            g_w[:, :, i] = np.reshape(grad_cost @ grad_activation @ grad_neuron[0], self.weights.shape, order='F')
            g_b[:, i] = grad_cost @ grad_activation @ grad_neuron[1]
        
        # sum of each sample's gradient
        return [np.sum(g_w, axis=-1), np.sum(g_b, axis=-1)]

import numpy as np
import copy
import matplotlib
from matplotlib.colors import colorConverter, ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from NN import LinearModel, cost_function, dv_cost_function, softmax, dv_softmax

# %% [code]
### Softmax and its derivative

# n_points=100
# x = np.linspace(-10, 10, n_points)
# xj, xi = np.meshgrid(x, x)
# y = np.zeros((100, 100, 2))

# # Softmax output
# for i in range(n_points):
#     for j in range(n_points):
#         y[i,j,:] = np.diag(dv_softmax(np.array([xj[i,j], xi[i,j]])))

# # Plot the activation for input
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xj, xi, y[:,:,0], cmap="plasma")
# ax.view_init(elev=30, azim=70)
# cbar = fig.colorbar(surf)
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$\partial y_1$')
# ax.set_title ('Diagonal of $\partial  y(\mathbf{x})_1$')
# cbar.ax.set_ylabel('$y_1$', fontsize=12)
# plt.show()

# %% [markdown]
'''
### 1.3. Rectified linear unit function
'''
#  %% [markdown]
'''
The vanishing gradient issue brought the idea of introducing relu activation function back in 2010 
(Nair, Vinod, and Geoffrey E. Hinton. "Rectified linear units improve restricted boltzmann machines." Proceedings of the 27th international conference on machine learning (ICML-10). 2010.)

$$
\begin{equation}
    y = \left\{
        \begin{array}{ll}
            x & \mbox{if } x > 0 \\
            0 & \mbox{else}
        \end{array}
    \right.
\end{equation}
$$

One clear advantage is that this activation is not computationnal intensive, fast and easy to use.
The gradient is always one for positive values, but it can still suffer from dying values if the input is negative (thus using [leaky ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLU
)). The fact that it is also non-derivable at zero can also raise some issues (check the [softplus](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus) instead).
'''
# %% [markdown]
'''
## 2. Hands on
'''
# %% [markdown]
'''
### 2.1. Input data
'''
#  %% [code]
### Data modelling

# Fix random state
np.random.seed(0)

# Distribution of the classes
n = 50
std_dev = 0.2

# Generate samples from classes
linspace = np.linspace(0, 2 * np.pi, n, endpoint=False)
X_a = np.random.randn(n, 2) * std_dev
X_b = np.vstack( (np.cos(linspace), np.sin(linspace))).T * 0.75 + np.random.randn(n, 2) * std_dev
X_c = np.vstack( (np.cos(linspace), np.sin(linspace))).T * 1.25 + np.random.randn(n, 2) * std_dev

# Create inputs X and targets C
X = np.vstack((X_a, X_b, X_c))
C = np.vstack( (np.zeros((n,1), dtype=int), np.ones((n,1), dtype=int), 2*np.ones((n,1), dtype=int)) )

# random permutation
idx = np.arange(X.shape[0])
idx = np.random.permutation(idx)
X = X[idx,]
C = C[idx,]

# one hot encoding
one_hot = np.zeros((len(C), np.max(C)+1))
one_hot[np.arange(len(C)), C.flatten().tolist()] = 1
C = one_hot
# %% [markdown]
'''
Let's take a look at the data.
'''
# %% [code]
### Plot both classes on the x1, x2 space

# plt.figure()
# plt.plot(X_a[:,0], X_a[:,1], 'ro', label='class $c_a$')
# plt.plot(X_b[:,0], X_b[:,1], 'b*', label='class $c_b$')
# plt.plot(X_c[:,0], X_c[:,1], 'g+', label='class $c_c$')
# plt.grid()
# plt.legend()
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.title('Class a, b and c in the space $X$')
# plt.show()

# %% [markdown]
'''
It is quite clear that the linear model from the previouis post #TODO:ref isn't adapted for this data.
As an exercice, you could try to plug this data into the previous post and see is you can train the linear model.
'''
# %% [markdown]
'''
### 2.2. Model design and parameters optimization
'''
# %% [markdown]
'''
The model that we will design will be close to the previouse one #TODO, the only difference
will be the addition of one layer that will have relu activation.
Such layer inserted in the middle of a neural network is called a hidden layer.
After, we compute the probabilities for the three classes through the softmax activation function.
Our final classifying rule will be that the highest probability gives us our class.

![](imgs/neural_network/nn_hidden.svg)
'''
# %% [markdown]
'''
We first need to implement the relu function and its derivative, which are quite easy to code.
It is worth also comparing it with the logistic activation.
'''
# %% [code]
# Activation function
def relu(input):
    return input * (input > 0) + 0.1 * input * (input < 0)

# Derivative of the activation function
def dv_relu(input):
    alpha = 0.3
    offset = alpha * np.ones_like(input) - alpha * np.eye(len(input))
    diag = np.ones_like(input) * (input > 0)
    return np.eye(input.shape[-1]) * diag + offset

def logistic(input):
    """Logistic function."""
    return 1. / (1. + np.exp(-input))

def dv_logistic(input):
    """Logistic function."""
    return np.eye(input.shape[-1]) * (logistic(input) * (1 - logistic(input)))
# %% [markdown]
'''
Given the weights $\mathbf{W_h}$ and bias $\mathbf{b_h}$ from the two hidden neurons, the relu activation of the hidden layer $h(\mathbf{x})$ over the input vector $\mathbf{x}=\{x_1, x_2\}$ is:

$$
\begin{equation}
    h(\mathbf{x}) =
    \begin{bmatrix}
        relu(x_1w_{h_{11}} + x_2w_{h_{21}} + b_{h_1}) \\
        relu(x_1w_{h_{12}} + x_2w_{h_{22}} + b_{h_2})
    \end{bmatrix}
\end{equation}
$$

The softmax activation of the output layer $y(h(\mathbf{x}))$ can be calculated given the weights $W_o$ and bias $\mathbf{b_o}$ from the three output neurons:

$$
\begin{equation}
	y(\mathbf{x}) = \frac{1}{e^{y_1w_{o_{11}} + y_2w_{o_{21}} + b_{o_1}} + e^{y_1w_{o_{12}} + y_2w_{o_{22}} + b_{o_2}} + e^{y_1w_{o_{13}} + y_2w_{o_{23}} + b_{o_3}}}
    \begin{bmatrix}
         e^{y_1w_{o_{11}} + y_2w_{o_{21}} + b_{o_1}}\\
         e^{y_1w_{o_{12}} + y_2w_{o_{22}} + b_{o_2}}\\
         e^{y_1w_{o_{13}} + y_2w_{o_{23}} + b_{o_3}}\\
    \end{bmatrix}
\end{equation}
$$
'''
# %% [markdown]
# Given the targets $\mathbf{t}$ and following the [chain rule](https://en.wikipedia.org/wiki/Chain_rule), we can decompose the derivative of the cost function $\xi(\mathbf{t}, y)$ w.r.t the output neuron parameters:
#
# $$
# \begin{equation}
#   \frac{\partial \xi(\mathbf{t}, y)}{\partial \mathbf{W_o}} = \frac{\partial \xi(\mathbf{t}, y)}{\partial y} \frac{\partial y}{\partial z_o} \frac{\partial z_o}{\partial \mathbf{W_o}},
# \end{equation}
# $$
#
# where $z_o$ is the output of the neurons for the output layer (just before the activation function).
#
# The derivative is quite different for $\mathbf{W_h}$ since we need to go "deeper" onto the model to compute the derivative.
# But it is still possible to reuse some previous results to avoid redundancy:
#
# $$
# \begin{equation}
#   \begin{split}
#       \frac{\partial \xi(\mathbf{t}, y)}{\partial \mathbf{W_h}} 
#           & = \frac{\partial \xi(\mathbf{t}, y)}{\partial h} \frac{\partial h}{\partial z_h} \frac{\partial z_h}{\partial \mathbf{W_h}} \\
#           & = \frac{\partial \xi(\mathbf{t}, y)}{\partial z_o} \frac{\partial z_o}{\partial h} \frac{\partial h}{\partial z_h} \frac{\partial z_h}{\partial \mathbf{W_h}},
#   \end{split}
# \end{equation}
# $$
#
# with 
# $$
# \begin{equation}
#   \frac{\partial z_o}{\partial h} = \mathbf{W_o}
# \end{equation}
# $$
#
# The same process stands for the bias parameters of the output $\mathbf{b_o}$ and hidden layer $\mathbf{b_h}$.
# %% [markdown]
'''
With all the previous code, we can now design the entire model:
'''
# %% [code]
### Model class definition
class NoneLinearModel:
    def __init__(self, weights, bias, hidden_activation="relu"):
        self.n_layers = len(weights)
        self.n_neurons = [weight.shape[-1] for weight in weights]
        self.hidden_activation = hidden_activation
        self.update_model(weights, bias)

    def update_model(self, weights=[], bias=[]):
        self.weights = weights
        self.bias = bias
        self.neurons = [Neuron(self.weights[i], self.bias[i]) for i in range(self.n_layers)]

    def feed_forward(self, inputs):
        n_samples = inputs.shape[0]
        activations = [inputs]
        activations += [np.zeros((n_samples, self.n_neurons[i])) for i in range(self.n_layers)]

        for i in range(self.n_layers):
            features = self.neurons[i].output(activations[i])
            if i < (self.n_layers - 1):
                activations[i+1] = relu(features) if self.hidden_activation == "relu" else logistic(features)
            else:
                for j in range(n_samples):
                    activations[i+1][j, :] = softmax(features[j, :])

        return activations[1::]

    def back_propagation(self, inputs, t):
        n_samples = inputs.shape[0]
        Jw = [np.zeros((weight.shape[0], weight.shape[1], n_samples)) for weight in self.weights]
        Jb = [np.zeros((weight.shape[1], n_samples)) for weight in self.weights]

        activations = self.feed_forward(inputs)
        for i in range(n_samples):
            # output layer
            grad_output_cost = dv_cost_function(activations[1][i, :], t[i, :])
            grad_output_activation = dv_softmax(self.neurons[1].output(activations[0][i, :]))
            grad_output_neuron = self.neurons[1].grad(activations[0][i, :])

            # hidden layer
            grad_hidden_cost_zo = grad_output_cost @ grad_output_activation
            grad_hidden_neuron_h = self.weights[1].T
            grad_hidden_activation = dv_relu(self.neurons[0].output(inputs[i, :])) if self.hidden_activation == "relu" \
                                        else dv_logistic(self.neurons[0].output(inputs[i, :]))
            grad_hidden_neuron = self.neurons[0].grad(inputs[i, :])

            # here we resize the jacobian w.r.t. W so it can be easily substracted to W
            Jw[0][:, :, i] = np.reshape(grad_hidden_cost_zo @ grad_hidden_neuron_h @ grad_hidden_activation @ grad_hidden_neuron[0], self.weights[0].shape, order='F')
            Jb[0][:, i] = grad_hidden_cost_zo @ grad_hidden_neuron_h @ grad_hidden_activation @ grad_hidden_neuron[1]
            Jw[1][:, :, i] = np.reshape(grad_output_cost @ grad_output_activation @ grad_output_neuron[0], self.weights[1].shape, order='F')
            Jb[1][:, i] = grad_output_cost @ grad_output_activation @ grad_output_neuron[1]

        # sum of each sample's gradient, for each layer
        Jw = [np.sum(Jw[l], axis=-1) for l in range(self.n_layers)]
        Jb = [np.sum(Jb[l], axis=-1)[np.newaxis, :] for l in range(self.n_layers)]

        return Jw, Jb
# %% [markdown]
'''
It is now time to train the model!
'''
# %% [code]
# learning phase

# hyper-parameters and model instanciation
lr = 0.01
n_iter = 1000
weights = [np.random.randn(2, 3), np.random.randn(3, 3)]
bias = [np.zeros((1, 3)), np.zeros((1, 3))]
cost_relu = np.array([])
cost_logits = np.array([])
relu_model = NoneLinearModel(weights=weights, bias=bias, hidden_activation="relu")
logits_model = NoneLinearModel(weights=weights, bias=bias, hidden_activation="logits")

for i in range(n_iter):
    # backpropagation
    Jw, Jb = relu_model.back_propagation(inputs=X, t=C)
    weights = [relu_model.weights[l] - lr * Jw[l] for l in range(relu_model.n_layers)]
    bias = [relu_model.bias[l] - lr * Jb[l] for l in range(relu_model.n_layers)]
    relu_model.update_model(weights=weights, bias=bias)
    # cost function
    probs = relu_model.feed_forward(X)[-1]
    cost_relu = np.append(cost_relu, cost_function(input=probs, t=C))

    # backpropagation
    Jw, Jb = logits_model.back_propagation(inputs=X, t=C)
    weights = [logits_model.weights[l] - lr * Jw[l] for l in range(logits_model.n_layers)]
    bias = [logits_model.bias[l] - lr * Jb[l] for l in range(logits_model.n_layers)]
    logits_model.update_model(weights=weights, bias=bias)
    # cost function
    probs = logits_model.feed_forward(X)[-1]
    cost_logits = np.append(cost_logits, cost_function(input=probs, t=C))

# plotting cost
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(cost_relu, 'b-+')
plt.grid()
plt.xlabel('iter')
plt.ylabel('cost')
plt.title('Relu model cost')
plt.subplot(1, 2, 2)
plt.plot(cost_logits, 'b-+')
plt.grid()
plt.xlabel('iter')
plt.ylabel('cost')
plt.title('Logits model cost')
plt.show()
# %% [markdown]
'''
### 2.3. Quantitative and qualitative analysis
'''
# %% [markdown]
'''
By comparing the decision functions for both relu and logits, we see that relu is faster to converge but its decision boundary is straight.
Because of the smoothness of the logits activation, the decision function is less prone to error.
'''
# %% [code]
### decision functions

# decision function for logits model
n_samples = 200
Xl = np.linspace(-5, 5, num=n_samples)
Xm, Ym = np.meshgrid(Xl, Xl)
Df = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        prob = logits_model.feed_forward( np.array([Xm[i, j], Ym[i, j]], ndmin=2) )[-1]
        Df[i, j] = np.argmax(prob, axis=1)

cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.3),
        colorConverter.to_rgba('b', alpha=0.3),
        colorConverter.to_rgba('g', alpha=0.3)])
plt.subplot(1, 2, 1)
plt.contourf(Xm, Ym, Df, cmap=cmap)

# ground truth for the inputs
plt.plot(X_a[:,0], X_a[:,1], 'ro', label='class $c_a$')
plt.plot(X_b[:,0], X_b[:,1], 'b*', label='class $c_b$')
plt.plot(X_c[:,0], X_c[:,1], 'g+', label='class $c_c$')
plt.grid()
plt.legend()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Logits model')

# decision function for relu model
for i in range(n_samples):
    for j in range(n_samples):
        prob = relu_model.feed_forward( np.array([Xm[i, j], Ym[i, j]], ndmin=2) )[-1]
        Df[i, j] = np.argmax(prob, axis=1)

cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.3),
        colorConverter.to_rgba('b', alpha=0.3),
        colorConverter.to_rgba('g', alpha=0.3)])
plt.subplot(1, 2, 2)
plt.contourf(Xm, Ym, Df, cmap=cmap)

# ground truth for the inputs
plt.plot(X_a[:,0], X_a[:,1], 'ro', label='class $c_a$')
plt.plot(X_b[:,0], X_b[:,1], 'b*', label='class $c_b$')
plt.plot(X_c[:,0], X_c[:,1], 'g+', label='class $c_c$')
plt.grid()
plt.legend()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Relu model')
plt.show()
# %% [markdown]
'''
## Conclusion
'''
# %% [markdown]
'''
You now have a deeper understanding behind the neural network mathematics!
Here I explained how to play with what we call a fully-connected network.
In the real world, dense layers are not really usable because of the huge increase in the number of parameters.
This is why everyone uses now convolutionnal neural network, which helps decreasing the number of parameters hence improving the learning!
Check [these nice animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) if you want to understand how convolutions are used in neural networks.
'''
# %% [markdown]
'''
## To go further
'''
# %% [markdown]
'''
Writing the gradients yourself is prone to errors, that is why it is interresting to compute the numerical gradient and compare it with your own gradient.
Check [peter roelants's blog](https://peterroelants.github.io/posts/neural-network-implementation-part04/#Gradient-checking) for more details!
Some also propose to add momentum during the learning phase, to avoid local-minima if the cost function is not convex. 
For this type of data I did not found it usefull, but it definitively worth [checking it](https://peterroelants.github.io/posts/neural-network-implementation-part04/#Backpropagation-updates-with-momentum)!
'''
# %% [markdown]
'''
## Acknowledgements
'''
# %% [markdown]
'''
Thanks to peter roelants who owns a nice blog on [machine learning](https://peterroelants.github.io/posts/neural-network-implementation-part01/).
It helped me to have deeper understanding behind the neural network mathematics. Some code were also inspired from his work.
'''
# %% [markdown]
'''
## Tags
'''
# %% [markdown]
'''
Artificial-Intelligence; Deep-Learning
'''