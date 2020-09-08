# %% [markdown]
'''
# What is the likelihood ?
'''
# %% [markdown]
'''
When refering to the likelihood of an event in the everyday language, we are often talking about the probability that this event occurs.
From a mathematical point of view, the likelihood of an event does not have the same meaning!

In this post I will try to explain you what do we refer by "likelihood", by following a simple example of a coin toss.
'''
# %% [markdown]
'''
## tl;dr
1. Likelihood is a function that represents the fitness of the data on a model.
2. The (negative) log-likelihood is usually refered as the "cost function".
3. Taking the derivative of the log-likelihood gives you a way to find optimal parameters of your model given the data.
'''
# %% [markdown]
'''
## 1. Theory
'''
# %% [markdown]
'''
Likelihood is a function that inform you how well the data $\mathbf{x} = [x_1 \ldots x_n]$ fits
to a model with parameters $\Theta = [\theta_1 \ldots \theta_n]$.  
Formalizing it was not an easy process, it took almost a decade for Fisher et al. between 1912 and 1922 <cite> aldrich1997ra </cite>!

The likelihood $L(\Theta)$ is defined by the product of each sample's conditionnal probability:
\begin{equation}
L(\Theta) = P(X=\mathbf{x}\mid\Theta) = P(x_1\mid\Theta)P(x_2\mid\Theta) \cdots P(x_n\mid\Theta)
        = \prod^n_{i=1} P(x_i\mid\Theta)
\end{equation}
Where $X$ is a random variable, $\mathbf{x}$ is the data.
'''
# %% [markdown]
'''
Because it is computationnaly easier to work with a sum instead a product, we compute the log-likelihood:

\begin{equation}
\log L(\Theta) = \log P(x_1\mid\Theta) + \log P(x_2\mid\Theta) + \cdots + \log P(x_n\mid\Theta)
        = \sum^n_{i=1} \log P(x_i\mid\Theta)
\end{equation} 

The basic idea is to find for which parameters $\Theta$ this function is maximized.

>**Note**
>It is also possible to minimize the negative log-likelihood, then we call it the "cost function".
'''
# %% [markdown]
'''
## 2. One little example
'''
# %% [markdown]
'''
Imagine you are walking the street and a street hustler is asking you to flip a coin for 5\$.
If the result is tail he will give you 11$, otherwise you lose your money.
You think it's a good trade, so you try 10 times:
'''
# %%
## imports
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# %%
# generating data
np.random.seed(3)
coin = np.random.binomial(1,0.6,10)
print(coin)
# %% [markdown]
'''
What a bad luck, you just lost 17\$ !

So depending on this experience, what is the probability $p$ of having a head ? You should answer something like : 
\begin{equation} \label{p_estimate}
p = \frac{n_{head}}{n}
\end{equation}
Congratulation, you just used a maximum likelihood estimate ! Maybe you can conclude that the coin is biased because $p = 0.7\neq 0.5$ ?
Of course you need more tries than that to conclude something, but now let's formalize how you calculated it.
'''
# %% [markdown]
'''
## 3. Theorical maximum likelihood estimation
'''
# %% [markdown]
'''
We suppose that we know the distribution from where our data comes from. For a coin flipping it can be a bernoulli distribution:
\begin{equation}
P(X = x) = p^x(1-p)^{1-x}
\end{equation}

Where $p$ is the parameter of the model (probability to have a head),
and $\mathbf{x}$ is the data (head or tail). We want to find the best estimation 
$\hat{p}$ for the probability of head $p = P(X=1)$ using maximum-likelihood estimation.

If we write the likelihood function it would be :

\begin{equation}    
L(p) = \prod^n_i P(x_i\mid p)
      = p^{x_1}(1-p)^{(1-x_1)}p^{x_2}(1-p)^{(1-x_2)} \cdots p^{x_n}(1-p)^{(1-x_n)}
\end{equation}

The log likelihood is:

\begin{equation}    
\log L(p) = \sum x_i \log p + (n - \sum x_i) \log (1-p)     
\end{equation} 

The maximum is given when the derivative is equal to 0 :
    
\begin{equation}
\frac{dL}{dp} = \frac{\sum x_i}{p} - \frac{n-\sum x_i}{1-p} = 0
\end{equation}

We can now evaluate the best estimation $\hat{p}$ of $p$ :
\begin{equation}
\hat{p} = \frac{\sum x_i}{n}
\end{equation}
Which is the formula \ref{p_estimate} that we used earlier!
'''
# %% [markdown]
'''
## 4. Numerical maximum likelihood estimation
'''
# %% [markdown]
'''
Another way to estimate $\hat{p}$ is to compute it numerically, using gradient descent.
This is the basic idea of machine learning : the input data can be really complicated (thousands of pixels),
and we make assumptions on the distribution (successive non-linear perceptron models, gaussian processes etc..).

Let's emulate 1000 coin flipping from a bernoulli distribution with parameter
$p = 0.7$, and see if we can estimate it correctly at the end.
'''
# %%
# input data
p = 0.7
n = 1000
x = np.random.binomial(1,p,n)
# %% [markdown]
'''
We hypothetize that our sample comes form a bernoulli distrubution :
'''
# %%
def bernoulli(p, x):
    return (p**x)*(1-p)**(1-x)
# %% [markdown]
'''
Definition of the likelihood (fitness of the data):
'''
# %%
def likelihood(p, x):
    return (p**(sum(x)))*((1-p)**(len(x) - sum(x)))

def log_likelihood(p, x):
    return np.sum(x)*np.log(p) + (len(x) - np.sum(x))*np.log(1-p)
# %% [markdown]
'''
Now we can do the gradient descent, used to update the parameter :

\begin{equation}
\Theta(t+1) = \Theta(t) - \mu\nabla\Theta(t+1) 
\end{equation}

$\mu$ is the learning rate, which defined how fast you will update your params.
    
$\nabla\Theta(t+1)$ is given by taking the gradient over all the parameters (which is in our case just the derivative over $p$).
'''
# %%
def derivative_log_likelihood(p, x):
    sum_xi = np.sum(x)
    return (-1)*(sum_xi/p - (len(x) - sum_xi)/(1-p))
# %% [markdown]
'''
We also need to fix the hyper-parameters used in gradient-descent.
These involve for example the learning rate $\mu$, the number of iterations max $i_{max}$ the stopping condition $\epsilon$ and the initial guess for the parameter $\Theta_{init}$.

I found these a posteriori by testing some values, and keeping the ones which gives the best results (done using a training set).
'''
# %%
# optimization
p = 0.1
learnRate = 1e-6
iterMax = 10000
iters = 0
lkls = []

for _ in range(iterMax):
    lkls += [log_likelihood(p,x)]
    grad = derivative_log_likelihood(p, x)
    p = p - learnRate*grad
    iters = iters + 1
    if( abs(grad)<1e-6 ):
        break

print("Found probability %1.6f in %d with log-likelihood %1.3f" %(p,iters,lkls[-1]))  
# %% [markdown]
'''
We can plot the log-likelihood to see how it variate on the parameter space (one dimension in our case).
'''
# %%
## plot likelihood
p = np.linspace(0.01,0.99,100)
lkls = log_likelihood(p,x)
plt.figure()
plt.plot(p,lkls)
plt.xlabel("p")
plt.ylabel("log-likelihood")
plt.show()
# %% [markdown]
'''
We can also look at the normalized likelihood
'''
# %%
## plot normalized likelihood
plt.figure()
lkls = likelihood(p,x)
lkls = lkls / np.max(lkls)
plt.plot(p,lkls)
plt.xlabel("p")
plt.ylabel("normalized likelihood")
plt.show()
# %% [markdown]
'''
## 5. Conclusion
'''
# %% [markdown]
'''
Maximum likelihood is probably one of the most important fundamental, without it there is no machine learning!

We saw two ways to use it, theorically and numerically.
The latter is mostly used because it is often to complicated to do the calculations.
'''
# %% [markdown]
'''
## Tags
'''
# %% [markdown]
'''
Data-Science; Statistics; Optimization
'''