# %% [markdown]
'''
# Gradient descent for geometric models : example with a sphere
'''
# %% [markdown]
'''
## tl;dr
1. Mathematical model definition for your data
2. Cost function
3. Optimisation (gradient descent with cost function derivative on all the parameters)
4. Hyper-parameters optimisation with training set (learning rate)
5. Model bias with testing set
'''
# %% [markdown]
'''
## 1. Introduction
'''
# %% [markdown]
'''
The process of fitting known geometric models to scattered data is well known and resolved since a long time. 
Indeed, Legendre was one of the first to use least-square to do such tasks <cite data-cite="legendre1805nouvelles">[1]</cite>,
where he wanted to fit an equation for the shape of the earth !

The idea is quite simple: knowing the model definition, it is possible to estimate its best parameters 
that fits well to the input noisy data.
'''
# %% [markdown]
'''
## 2. Optimisation
'''
# %% [markdown]
'''
Given a mathematical model with parameters $\Theta$, we can define its minimization function $\xi(\Theta)$ (the inverse likelihood).
This represents the fitness of the model given some data, so if $\xi(\Theta)$ is minimum, then the parameters are best suited to the data.

To optimize the parameters, one could check every possible parameters $\Theta_i$ and compute directly $\xi(\Theta_i)$.
Unfortunately, with our current compute power, this is time consuming.
Especially if we have more than one parameter, then the manifold would be to large to explore.
Imagine the difference between exploring a one dimension line vs exploring a 3D surface!

<img src="imgs/sphere_fitting/manifold.svg" alt="manifold" style="width: 200px;"/>

This is why it is important to have a dynamic strategy to find the local minimum, and this can be done using gradients.
We can calculate the gradient of the energy function among our parameters:
\begin{equation}
\frac{\delta \xi(\Theta)}{\delta \Theta} = \nabla\xi(\Theta)
\end{equation}

The gradient of the energy function can then be used to update the parameter every step, and converge to a local minima:
\begin{equation}
\Theta(t+1) = \Theta(t) - \mu\nabla\xi(\Theta(t)) 
\end{equation}

One could compute directly the optimal solution by equalizing the gradient of energy function with 0:
\begin{equation}
\frac{\delta \xi(\Theta)}{\delta \Theta} = 0
\end{equation}

When this method works well with little data and few parameters, it can be computationnaly impossible to compute the best solution [see this thread for more information](https://stats.stackexchange.com/questions/278755/why-use-gradient-descent-for-linear-regression-when-a-closed-form-math-solution). 
Morevover, it is not always easy to mathematically extract the optimal parameters.
'''
# %% [markdown]
'''
## 3. Example with sphere fitting
'''
# %% [markdown]
'''
### 3.1 Cost function
'''
# %% [markdown]
'''
To extract the cost function, we first need to define what is a sphere.
The equation of a 3D sphere with radius $r$ and center $(a, b, c)$ is known as:
\begin{equation}
r^2 = (x-a)^2 + (y-b)^2 + (z-c)^2
\end{equation}

Every point $(x,y,z)$ must satisfy this equation to be on the sphere, if not, the point is outside the sphere.
Using this condition, we can deduce the energy function:
\begin{equation}
\xi(\Theta) = \sum_i^n(L_i - r)^2
\end{equation}
with,
\begin{equation}
L_i = \sqrt{(x_i-a)^2 + (y_i-b)^2 + (z_i-c)^2}
\end{equation}
This function will verify if every known point $i$ fits well with the parameters $a, b, c$ and $r$.
Let's implement it in python !
'''
# %%
## imports
import numpy as np
import math
import plotly.graph_objects as go

# fixing numpy random state for reproducibility
np.random.seed(0)
# %%
def sph_loss(T, x):
    L = np.sqrt((x[:,0] - T[0])**2 + (x[:,1] - T[1])**2 + (x[:,2] - T[2])**2)
    return L
# %%
def cost_function(T, x):
    L = sph_loss(T, x)
    return np.sum( (L - T[3])**2 )
# %% [markdown]
'''
### 3.2 Gradient of the cost function
'''
# %% [markdown]
'''
Now that we have the cost funtion, we can compute its gradient for every parameters (a, b, c, r). 
\begin{equation}
\nabla\xi(\Theta) =
  \begin{bmatrix}
  \frac{\delta \xi(\Theta)}{\delta r} \\
    \frac{\delta \xi(\Theta)}{\delta a} \\
    \frac{\delta \xi(\Theta)}{\delta b} \\
    \frac{\delta \xi(\Theta)}{\delta c} 
  \end{bmatrix}
\end{equation}

Then, with $m$ as the number of 3D points:

\begin{equation}
\frac{\delta \xi(\Theta)}{\delta r} = -2\sum_{i=1}^m (L_i -r)
\end{equation}

\begin{equation}
\frac{\delta \xi(\Theta)}{\delta a} = 2\sum_{i=1}^{m}((x_i-a) + r\frac{\delta L_i}{\delta a}); \qquad  \frac{\delta L_i}{\delta a} = \frac{a-x_i}{L_i}
\end{equation}

\begin{equation}
\frac{\delta \xi(\Theta)}{\delta b} = 2\sum_{i=1}^{m}((y_i-b) + r\frac{\delta L_i}{\delta b}); \qquad  \frac{\delta L_i}{\delta b} = \frac{b-y_i}{L_i}
\end{equation}

\begin{equation}
\frac{\delta \xi(\Theta)}{\delta c} = 2\sum_{i=1}^{m}((z_i-c) + r\frac{\delta L_i}{\delta c}); \qquad  \frac{\delta L_i}{\delta c} = \frac{c-z_i}{L_i}
\end{equation}
'''
# %% [markdown]
# In python,

# %%
## cost function derivative
def derivative_cost_function(T, x):
    L = sph_loss(T, x)
    
    dr = (-1)*(-2)*np.sum( (L - T[3]) )
    
    dLa = (T[0] - x[:,0])/L
    da = 2*np.sum( (x[:,0] - T[0]) + T[3]*dLa )
    
    dLb = (T[1] - x[:,1])/L
    db = 2*np.sum( (x[:,1] - T[1]) + T[3]*dLb )
    
    dLc = (T[2] - x[:,2])/L
    dc = 2*np.sum( (x[:,2] - T[2]) + T[3]*dLc )
    
    return np.array([da, db, dc, dr])
# %% [markdown]
'''
### 3.3 Gradient descent
'''
# %% [markdown]
'''
Using the gradient of the cost function, it is now possible to optimize the best parameters with gradient descent.
'''
# %%
def grad_descent(data, param_init):
    T = param_init #initial guess
    lr = 5e-3
    it_max = 10000
    grad = [1e99, 1e99, 1e99]

    for it in range(it_max): 
        if( abs(np.sum(grad))<1e-6 ):
            continue
        # gradient descent
        grad = derivative_cost_function(T, data)
        T = T + lr*grad
        it_max = it
    print("Done in %d epochs with grad: [%1.4e, %1.4e, %1.4e, %1.4e]" %(it_max, grad[0], grad[1], grad[2], grad[3]))
    
    return T, grad

# %% [markdown]
'''
### 3.4 Training phase
'''
# %% [markdown]
'''
We will first generated data for a standard 3D sphere, we can use spherical coordinates to sample random points given the sphere parameters. To reduce CPU time, we will not use a high number of points.
'''
# %%
def gen_points_from_sph(model, n):
    res = int(np.ceil(math.sqrt(n)))
    theta = np.linspace(0,2*np.pi,res)
    phi = np.linspace(0,np.pi,res)
    
    x = model[3]*np.outer(np.cos(theta),np.sin(phi))
    y = model[3]*np.outer(np.sin(theta),np.sin(phi))
    z = model[3]*np.outer(np.ones(res),np.cos(phi))
    
    X = x + model[0]
    Y = y + model[1]
    Z = z + model[2]
    
    points = np.zeros((n, 3))
    points[:,0] = X.ravel()[0:n]
    points[:,1] = Y.ravel()[0:n]
    points[:,2] = Z.ravel()[0:n]

    return points, X, Y, Z
# %% [markdown]
'''
Let's generate 250 points with gaussian noise, from an unknown sphere centered at (1,2,4) and 10 radius.
Then, we will use 2/3 of the data and try to find these parameters. Remember that the hyper-parameters for the gradient descent are supposed to be optimized during training set.
'''
# %%
#Training phase
sph_model = [1, 2, 4, 10]
n = 150
sph_points = gen_points_from_sph(sph_model, n)[0] + np.random.randn(n,3)
sph_points_train = sph_points[0:int(n/3),:]

param_init = np.array([0, 0, 0, 10]) #initial guess

param = grad_descent(sph_points_train, param_init)[0]

# We use the formula for the radius
print(param)
# %% [markdown]
'''
The optimization returned a sphere centered at (1.21, 2.09, 3.94) with 9.94 radius.
'''
# %% [markdown]
'''
### 3.4 Testing phase
'''
# %% [markdown]
'''
Now that we have an estimation on the model, we can test it and see how weel this model fits to the data. We use the 1/3 remaining points to estimate the error of the model.

The error estimation can be done using the fitting function $\xi(\Theta)$.
'''
# %%
#Testing phase
sph_points_test = sph_points[int(n/3)::,:]
model_error = cost_function(sph_model, sph_points_test)/sph_points_test.shape[0]

print("Model has an error of %.2f per point"%model_error)
# %% [markdown]
'''
The error per point is quite high. But considering the noise that was introduced, we see that the algorithm performs quite well. 
'''
# %%
model_error = cost_function(param, sph_points_test)/sph_points_test.shape[0]

print("Noise of the data %.2f per point"%model_error)

# %% [markdown]
'''
This is using these error can we can optimized the hyper-parameters (optimization parameters), to find which one are best suited for the data.
'''
# %% [markdown]
'''
### 3.5 Qualitative result
'''
# %% [markdown]
'''
We will use [plotly](https://plot.ly/python/) to render the result.
'''
# %%
## Qualitative results with plotly
_,X, Y, Z = gen_points_from_sph(param, 5*n)
    
trace = go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            opacity = 0.5,
            colorscale=[[0, 'rgb(50,50,125)'], [1, 'rgb(50,50,125)']])

trace2=go.Scatter3d(
           x=sph_points[:,0].ravel(),
           y=sph_points[:,1].ravel(),
           z=sph_points[:,2].ravel(),
           mode='markers',
           name='points',
           marker=go.scatter3d.Marker(symbol='circle',
                            size=3,
                            color='rgb(0,0,255)',
                            opacity=1),)

layout = go.Layout(
            title="Sphere fitting",
            scene=go.layout.Scene(
                aspectmode = "data",
                xaxis_title="x(mm)",
                yaxis_title="y(mm)",
                zaxis_title="z(mm)",            
                camera=dict(center=dict(x=0.1, y=0.1, z=0))))

fig = go.Figure(data=[trace, trace2], layout=layout)
fig.show(renderer="iframe_connected", config={'showLink': False})
# %% [markdown]
'''
## To go further
'''
# %% [markdown]
'''
You can look at other examples for standard mathematical models [here](https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf).
'''