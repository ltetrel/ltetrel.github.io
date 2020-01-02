# %% [markdown]
'''
# Fit a n-dimensionnal line with singular value decomposition
'''
# %% [markdown]
'''
## tl;dr
1. Organize your input data as an $d\times n$ matrix.
2. Normalize the data by subtracting the mean from the matrix.
3. Calculate the SVD to extract the eigen vectors of the data, the first one is the direction of the line.
4. Use the direction and average to write the parametric form of the line.
5. Calculate any point of the line by fixing the free parameter.
'''
# %% [markdown]
'''
##  1. Introduction
'''
# %% [markdown]
'''
Singular value decomposition (SVD) is a widely used method thast plays a huge role in machine learning data preprocessing.
It is mostly used to filter out noise from the data, reduce its dimensionnality/complexity and to have uncorrelated features.
SVD will extract the best meaningfull basis that describe much the data using linear algebra.

The idea is that any matrix $M$ with positive determinant can be factorized in the form :
\begin{equation}
M = U\Sigma V^* 
\end{equation}
Where $U$ and $V^*$ are rotation matrices and $\Sigma$ is a scale matrix.
Here $V^*$ wil give you the set of vector to project on the data onto the new dimension space.

Check in the following figure, the axis $y_1$ best explain the data because when you project the data into $y_1$, the variance is higher than for $y_2$.
![](imgs/line_svd/svd.pdf)

For the rest of this tutorial, we will use SVD to fit a 3-dimensionnal line based on some random 3D points (but this method can be extended to n-d!).
'''
# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# fixing numpy random state for reproducibility
np.random.seed(0)
# %% [markdown]
'''
##  2. Data generation
'''
# %% [markdown]
'''
First we want to generate n points with a random gaussian noise.
We organize the input data as an $d$×$n$ matrix, where $d$ is the number of dimensions (or features) and $n$ the number of samples.
'''
# %% [markdown]
n = 25
points = np.array( [5*np.arange(n),5+5*np.arange(n),5*np.ones(n)] ).T + 0.5-np.random.rand(n,3)
print(points)
# %% [markdown]
'''
##  2. Performing SVD
'''
# %% [markdown]
'''
Before performing SVD, it is necessary to center the data by subtracting the mean of the data.
Without centering, the first eigen vector would explain all the data.
This is because this method performs just scaling (as we saw earlier), but cannot take into account the bias of the data (i.e. the intercept in linear regression).
Sometimes it can also be usefull to normalize the input, because our data has not huge difference in each dimension we don't need to.
We will use [linear algebra package from numpy](https://docs.scipy.org/doc/numpy/reference/routines.linalg.html) for the SVD.
'''
# %%
# calculating the mean of the points
avg = np.mean(points, axis=0)

# subtracting the mean from all points
subtracted = points - avg

# performing SVD
_, _, V = np.linalg.svd(subtracted)
# %% [markdown]
'''
##  3. Finding the line
'''
# %% [markdown]
'''
To estimate the equation of a line, we need its direction (a vector) and one point that goes trough that line.
Previously, we performed SVD and extracted $V^*$ matrix that describe the eigen vectors. The first eigen vector is the one that best describe the data,
which in our case is the line that best fits all the points!

One example of a point that can go through this line is the average of the sample that we calculated previously. Then, any point of the line can be given by:
\begin{equation}
p(t) = p_0 + dt
\end{equation}
Where $t$ is the free parameter that is allowed to be any real number.
'''
# %%
# find the direction vector (which is the right singular vector corresponding to the largest singular value)
direction = V[0, :]

# A line is defined by the average and its direction
p0 = avg
d = direction
print(d)
# %% [markdown]
'''
We can calculate the angle $\alpha$ between two lines with direction $d_0$ and $d_1$ using:
\begin{equation}
\alpha = \arccos\Big(\frac{d_a.d_b}{\|d_a\|.\|d_b\|}\Big)
\end{equation}

For example, this is the angle between our line and the normal axis $(0, 0, 1)$.
'''
# %%
d0 = np.array([0, 0, 1])
angle = np.arccos(np.dot(d0,d)/(np.linalg.norm(d0) * np.linalg.norm(d)))
print(angle*180/np.pi)
# %% [markdown]
'''
##  4. Plotting the line
'''
# %% [markdown]
'''
Using the parametric form of the line, we can extract two different points by fixing the free parameter (make sure to choose a big one).
'''
# %%
pa = p0 + (-100)*d
pb = p0 + 100*d

# %% [markdown]
'''
To plot the 3D line, we will use [plotly](https://plot.ly/python/) that have really good html embeddings and smooth 3D rendering.
'''
# %%
trace1 = go.Scatter3d(
    x=[pa[0],pb[0]],
    y=[pa[1],pb[1]],
    z=[pa[2],pb[2]],
    mode='lines',
    name='3D fitted line',
    line=go.Line(color='rgb(255,0,0)', width=10),
    hoverinfo='none')

labels = []
for i in range(n): labels += [str(i)]  
trace2 = go.Scatter3d(
    x=points[:,0],
    y=points[:,1],
    z=points[:,2],
    mode='markers',
    name='Points',
    marker=go.Marker(
        symbol='cross',
        opacity=1,
        color='rgb(0,200,127)'),
    text=labels,
    hoverinfo='text')

layout = go.Layout(
            title="3D line fitting",
            scene=go.Scene(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="z",
                    camera=dict(
                           up=dict(x=0, y=0, z=1),
                           center=dict(x=0, y=0, z=0),
                           eye=dict(x=0, y=2.5, z=0))))     


fig=go.Figure(data=go.Data([trace1, trace2]), layout=layout)
fig.show(renderer="notebook_connected", config={'showLink': False})
# %% [markdown]
'''
## To go further
'''
# %% [markdown]
'''
If you want improve your understanding of SVD and its relation with PCA, check this [nice paper](https://arxiv.org/pdf/1404.1100.pdf) on the web.
On the importance of data normalization, check [this thread](https://stats.stackexchange.com/questions/22329/how-does-centering-the-data-get-rid-of-the-intercept-in-regression-and-pca).
'''