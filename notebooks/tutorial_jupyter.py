# %% [markdown]
# # **Notes on** : Jupyter notebooks
# %% [markdown]
# ## What are these ?
# 
# Notebook documents (\*.[ipynb](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html)) are documents that can contain code (like Python) and rich text elements (links, latex equations, figures).
# They are powerfull when you want to share interacting documents (you can interact with it), easilly usable (natively supported by your html compatible browser), and precious to share information in a fun way.
# 
# That is why I use it for my blog. 
# %% [markdown]
# ## An example ?
# 
# Yeah sure !
# 
# You will see below an example of a python code inside a jupyter notebook, using a simple matplot.
# %% [markdown]
# First, let's import the modules

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# Then, we initialize some values

# %%
# Fixing random state for reproducibility
np.random.seed(1)

mu = 500
sigma = 10
x = mu + sigma * np.random.randn(1000)

# %% [markdown]
# Now, we calculate and plot the histogram

# %%
# the histogram of the data
plt.figure()
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.8)
plt.xlabel('x')
plt.ylabel('p')
plt.title('Histogram')
plt.grid(True)
plt.show()

