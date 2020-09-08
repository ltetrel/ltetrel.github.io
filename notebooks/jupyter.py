# %% [markdown]
'''
# Jupyter notebooks
'''
# %% [markdown]
'''
Jupyter Notebook <cite> kluyver2016jupyter </cite> (with the `ipynb` extension) are documents that can contain code (including Python) and rich text elements (links, latex equations, figures).
They are powerfull and highly customizable for sharing interactive documents (through python widgets), and also easy to use (through your browser).

That is why I use it for my blog. 
'''
# %% [markdown]
'''
## tl;dr
1. Jupyter notebooks has lot of benefits for data science works (interactive, customizable, easy of use).
2. Use magic commands for system specific or other language.
'''
#%% [markdown]
'''
## An example ?
'''
#%% [markdown]
'''
Yeah sure !

You will see below an example of a python code inside a jupyter notebook.
We will extract the histogram of randomly sampled data from a gaussian distribution, and plot it using matplotlib.
'''
# %% [markdown]
'''
First, let's import the modules:
'''
# %%
import numpy as np
import matplotlib.pyplot as plt
# %% [markdown]
'''
Then, we initialize some values:
'''
# %%
# Fixing random state for reproducibility
np.random.seed(1)

mu = 500
sigma = 10
x = mu + sigma * np.random.randn(1000)
# %% [markdown]
'''
Now, we calculate and plot the histogram
'''
# %%
# the histogram of the data
plt.figure()
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.8)
plt.xlabel('x')
plt.ylabel('p')
plt.title('Histogram')
plt.grid(True)
plt.show()
# %% [markdown]
'''
That's it!

>**Note**  
>There is native support for matplotlib, so the figures are directly rendered in a cell.  
If you want to render them in a new window, you should add in the cell:
```python
%matplotlib qt
```
'''
#%% [markdown]
'''
## To go further
'''
#%% [markdown]
'''
By default, all code cells run the language from the kernel you selected. Here I used the python kernel, so all the cells run python code.  
If you want to run specific system command or code from another language, Jupyter notebooks supports [magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html).

For example to run a bash command you would do, 
'''
#%% [code]
%%bash
echo "Hello jupyter world!"
#%% [markdown]
'''
>**Note**  
>Specifically for shell commands, you can also use this simpler syntax:
```shell
!echo "Hello jupyter world!"
```
'''
# %% [markdown]
'''
## Tags
'''
# %% [markdown]
'''
Software-Development; Interactivity; Open-Science
'''