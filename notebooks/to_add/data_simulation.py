#%%
'''
# Open science with closed data
'''

#%%
'''
# Introduction

Open science has a big trend nowadays, especially thanks to the huge amounts of free softwares available on the web.
It emphasis lot of important features in software development such as: reproducibility (Docker, singularity), collaborative work (github), communication (twitter) and easy development (python inside jupyter notebooks).
Recently, I have been quite active in the open neuroscience community, among other projects I contributed to the development for [neurolibre](https://github.com/neurolibre) within the canadian open neuroscience platform.
But we faced an issue when sharing some notebooks : how to manage and share your code that contain closed data ?
'''

#%%
'''
# TL;DR

1. Download the closed data locally
2. Fit a non/parametric model to the data (kernel density estimation for ex.)
3. Generate random samples from it

'''

#%%
'''
# 1. Data in medical imaging

In medical imaging, data are really expensive, follows strict security measures and take huge amounts of time to be acquired.
After the experiment has been designed, it has to be accepted by an independent ethical committee. The participants are convoqued and you have to pray that everything goes smoothly inside the scanner.
Moreover, they are special research certified scanner which allows for better data quality, robustness, security and are mandatory if you want to do research (this is specially the case in neuroscience).
This is why the data is usually really hard to get and have verry strict data user agreement, check for example the [data user agreement](http://www.ppmi-info.org/documents/ppmi-data-use-agreement.pdf) for the previous post.
Hopefully, data for medical imaging follows the trend of open science with versioning (datalad) and open initiatives that are coming up out there like [open-neuro](https://openneuro.org/) or [conp](https://github.com/CONP-PCNO/conp-dataset).

'''

#%%
'''
# 2. Example with IRIS dataset

Let's say you have a garden, you want to indentify flowers based on their petals and sepals with a camera.
You managed to create a super new model that classify the iris flowers but you had to used the iris flower dataset <cite data-cite=\"fisher1936use\"></cite> that we will suppose is non-shareable (this is not the case in real). 
Because you are so proud of it, you want to publish it to the world through github (data is small) but cannot share the dataset you used outside !


![](/imgs/open_data/flower.svg)
'''

#%%
'''
# 2.1 Imports

We will use scikit-learn to load the [iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). 
The pickle library will be use to save the data into an archive.
'''

#%%
# imports
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.datasets import load_iris

# fix random state
seed = 0
np.random.seed(seed)

#%%
'''
# 2.2 Data loading and analysis
'''

#%%
# Loading data from IRIS dataset
data = load_iris()

# metadata
nb_classes = len(np.unique(data.target))
n_samples = data.data.shape[0]
idx = np.random.permutation(n_samples)

plt.figure()
plt.hist2d(data.data[:, 0], data.data[:, 1], bins=30)
#plt.scatter(inputs_parkinson, pdf_parkinson, marker='+' ,color='darkred', linewidth=2, label='Parkinson')
#plt.scatter(inputs_healthy, pdf_healthy, marker='+' ,color='darkgreen', linewidth=2, label='Healthy')
# plt.legend()
plt.title('histogram for iris flowers')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

#%%
'''
# 2.3 Modeling

We can use either a parametric model or a non-parametric model. Because the data is quite
'''

# modeling
model_parkinson = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(right_putamen[labels])
model_healthy = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(right_putamen[~labels])
data = {"parkinson":model_parkinson, "healthy":model_healthy}
with open('./data/parkinson/parkinson_data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

inputs_parkinson = np.linspace(np.min(right_putamen[labels]), np.max(right_putamen[labels]), 1000).reshape(-1, 1)
inputs_healthy = np.linspace(np.min(right_putamen[~labels]), np.max(right_putamen[~labels]), 1000).reshape(-1, 1)
pdf_parkinson = model_parkinson.score_samples(inputs_parkinson)
pdf_healthy = model_healthy.score_samples(inputs_healthy)

plt.figure()
plt.scatter(inputs_parkinson, pdf_parkinson, marker='+' ,color='darkred', linewidth=2, label='Parkinson')
plt.scatter(inputs_healthy, pdf_healthy, marker='+' ,color='darkgreen', linewidth=2, label='Healthy')
plt.legend()
plt.title('pdf SBR')
plt.xlabel('SBR right putamen')
plt.ylabel('pdf')
plt.show()
