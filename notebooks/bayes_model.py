# %% [markdown]
'''
# Generative model to classify parkinson subject
'''
# %% [markdown]
'''
## tl;dr
1. Generated data from PPMI cohort initiative
2. Choice of model and parameter estimation on the training set (testing set to check accuracy).
4. Evaluation of the likelihood for all the classes on the validation set.
5. Posterior estimation and MAP for class prediction.
6. Classsifier accuracy study
'''
# %% [markdown]
'''
## 1. Introduction
'''
# %% [markdown]
'''
Generative model is part of machine learning, and is usually used when you have easy access to your data and you want to infer the distribution on it.
We will take a look at one generative model widely known : **Naive Bayes**.<br>
Thomas Bayes is a well known mathematician, with work focused in probability.
Bayes theory <cite data-cite="bayes1958essay"></cite> is central in machine learning and has numerous applications in automatic classification.
'''
# %% [markdown]
'''
Consider two event $a$ and $b$, the Bayesian rule define the probability $p$ that $a$ occurs, knowing the result of the event $b$ :
	
\begin{equation}
	p(a|b) = \frac{p(b|a) \times p(a)}{p(b)} 
	\end{equation}
	
It can be interpreted as is :
<br><br> _if we know a priori the probabilities of the events_ $a$, $b$, _and the result of the event_ $b$ (knowing result of $a$), _then we can deduct the result_ $a$ (knowing result of $b$).<br><br>
This rule will allow us to infer things about the income of events, with some aprioris.
The idea behing generative models is to learn the probability $p(b,a) = p(a) . p(b|a)$, which will later be used to classify the data $a$ with $p(a|b)$ through bayesian rule. 
'''
# %% [markdown]
'''
## 2. Classifying parkinson patient with naive bayes classifier
'''
# %% [markdown]
'''
### 2.1 Parkinson disease biomarkers
'''
# %% [markdown]
'''
Parkinson disorder is a neurodegenerative disease, characterized by the loss of cells creating the dopamine <cite data-cite="poewe2017parkinson"></cite>. 
Dopamine is an organic chemical ensuring the communication between the neurons in the brain. Early diagnosis of parkinson is crucial because to initiate neuroprotective therapies and enhancing their success rate.
Here we will use biomarkers using single-photon emission computed tomography (SPECT) from one region of the brain (striatal binding ratio from right putamen).
Then, we will try to classify healthy subject and parkinson, following the paper from prashanth et al. <cite data-cite="prashanth2014automatic"></cite>

![](imgs/parkinson/right_putamen.svg)
'''
# %% [markdown]
'''
Given the SBR of the righ putamen $\mathbf{x}$, and the outcome $\Omega$ for a patient to have parkinson $\omega_p$ or being healthy $\omega_h$,  $\omega_i = \{\omega_p, \omega_h\}$, the bayesian rule can be rewritten as :
\begin{equation}
	P(\Omega = \omega_i\mid \mathbf{x}) = \frac{p(\mathbf{x}\mid\Omega = \omega_i) \times p(\Omega = \omega_i)}{p(\mathbf{x})} 
	\end{equation}
    
$p(\omega_i)$ and $p(X)$ are the marginal probabilities, and $p(\mathbf{x}|\omega_i).p(\omega_i)$ is known as the likelihood (refer to post \url{}) which is used to create the generative model.
'''
# %% [markdown]
'''
So let's load and visualize the data, for the rest of the post we suppose that the sample $\mathbf{x}$ are independent and identically distributed.
To be in agreement with the data use agreement, "fake" data will be sampled from a model I generated using the trial see appendix A for more details.
'''
# %%
### imports
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_curve

import warnings
warnings.filterwarnings('ignore')
# %%
# Data loading and analysis
# fix random state
seed = 0
np.random.seed(seed)

# number of subjects
n = 10000
n_parkinson = 3000

# Generating parkinson data
with open('data/parkinson/parkinson_data.pickle', 'rb') as handle:
    models = pickle.load(handle)
parkinson_data = np.squeeze(models["parkinson"].sample(int(n_parkinson), random_state=seed))
healthy_data = np.squeeze(models["healthy"].sample(int(n - n_parkinson), random_state=seed))
data = np.concatenate([parkinson_data, healthy_data])
labels = np.ones(n)
labels[int(n_parkinson)::] = 0

# shuffling the data so it appears more real
rnd_idx = np.random.permutation(n)
data = data[rnd_idx]
labels = labels[rnd_idx]

# Probability density functions
plt.figure()
plt.hist(data[labels==1], density=True, bins=25, color='darkred', alpha=0.5, label='early parkinson')
plt.hist(data[labels==0], density=True, bins=25, color='darkgreen', alpha=0.5, label='healthy')
plt.legend()
plt.title('Data for early parkinson and healthy subjects')
plt.xlabel('SBR of right putamen')
plt.ylabel('pdf')
plt.show()
# %% [markdown]
'''
### 2.2 Data modelisation
'''
# %% [markdown]
'''
Given the data, you want to model the distribution of the two conditions.
In machine learning, we usually refer the probability as a function called pdf (probability density function).
Because we hypothetized that the data are coming from a gaussian, we can calculate its model parameters ($\mu$, $\sigma$) (usually you want to calculate the likelihood of your data to verify if it's truly gaussian, or use a test like Student's t-test).
To estimate the parameters, we will divide the dataset into ($1/3$) training set (that we will use to tune our model), and ($2/3$) validation set (used later to evaluate accuracy).
'''
# %%
# data modeling
# creation of training and dataset
train_data = data[0:np.int(n/3)]
train_label = labels[0:np.int(n/3)]
valid_data = data[np.int(n/3)+1::]
valid_label = labels[np.int(n/3)+1::]

# Estimators for gaussian mean and deviation
mean_parkinson = np.median(train_data[train_label==1])
mean_healthy = np.median(train_data[train_label==0])
std_parkinson = np.std(train_data[train_label==1])
std_healthy = np.std(train_data[train_label==0])

#pdf of data
pdf_parkinson = stats.norm(mean_parkinson, std_parkinson).pdf(train_data)
pdf_healthy = stats.norm(mean_healthy, std_healthy).pdf(train_data)

# Probability density functions
plt.figure()
plt.scatter(train_data, pdf_parkinson, marker='.' ,color='darkred', linewidth=2, label='early parkinson')
plt.scatter(train_data, pdf_healthy, marker='.', color='darkgreen', linewidth=2, label='healthy')
plt.legend()
plt.title('The two models for early parkinson and healthy')
plt.xlabel('SBR of right putamen')
plt.ylabel('pdf')
plt.show()

# %% [markdown]
'''
### 2.3 Prior estimations (with learning set)
'''
# %% [markdown]
'''
Now that we have our models $p(x_i | \omega_i)$, we can evaluate the likelihood of the validation data for our classes and make the predictions $p(x_i | \Omega)$.
So we just fix the data and evaluate all the possible probabilities for the two classes.
Usually it is not recommended to use the learning set to check the prediction accuracy but instead a testing set, this to avoid overfitting your model.
Here for simplicity of this post, we will just use the training set.

The apriori probabilities will just be the number of parkinson subjects $n_{\omega_0}=3000$ divided by the total number of samples $n$.
The marginal probability $p(X)$ is just the probability to have a data (for example the probability of a successfull scanning), we will assume that all subjects were successfully scanned.

Let's try with a SBR right putamen of 1.3.
'''
# %%
# prior estimation on training set
x = 1.3

#Bayes
p_parkinson = n_parkinson/n
p_healthy = (n-n_parkinson)/n
p_data = 1
lklh_parkinson = stats.norm(mean_parkinson, std_parkinson).pdf(x)*p_parkinson
lklh_healthy = stats.norm(mean_healthy, std_healthy).pdf(x)*p_healthy

#Plot Likelihoods
plt.figure()
plt.xlim([0, 1])
plt.stem([0.25, 0.75], [lklh_parkinson, lklh_healthy], linefmt='b-', markerfmt='bo', basefmt='', use_line_collection=True)
plt.xticks( [0.25, 0.75], ('early parkinson', 'healthy') )
plt.ylabel('Likelihood of class')
plt.title('Likelihood for x=1.3')
plt.show()

# %% [markdown]
'''
A common error is to assume that the likelihood is a pdf but it is not the case, the total area is not equal to 1.
You can see that for $\mathbf{x}=1.3$, the likelihoods are really close so you cannot really make any prediction there.
'''
# %% [markdown]
'''
### 2.4 Posterior evaluation (with validation set)
'''
# %% [markdown]
'''
Now that we calculated the likelihood evaluated for the data, we can evaluate the posterior probability (predict what condition the subject has), given the marginals and the likelihood
(sometimes, it can be simpler and faster to evaluate just the likelihoods to do predictions).
'''
# %%
# posterior estimation on validation set
#Bayes
p_parkinson = n_parkinson/n
p_healthy = (n-n_parkinson)/n
p_data = 1
proba_parkinson = stats.norm(mean_parkinson, std_parkinson).pdf(valid_data)* p_parkinson/p_data
proba_healthy = stats.norm(mean_healthy, std_healthy).pdf(valid_data)* p_healthy/p_data

# Probability density functions
plt.figure()
plt.scatter(valid_data, proba_parkinson, marker='.' ,color='darkred', linewidth=2, label='early parkinson')
plt.scatter(valid_data, proba_healthy, marker='.', color='darkgreen', linewidth=2, label='healthy')
plt.legend()
plt.title('Prediction for the data')
plt.xlabel('SBR of right putamen')
plt.ylabel('probabilities')
plt.show()

# %% [markdown]
'''
After we calculated the posterior for every data, the MAP (maximum aposteriori prediction) can be used to find to which class our measure comes from.
If $p(\omega_p|\mathbf{x}) > p(\omega_h|\mathbf{x})$  that means that the subject is more likely to have early parkinson.
'''
# %%
# MAP
estim_labels = np.ones(len(proba_parkinson), dtype = np.int)
estim_labels[proba_healthy > proba_parkinson] = 0

# %% [markdown]
# ### 2.5 Precision of the classifier
# %% [markdown]
# There are many tools to compare the precision of a classifier with another.
# I will focus on two widely used : the confusion matrix and the ROC curve.
# %% [markdown]
# The confusion matrix is usefull to compare how our classifier react to one data or another. It can be used for many classes but in our case we have just two conditions, either the subject has early parkinson $\omega_i = 1$, or is healthy $\omega_i = 0$.

# %%
### confusion matrix
confusion = pd.crosstab(pd.Series(estim_labels, name='Predicted'), pd.Series(valid_label, name='Actual'))

print(confusion)
print("\nPrecision healthy : %1.3f" %(confusion[0][0]/ (confusion[0][0]+confusion[0][1])))
print("Precision early parkinson : %1.3f" %(confusion[1][1]/ (confusion[1][1]+confusion[1][0])))

# %% [markdown]
'''
Another usefull tool is called receiver operating characteristic (ROC) curve, it was used in during World War II to analyse the ability of radars to detect ennemies.
 
The idea is to visualize the true positive rate (TPR), and false positive rate (FNR) for some specific data points. A perfect is an horizontal line passing through (0,1) to (1,1).
For this example, we will take uniformly cut points inside all the validation data. Then we can calculate step by step the TPR and FPR, by taking all the data inferior to the specific cut-point. In our case, a true positive  is an early parkinson subject predicted as parkinson, a false positive is an healthy patient predicted as parkinson.
If you want more details on ROC curve, check this : https://twitter.com/cecilejanssens/status/1104134423673479169/photo/1
'''
# %%
# ROC curve
# using 50 cut points log spaced
n_cut_points = 50
mini = np.min(valid_data[valid_data>0])
maxi = np.max(valid_data)
cut_points = np.geomspace(mini, maxi, n_cut_points)
# cut_points = np.sort(cut_points)
TPR = np.array([])
FPR = np.array([])

for cut_point in cut_points:
    estim = estim_labels[valid_data < cut_point]
    actual = valid_label[valid_data < cut_point]
    TPR = np.append(TPR, (sum((estim == 1) & (actual == 1)) + 1e-16)/(sum(actual == 1) + 1e-16) )
    FPR = np.append(FPR, (sum((estim == 1) & (actual == 0)) + 1e-16)/(sum(actual == 0) + 1e-16) )
TPR = np.append(TPR, 0) 
FPR = np.append(FPR, 0)

plt.figure()
plt.plot(np.array([0,1]), np.array([0,1]), 'k--', label = 'random classifier')
plt.plot(FPR, TPR, 'b+-', label = 'bayes classifier')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.show()

# %% [markdown]
'''
I advice you to use ROC curve implementation from scikit learn, which is more precise, and less prone to floating point arithmetic error.
'''
# %%
# ROC curve with scikit-learn
FPR, TPR, _ = roc_curve(valid_label, proba_parkinson)

# %% [markdown]
'''
## Conclusion
'''
# %% [markdown]
'''
We can see that the bayesian classifier can perform quite well to predict healthy patient (precision healthy > 95%) but there are still improvements to be made to detect early parkinson condition.
Maybe the feature is not enough separable ?
Of course the biggest challenge for this type of classifiers is the need to know the distribution of the data.
That's why many works simply bypass the data feature extraction and modelling (because the data is too complicated or highly-dimensionnal) by using a complex classifier and tuning its parameters (like deep NN, random forest...)
or non-parametric generative model (gaussian processes, kernel density estimation..).
This with the hope that the high dimensionnality of the model can help to learn the latent distribution of the data itself.
'''
# %% [markdown]
'''
## Appendix A
'''
# %% [markdown]
'''
Because I am not allowed to share the data from the PPMI cohort, we will just estimate the probability density function representing the two conditions.
We could fit a simple non-parametric generative model with gaussian kernel estimation on the SBR on right putamen, using scikit-learn (https://scikit-learn.org/stable/modules/density.html).
After estimating the distribution of the data, we can sample the data from these two distributions as above.
'''
# %%
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# fix random state
seed = 0
np.random.seed(seed)

# Loading of data from PPMI database
with open('YOUR_DIR/parkinson_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

nb_classes = len(np.unique(data["label"]))
n_samples = len(data["label"])
idx = np.random.permutation(n_samples)
right_putamen = data["SBR"][idx, 2].reshape(-1, 1)
labels = data["label"][idx] == 1

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
'''