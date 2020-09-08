# %% [markdown]
'''
# Covid-19 - The danger of false-positive
'''
# %% [markdown]
'''
Covid-19 is an infectious disease caused by the SARS-CoV-2 virus that evolved into a pandemic, because it massively spread around the world in a similar way to the flu. 
Most people with Covid-19 have a low to mild respiratory illness, or no symptoms at all.
Others people (old or with chronic disease) experience severe symptoms like pneumonia and need intensive care.

In order to protect the most fragile, politics decided to put most nations of the earth into one of the biggest confinement humanity had ever faced.
This is now usually referred as *the great lockdown* <cite> wahrungsfonds2020world </cite>.

Behind this act hides a fear of the unknown. Hence lot of decisions were taken in a hurry, most notably regarding the screening of the disease.

In this post, I will talk about the danger of innaccuracy and mostly the false positive rate in the tests.

>**Disclaimer**:  
>This post represents solely my opinion on the subject, and should not be used as a scientific evidence.
>My opinion is mostly supported after a litterature review.
'''
# %% [markdown]
'''
## tl;dr
1. Two types of tests, molecular (match virus profile) and serological (detect immune response).
2. Current Covid-19 screening has really low specificity (lot of false positive).
3. Low specificity leads to over-diagnosis and biased political responses (leading to financial crisis, civil liberties reduction etc...).
'''
# %% [markdown]
'''
## 1. Mathematical background
'''
# %% [markdown]
'''
### 1.1 Data modeling
'''
# %% [markdown]
'''
I will first define what do we mean by accuracy in statistics.

Let's say we want to validate a new Covid-19 test.
In statistics, this problem can be formalized with the following null hypothesis ($H_0$): "the patient has Covid-19".  
The features used as a discrimination criteria could be for example the amount of maximum fluorescence in a fluid sample from the nose (in case of a PCR test).
We can represent the distributions for the two populations (healthy vs non-healthy) as two gaussians, where the feature is on the $x$-axis.
These distributions are of course known when validating a model, the ground truth data is labelled with the information of who has the covid.
'''
# %% [code]
### imports
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_curve

import warnings
warnings.filterwarnings('ignore')
# %% [code]
# Data creation
# fix random state
seed = 0
np.random.seed(seed)

# number of subjects and infected
n_healthy = int(7.76e6)
n_covid = int(2.4e5)

# parameters of the distributions
a_mean = 40
b_mean = 80
std_dev = 10

# Generate samples from classes
X_a = np.random.randn(n_healthy) * std_dev + a_mean + np.random.rand(n_healthy)
X_b = np.random.randn(n_covid) * std_dev + b_mean + np.random.rand(n_covid) 

# %% [code]
### plot pdfs
# data preprocessing
Xa_sorted = np.sort(X_a[np.random.choice(X_a.shape[0], 10000, replace=False)])
Xb_sorted = np.sort(X_b[np.random.choice(X_b.shape[0], 10000, replace=False)])
pdf_a = stats.norm.pdf(Xa_sorted, a_mean, std_dev)
pdf_b = stats.norm.pdf(Xb_sorted, b_mean, std_dev)
# plot
plt.figure()
plt.plot(Xa_sorted, pdf_a, color='green', linewidth=3, linestyle='-', label='healthy')
plt.plot(Xb_sorted, pdf_b, color='darkred', linewidth=3, linestyle='-', label='Covid-19')
plt.fill(Xa_sorted, pdf_a, color='green', alpha=0.75)
plt.fill(Xb_sorted, pdf_b, color='darkred', alpha=0.75)
plt.grid()
plt.legend()
plt.title('Distributions for healthy and Covid-19 subjects')
plt.xlabel('Maximum fluorescence level')
plt.show()
# %% [markdown]
'''
As you can see, the two gaussians overlaps by definition (the gaussian is strictly positive anywhere if $\sigma > 0$).
And because they overlap they will be innacuracy, of course the point is to minimize this error (maximize the distance between the two curves) so that the test is more acccurate.
'''
# %% [markdown]
'''
### 1.2 Type I and Type II errors
'''
# %% [markdown]
'''
Anytime we make a statistical decision there are four possible outcomes: two representing correct decisions and two representing errors. These outcomes can be represented by the following **confusion matrix**:

|                                            | Patient is <br> "Covid-19" | Patient is<br> "not Covid-19" |   |
|--------------------------------------------|----------------------------|-------------------------------|---|
| Test gives "Covid-19" <br> (accept H0)     | True positive              | Type I error                  |   |
| Test gives "not Covid-19" <br> (reject H0) | Type II error              | True negative                 |   |
|                                            |                            |                               |   |

The type I error is often called **false positive** (the patient does not have Covid-19 but the test is positive), where the type II error is the **false negative** (the patient has Covid but the test is negative).

By defining our decision threshold, we can highlight all the different outcomes on our ditributions where each probability is given by integrating the correct regions.  
In the following example, we fix the threshold to $56$ (if a patient has a fluorescence level more than $56$, he is considered as having Covid-19):
'''
# %% [code]
### Probabilities of the four possible outcomes
# data classification
decision = 56.051
tp = Xa_sorted[Xa_sorted < decision]
fp = Xa_sorted[Xa_sorted >= decision]
fn = Xb_sorted[Xb_sorted < decision]
tn = Xb_sorted[Xb_sorted >= decision]
# plot
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(Xa_sorted, stats.norm.pdf(Xa_sorted, a_mean, std_dev), color='green', linewidth=3, linestyle='-', label='healthy')
plt.plot(Xb_sorted, stats.norm.pdf(Xb_sorted, b_mean, std_dev), color='darkred', linewidth=3, linestyle='-', label='Covid-19')
plt.fill_between(tn, stats.norm.pdf(tn, b_mean, std_dev), color='darkred', alpha=0.75, label='true positive')
plt.fill_between(fn, stats.norm.pdf(fn, b_mean, std_dev), color='green', alpha=0.75, label='false negative')
plt.grid()
plt.legend()
plt.title('Distributions for healthy and Covid-19 subjects')
plt.subplot(2, 1, 2)
plt.plot(Xa_sorted, stats.norm.pdf(Xa_sorted, a_mean, std_dev), color='green', linewidth=3, linestyle='-', label='healthy')
plt.plot(Xb_sorted, stats.norm.pdf(Xb_sorted, b_mean, std_dev), color='darkred', linewidth=3, linestyle='-', label='Covid-19')
plt.fill_between(tp, stats.norm.pdf(tp, a_mean, std_dev), color='green', alpha=0.75, label='true negative')
plt.fill_between(fp, stats.norm.pdf(fp, a_mean, std_dev), color='darkred', alpha=0.75, label='false positive')
plt.grid()
plt.legend()
plt.xlabel('Maximum fluorescence level')
plt.show()
# %% [markdown]
'''
>**Note**:  
>We can see that the false negatives and false positives are complementary. Indeed, decreasing the decision threshold to lower the false negatives increase the false positives.
'''
# %% [markdown]
'''
### 1.3 Specificity vs Sensitivity
'''
# %% [markdown]
'''
The different outcomes probabilities can be used to define two other measures, the "specificity" and "sensitivity".
The **sensitivity** or true positive rate ($TPR$) is the proportion of positives that are correctly identified, on the other hand the probability of correctly identified negatives is the **specificity** or true negative rate ($TNR$).
They are calculated as follow <cite> yerushalmy1947statistical </cite>:

\begin{equation}\label{eq_s}
TPR = \frac{TP}{TP + FN},
\end{equation}
\begin{equation}
TNR = \frac{TN}{TN + FP},
\end{equation}

Where the numbers of true posistives ($TP$), true negatives ($TN$), false negatives ($FN$) and false positives ($FP$) are defined.

These two probabilities are widely used in the medical litterature <cite> wang2020detection </cite> <cite> ahn2014korea </cite> and **essential for validation of antigen tests**.
'''
# %% [markdown]
'''
## 2. Current screening tests for the Covid-19
'''
# %% [markdown]
'''
### 2.1 The two different types of tests
'''
# %% [markdown]
'''
<img src="imgs/false_positive/test.jpg" alt="drawing" width="500"/>

We can divide the tests for the Covid-19 into two main categories: the first one detects the immune response to the virus, the second one find the genetic code of this virus.

Detecting the immune response to the virus is a **serological test**. It looks for evidence in the blood of the first responder IgM antibodies (that appear about a week after infection), as well as longer-lasting IgG antibodies (two to four weeks after infection) <cite> world2020coronavirus </cite>.

Finding the trace for a virus is called a **molecar test**. Indeed, coronaviruses contains genetic material through Ribonucleic acid (RNA) that is unique to that virus.
To transform the RNA into a DNA (thus allowing matching the profile of the virus), a process called reverse transcriptase-polymerase chain reaction (RT-PCR) is used <cite> world2020coronavirus </cite>.  
To perform the test, a practitioner sample liquid from the patient's nose, this sample is later tested for the amount of fluorescence to check if it matches the virus profile (after the RT-PCR process).

The ideal test is of course accurate, cheap, return results quickly and should be easy to perform outside of a laboratory.
'''
# %% [markdown]
'''
### 2.2 Accuracy
'''
# %% [markdown]
'''
The is a lot of controversial in the accuracy of the tests. Just google it and you will find a huge number of posts raising warnings like [this](https://theconversation.com/coronavirus-tests-are-pretty-accurate-but-far-from-perfect-136671), or [here](https://www.medicinenet.com/script/main/art.asp?articlekey=228250)!

In the litterature, Wang et al. pointed out that for nasal swabs samples, the specificity was as low as 63% <cite> wang2020detection </cite>!
Even if the PCR tests are considered the gold standard <cite> lippi2020potential </cite> for lab measurement, the issue here is that it clearly lacks of accuracy in a real-world context (too much human uncertainties).
Also, the cross-effect of different symptoms from the bacteria and viruses that are active during the flu season tends to increase the false-positive rate.

Another issue here is that medicine in general tends to **better consider sensitivity over specificity** when developping drugs.
And this is especially true for Covid (but not for all tests, check the [table 12 from this RT-PCR Kit](https://www.fda.gov/media/136472/download)).  
Why? Because it is better to say that someone if infected but he is not, rather than saying he is not infected but he is (hence spreading the disease).
And because sensitivity and specificity are complementary, increasing one decrease the other (this can be easily seen with eq \ref{eq_s}, lowering $FN$ increase $FP$ thus reducing $TNR$).

Finally, they are evidences that the more tests we do, the more likely we have the disease <cite> ahn2014korea </cite>.
'''
# %% [markdown]
'''
### 2.3 Experimentation
'''
# %% [markdown]
'''
Coming back to our little experimentation, let's say that in Auvergne-Rhone-Alpes (France) the prevalence of Covid-19 is estimated at around 3%, so there would be $240$k infected and $7.76$M healthy.
what happenned if we test all this populaiton with a PCR of 94% specificity?

'''
# %% [code]
# create inputs X and targets C
X = np.hstack((X_a, X_b))
C = np.hstack( (np.zeros((n_healthy), dtype=bool), np.ones((n_covid), dtype=bool)) )

# random permutation
idx = np.arange(X.shape[0])
idx = np.random.permutation(idx)
X = X[idx,]
C = C[idx,]
# %% [code]
### Confusion matrix and probabilities
decision = 56.051
tp = np.sum( ((X >= decision) == C)[C] )
fn = n_covid - tp
tn = np.sum( ((X < decision) == ~C)[~C] )
fp = n_healthy - tn

print("Sensitivity: {}".format(tp/(tp + fn)))
print("Specificity: {}".format(tn/(tn + fp)))
print()
print("TP: {}\tFP: {}".format(tp, fp))
print("FN: {}\tTN: {}".format(fn, tn))

# %% [markdown]
'''
At this really low-level, the risk of false positives becomes a major problem.
The number of false positive ($465.6$k) is much bigger than the actual number of Covid cases ($240$k).
Basically, this is nearly twice more people detected as infected than what it should be!
'''
# %% [markdown]
'''
### 2.4 Consequences of a high false-postive rate
'''
# %% [markdown]
'''
One of the first consequence of a high false-postive rate is the **overdiagnosis** of patients.
Overdiagnosis refers to the identification of abnormalities  that  were  never  going  to  cause  harm <cite> brodersen2018overdiagnosis </cite>.

For example imagine you go to a dentist, and he says that you have a tooth decay but you don't have one.
The consequence of a bad chirurgy can be catastrophic for your mouth, and not to mention the impact on your finances (even worth if you "need" a prothesis).
*From my professional and personnal experience, this practice it is not as unusual in this field*.

Not to mention that from the point of view of a bad doctor, the legal risk of healing a non-existing disease (and making lot of money out of it) is much lower than risking to do nothing (even if not sure).
I think this is one of the biggest issue in medicine.

Another consequence, and we are in the midddle of it, is **population frightening**.
This causes for example our governments to remove [some important restrictions to publish new tests](https://www.scientificamerican.com/article/breakthrough-Covid-19-tests-are-currently-cheap-fast-and-not-very-accurate1/).
And I am not even talking about the reduction on our civil liberties, or exposing the world to finanncial breakdowns and crisis that can affects millions of people (and lot of deaths in low-income countries).
'''
# %% [markdown]
'''
## Conclusion
'''
# %% [markdown]
'''
We saw the importance of the false positive rate (related to the specificity measure) and its geopolitical, health impact.
I think the biggest error when comparing specificity and sensitivity is that both should be important, and that no test is better than a bad test.
'''
# %% [markdown]
'''
## To go further
'''
# %% [markdown]
'''
A question I asked myself when writing this post was how to know that a patient is really covid (when validating a new test)?
It is possible to find lesions in the lung or heart (caused by covid-19) by performing a chest CT scan <cite> bernheim2020chest </cite>.

Another important topic is the reproducibility of the Covid-19 research <cite> colquhoun2017reproducibility </cite>.

Finally, a question that was not raised here but rather important is the dynamic of the epidemy.
We still lack some parameters to fully understand the dynamic of the virus <cite> xiang2020antibody </cite>.
'''
# %% [markdown]
'''
## Tags
'''
# %% [markdown]
'''
Data-Science; Statistics; Health-Care
'''
