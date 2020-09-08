import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Creation of data
RadiusCherries = np.random.normal(4, 0.5, 1000)
RadiusGrapes = np.random.normal(2, 0.5, 1000)

# Estimators for gaussian mean and deviation
MeanCherries = np.median(RadiusCherries)
DevCherries = np.std(RadiusCherries)
MeanGrapes = np.median(RadiusGrapes)
DevGrapes = np.std(RadiusGrapes)

#pdf of data
xCherries = np.linspace(MeanCherries - 3*DevCherries, MeanCherries + 3*DevCherries, 1000)
fCherries = stats.norm(MeanCherries, DevCherries).pdf(xCherries)
xGrapes = np.linspace(MeanGrapes - 3*DevGrapes, MeanGrapes + 3*DevGrapes, 1000)
fGrapes = stats.norm(MeanGrapes, DevGrapes).pdf(xGrapes)

# Probability density functions
plt.figure()
plt.hist(RadiusCherries, normed = 1, bins = 25, color = 'red', alpha = 0.5, label = 'Cherry data')
plt.hist(RadiusGrapes, normed = 1, bins = 25, color = 'green', alpha = 0.5, label = 'Grape data')
plt.plot(xCherries, fCherries, color = 'darkred', linewidth = 2, label = 'Cherry model')
plt.plot(xGrapes, fGrapes, color = 'darkgreen', linewidth = 2, label = 'Grape model')


#New radius data coming in
Data = np.array([1.52, 2.68, 3, 5, 6])

#Bayes
# P(w_c) = P(w_g)
PAprioriCherry = PAprioriGrape = 0.5
PData = 1
PCherry = stats.norm(MeanCherries, DevCherries).pdf(Data) * PAprioriCherry/PData
PGrape = stats.norm(MeanGrapes, DevGrapes).pdf(Data) * PAprioriGrape/PData

for i in range(0, len(Data)):
    if PGrape[i] > PCherry[i]:
        plt.plot(Data[i], 0.02, 'o', color='darkgreen', linewidth=10, markersize=12)
    else:
         plt.plot(Data[i], 0.02, 'o', color='darkred', linewidth=10, markersize=12)
plt.legend()
plt.title('Fruit detection depending on radius')
plt.xlabel('Radius (cm)')
plt.ylabel('Probability')
