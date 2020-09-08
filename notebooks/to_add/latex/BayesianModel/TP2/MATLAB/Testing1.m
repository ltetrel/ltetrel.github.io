
clear
clc
% load retine_10x10_learning
% rate = overlap_v3(database',labels)

% load Retine_10x10_training
% data = trans_acp(database,15);
load retine_10x10_acp_52_learning
data = database;

index = randperm(6000);
ind1 = index(1:5000);
ind2 = index(5001:6000);

% mais utiliser mes indices (les memes pour tout le monde)
load indexTr
load indxTst
gkernel=0.002;
gam = 10;

for k=1:2:31

% [error_rate, temps_ecoule] = Classify_KNN(data(ind1,:), labels(ind1), data(ind2,:), labels(ind2), k);
[error_rate, temps_ecoule] = Classify_KNN(data(ind1,:), labels(ind1), data(labels(ind2)==10,:), labels(ind2), k);
%[rate1 predict] = SVM_1vsAll(data(ind1,:), labels(ind1), data(ind2,:), labels(ind2), gkernel,gam) ;
%[rate3 predict]= Classify_QBayes(data(ind1,:), labels(ind1), data(ind2,:), labels(ind2));

error(round(k/2)) = error_rate;
temps(round(k/2)) = temps_ecoule;

end
