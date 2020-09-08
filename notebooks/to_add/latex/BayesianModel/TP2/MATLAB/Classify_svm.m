function [rate output] = Classify_svm(X,Y,Xt,Yt,gkernel,C,Cj)
% [rate output] = Classify_svm(X,Y,Xt,Yt,gkernel,C,Cj)
% Train a SVM classifier and compute the error rate on teste set
% Also this function return the row output of the SVM for each set sample
% The learning algorithm use SVMLight of Joachims and mex function of Tom
% Inputs:
% X -- Training set in columns of dim (num pattern, num features)
% Y -- Labels of training set of size (num pattern,1)
% Xt -- Testing set in columns of dim (num pattern, num features)
% Yt -- Labels of testing set of size (num pattern,1)
% gkernel -- gaussian kernel parameter gamma=1/sigm^2
% C -- hyperparameter balancing the trade-off between margin and training error
% Cj -- ratio num_negative/num_positive
% Returns:
% rate -- error rate on test set Xt
% output -- row output of the SVM 
%
% Mathias Adankon -- November 2009 -- mathias.adankon@synchromedia.ca


fprintf('Training ...\n');
param=['-t 2 -g ' num2str(gkernel) ' -c ' num2str(C) ' -j ' num2str(Cj)] ;
model = mexsvmlearn(X,Y,param);    
fprintf('Computing output for test samples, ...\n');
[rate,output] = mexsvmclassify(Xt,Yt,model); 
