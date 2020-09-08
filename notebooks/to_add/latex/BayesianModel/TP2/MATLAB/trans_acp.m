function data_acp = trans_acp(data,nc)

%data_acp = trans_acp(data,nc)
% data: number of samples times dimension
% nc: number of composants

[n d] = size(data);
moy = mean(data);
X = data - ones(n,1)*moy;
V = cov(data);
[vp h] = eig(V);
vp0 =  fliplr(vp);
data_acp = data*vp0(:,1:nc);
% Computing the variability curve
% figure
% [val_p, ind] = sort(diag(h));
% val_p = flipud(val_p);
% plot(cumsum(val_p)/sum(val_p)*100,'b-')
% sum(val_p(1:8))/sum(val_p)*100
