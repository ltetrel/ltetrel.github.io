function [PyKx,lklh_gauss,sum_log_lklhgmm,lklh_gmm] = expectation(M,model)
% M : Feature space of image
% model : parameters of the gmm

nb_pixel = size(M,1);
nb_k = size(model.means,1);

for k=1:nb_k

    y(:,k) = mvnpdf(M,model.means(k,:),model.covariances(:,:,k));
    y(y(:,k)==0,k) = realmin('double');
    
end

lklh_gauss = y.*(ones(nb_pixel,1)*model.weights);
lklh_gmm = sum(lklh_gauss,2);
sum_log_lklhgmm = sum(log(lklh_gmm));
PyKx = lklh_gauss./(lklh_gmm*ones(1,nb_k));

end