function [Py,label,model,sum_log_lklh,lklh_gmm,lklh_gauss,time] = EM_GMM(M,nb_k,rnd_init)
%GMM 
% nb_k : Number of class
% rnd_init : If true random initialization


disp(['EM for Gaussian mixture model with ' num2str(nb_k) ' gaussians ...']);
nb_pixel = size(M,1);
maxiter = 1000;
miniter = 5;
sum_log_lklh = [];

%% With first feature space

nb_pixel = size(M,1); 
d = size(M,2); %Number of dimensions for the features
model = struct( 'weights', zeros(1,nb_k),...
                'means', zeros(nb_k,d),...
                'covariances', zeros(d,d,nb_k));
label = uint8(zeros(nb_pixel,1));
label_tmp = uint8(M*0);

if rnd_init
    label = randi(nb_k,nb_pixel,1);
else
    fen = [0:(1/nb_k):1];
    q = quantile(M,fen);
    if d==1
        q = q(:);
    end
    for i=1:nb_k
        label_tmp(and(ones(nb_pixel,1)*q(i,:)<=M, M<=ones(nb_pixel,1)*q(i+1,:)))=i;
    end
    label = mode(label_tmp,2);
    if 0.95*nb_pixel < sum(label == uint8(ones(nb_pixel,1))*label(1)) %If the labels are not good
        label = randi(nb_k,nb_pixel,1);
    end
end

conv = false;
i = 1;

for k=1:nb_k
    ind = (label == k);
    model.weights(k) = 1/nb_k;
    model.means(k,:) = mean(M(ind,:));
    model.covariances(:,:,k) = cov(M(ind,:));
    if det(model.covariances(:,:,k))<1
        tmp = model.covariances(:,:,k);
        tmp(1:(size(tmp,1)+1):end) = tmp(1:(size(tmp,1)+1):end)+rand(1,d)+100;
        model.covariances(:,:,k) = tmp;
    end
end

[PyKx,lklh_gauss,sum_log_lklh(1)] = expectation(M,model);
Py{1}=lklh_gauss;

tic;

while i < maxiter && ~conv
    
%     disp(['Iteration ' num2str(i) ' and Current Log-Likelihood = ' num2str(sum_log_lklh(i))]);
    i = i+1;
    model = maximization(M,PyKx);
    [PyKx,lklh_gauss,sum_log_lklh(i),lklh_gmm] = expectation(M,model);
    % Convergence
    if i>miniter
%         conv = or((sum_log_lklh(i) - sum_log_lklh(i-1)) < abs(sum_log_lklh(i))*1e-6 , (sum_log_lklh(i)-sum_log_lklh(i-5))/5 < sum_log_lklh(i)*1e-6);  
        conv = (sum_log_lklh(i) - sum_log_lklh(i-1)) < 0.01;
    end
end

time = toc;

% disp(['Iteration ' num2str(i) ' and Current Log-Likelihood = ' num2str(sum_log_lklh(i))]);

%Labellisation
[~,label] = max(log(lklh_gauss),[],2);
Py{2} = lklh_gauss;


if conv
    disp(['Converged in ' num2str(i) ' steps and ' num2str(time) 's.']);
else
    disp(['Not converged after ' num2str(i) ' steps and ' num2str(time) 's.']);
end

time = time/i;

end