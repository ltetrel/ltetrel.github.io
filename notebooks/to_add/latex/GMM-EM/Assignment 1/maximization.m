function model = maximization(M,PyKx)
% M : Feature space of image
% label : labels for all the pixels

y=[];
nb_pixel = size(M,1);
nb_k = size(PyKx,2);
d = size(M,2);
sumy = sum(PyKx,1);

for k=1:nb_k   
    model.weights(k) = sumy(k)/nb_pixel;
    model.means(k,:) = sum((PyKx(:,k)*ones(1,d)).*M)/sumy(k);
    
    model.covariances(:,:,k) = diag(sum((PyKx(:,k)*ones(1,d)).*(M-ones(nb_pixel,1)*model.means(k,:)).^2)/sumy(k));
    ind = combnk(1:d,2);
    
    for j=1:size(ind,1)
        model.covariances(ind(j,1),ind(j,2),k) = sum(PyKx(:,k).* ((M(:,ind(j,1))-ones(nb_pixel,1)*model.means(k,ind(j,1)))).* ...
                                                             (M(:,ind(j,2))-ones(nb_pixel,1)*model.means(k,ind(j,2))))/sumy(k);
        model.covariances(ind(j,2),ind(j,1),k) = model.covariances(ind(j,1),ind(j,2),k);
    end
    if det(model.covariances(:,:,k))<1
        tmp = model.covariances(:,:,k);
        tmp(1:(size(tmp,1)+1):end) = tmp(1:(size(tmp,1)+1):end)+rand(1,d)+100;
        model.covariances(:,:,k) = tmp;
    end
end

end