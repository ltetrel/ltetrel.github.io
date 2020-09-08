function [label,label_dist] = eig_classify(weights_train,weights_test,n_eigen,NN)

label = [];
label_dist = [];

if NN
    for j=1:size(weights_test,1)
        diff = ones(size(weights_train,1),1)*weights_test(j,1:n_eigen) - weights_train(:,1:n_eigen);
        dist_eucli(:,j) = sqrt(sum(diff.^2,2));
    end
    [label_dist,label] = min(dist_eucli);
else
    
%% Part 1 
%     ind_train = (1:28:(53*28));
%     for i=1:52
%         w_train = weights_train(ind_train(i):(ind_train(i+1)-1),1:n_eigen);
%         p(:,i) = mvnpdf(weights_test(:,1:n_eigen),mean(w_train),cov(w_train));
% %         y = normpdf([0:0.1:200],mean(w_train),std(w_train));
% %         plot([0:0.1:200],y);
%     end
%     [~,label]=max(p,[],2);
    
%     ind = [145 60 198 7 102 87];
%     load('Data_im_single','name_test');
%     figure;
%     for i=1:6
%         subplot(2,3,i);
%         stem(1:52,p(ind(i),:));
%         xlabel('Subject label');
%         ylabel(['Probability of ' name_test(ind(i),:)],'Interpreter','none');
%     end

%% Part 2
    w_train = weights_train(:,1:n_eigen);
    label_dist = mvnpdf(weights_test(:,1:n_eigen),mean(w_train),cov(w_train));
%         y = normpdf([0:0.1:200],mean(w_train),std(w_train));
%         plot([0:0.1:200],y);

    
end
end

