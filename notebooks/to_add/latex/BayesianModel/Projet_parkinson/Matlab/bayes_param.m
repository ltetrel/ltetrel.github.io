function [label_out, P, mat_conf]= bayes_param(database, label)
%function [label_out, P, mat_conf]= bayes_param(database, label)

%%
nb_classes = length(unique(label));
n = length(label);
indices = randperm(n);
label_out = zeros(1,n);
mat_conf = zeros(nb_classes);
length_train = floor(n/3);
length_test = n - length_train;

data_learning = database(indices(1:length_train),:);
labels_learning = label(indices(1:length_train));
data_test = database(indices(length_train+1:n),:);
labels_test = label(indices(length_train+1:n));

%% Training

for c = 1:nb_classes
    S(:,:,c) = cov(data_learning(labels_learning==c-1,:));
    M(c,:) = mean(data_learning(labels_learning==c-1,:))';
end

iy1 = pdf('normal', data_learning(~labels_learning,1), M(1,1), S(1,1,1));
iy1 = iy1/10;
iy2 = pdf('normal', data_learning(labels_learning,1), M(2,1), S(1,1,2));
iy2 = iy2/10;

[new_x1,I]=sort(data_learning(~labels_learning,1));
new_density1= iy1(I);

[new_x2,I]=sort(data_learning(labels_learning,1));
new_density2= iy2(I);

plot(new_x2, new_density2, '-r', new_x1, new_density1,'-g','Linewidth',2);
hold on;
plot(data_learning(labels_learning,1), iy2, '+r', data_learning(~labels_learning,1), iy1,'+g','Linewidth',2);
hold off;

legend('Early Parkinson','Normal');
title('Gaussian estimation of SBR of the Right Caudate');
xlabel('SBR of the Right Caudate');

%% Labelisation

for i = 1:length_test

    for j = 1:nb_classes
        P(i,j) = posteriori(data_test(i,:)',j,M,S);
    end
 
    [~, ind] = max(P(i,:));
    label_out(i) = ind(1)-1;
    mat_conf(labels_test(i)+1, label_out(i)+1) = mat_conf(labels_test(i)+1, label_out(i)+1) + 1;
 
end
 
label_out = logical(label_out);
end

function Pcx = posteriori(x, c, M, S)
    Sc = S(:,:,c);
    Mc = M(c,:)';
    Pcx = -0.5*log(det(Sc))-0.5*((x-Mc)')*inv(Sc)*(x-Mc);
end