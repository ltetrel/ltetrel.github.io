function [error_rate, C, K_param] = SVM_param(database, label)
% function [error_rate, C, K_param] = SVM_param(database, label)

%% Variables

C = [1e5 1e-4 1e-3 1e-2 1e-1 1e1 1e2 1e3 1e4 1e5];
% K_param = [1]; % Lineaire
% K_param = {1:5; [1e-4 1e-3 1e-2 1e-1 1e1 1e2 1e3 1e4]; [1e-4 1e-3 1e-2 1e-1 1e1 1e2 1e3 1e4]}; % Polynomial avec [d; s; c]
K_param = {[1e-5 1e-4 1e-3 1e-2 1e-1 1e1 1e2 1e3 1e4 1e5]}; % Gaussien
% K_param = {[1e-5 1e-4 1e-3 1e-2 1e-1 1e1 1e2 1e3 1e4 1e5]; [1e-5 1e-4 1e-3 1e-2 1e-1 1e1 1e2 1e3 1e4 1e5]}; % Sigmoid

error_rate = zeros(length(C),length(K_param));

n = length(label);
Cf = sum(~label)/sum(label);
indices = randperm(n);
length_train = floor(n/3);
length_valid = n - length_train;

if size(label,1) == 1             % Pour la fonction SVM
    label = double(label)';
end

for i=1:n 
   if label(i) == 0  
      label(i) = -1;   
   end
end

data_train = database(indices(1:length_train),:);
label_train = label(indices(1:length_train));
data_valid = database(indices(length_train+1:n),:);
label_valid = label(indices(length_train+1:n));

%% Processus

tic;
for i = 1:length(C)
    tstart = tic;
    
    for j = 1:length(K_param{1})     
                
    param=['-v 0 -t 2 -g ' num2str(K_param{1}(j)) ' -c ' num2str(C(i)) ' -j ' num2str(Cf)];  %Ligne à changer en fonction du noyau
    model = mexsvmlearn(data_train, label_train, param); 
    [error_rate(i,j),~] = mexsvmclassify(data_valid, label_valid, model);  

        j,
    end
    
    temps=toc(tstart)*(length(C)-i);
    disp([num2str(i/length(C)*100) '%. Estimation du temps restant : ' num2str(temps) 's']);
end

temps_ecoule = toc;
disp(['Temps ecoule : ' num2str(temps_ecoule) 's']);
end
