function [error_rate, temps_ecoule, label_kpp, mat_confusion] = Classify_KNN(database_training, labels_training, database_valid, labels_valid, k)
% [error_rate, temps_ecoule, label_kpp] = Classify_KNN(database_training, labels_training, database_valid, labels_valid, k)

error_rate = 0;
eucli = zeros(length(database_training),length(database_valid));
mat_confusion = zeros(10);

tic;

for i=1:length(database_valid)
    
    tstart = tic;

%% Recuperation des plus proches voisins

    diff = repmat(database_valid(i,:),length(database_training),1)-database_training ;
    eucli(:,i)= sum(diff.^2,2);
    [~,I]=sort(eucli(:,i));
    label_voisin = labels_training(I(1:k)); %On recupere les k proches voisins du ieme point
    label_kpp(i) = mode(label_voisin);
    
    if label_kpp(i) ~= labels_valid(i) %Alors on a mal classifie car le vote est mauvais
        error_rate = error_rate+1;
    end

    mat_confusion(labels_valid(i),mode(label_voisin)) = mat_confusion(labels_valid(i),mode(label_voisin)) + 1;
    temps=toc(tstart)*(length(database_valid)-i);
    disp([num2str(i/length(database_valid)*100) '%. Estimation du temps restant : ' num2str(temps) 's']);
    
end

error_rate = error_rate/length(labels_valid);
temps_ecoule = toc;
disp(['Temps ecoule : ' num2str(temps_ecoule) 's']);

end

