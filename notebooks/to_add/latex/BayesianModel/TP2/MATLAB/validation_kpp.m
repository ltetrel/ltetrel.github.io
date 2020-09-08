function [error_rate, temps_ecoule, densites] = validation_kpp(database, labels, k)
% [taux_error, toc, densites] = validation_kpp(database, labels, k)

nb_classes = 10;
error_rate = 0;
q = 0.7; 
% 70% de la base de training sera utilise comme entrainement. 
% 30% de la base de training servira de test. On calcule pour chaque élément de cette base la distance euclidienne avec à chaque fois tous les elements de base de training.
m = q*length(database)/nb_classes;
densites = cell(1,nb_classes);

lines = zeros(m/q,1);
lines(1:m,1) = 1;
lines = logical(repmat(lines,10,1));
database_training = database(lines,:);
labels_training = labels(lines);
database_valid = database(not(lines),:) ;
labels_valid = labels(not(lines));

% eucli =  sqrt(sum(database_valid.^2,2)); %Si on calcule les densités sur les normes

eucli =  database_valid(:,1);%Si on calcule les densités sur la premierre composante (le plus d'informations selon ACP)

for i=1:nb_classes
    
    densites{i}=zeros(length(labels_valid),2); % Pour chaque classe il y a n_valid points, 1ere colonne proba et 2eme colonne norme du point
    densites{i}(:,2) = eucli;
end

eucli = zeros(length(database_training),length(database_valid));

tic;

for i=1:length(database_valid)
    
    tstart = tic;

%% Recuperation des plus proches voisins

    diff = repmat(database_valid(i,:),m*nb_classes,1)-database_training ;
    eucli(:,i)= sum(diff.^2,2);
    [~,I]=sort(eucli(:,i));
    label_voisin = labels_training(I(1:k)); %On recupere les k proches voisins du nieme point
    
    if mode(label_voisin) ~= labels_valid(i) %Alors on a mal classifie car le vote est mauvais
        error_rate = error_rate+1;
    end
   
%% Calcul des resultats

%     V =  2*abs(norm(database_valid(i,:),2) - norm(database_training(I(k),:),2)); % Volume de la boule pour les probabilites 2*|Xi - Xkième voisin|
    V =  max(2*abs(ones(k,1)*database_valid(i,1) - database_training(I(1:k),1))); % Volume (le max pour éviter probleme) si on choisit de travailler sur la 1ere composante
    
    for j=1:length(label_voisin) % On calcule les probabilités de chaque classe pour le point en cours (utiliser unique(label_voisin) si trop long)
       P = (sum(label_voisin(j)==label_voisin)/sum(label_voisin(j)==labels_valid))/V; 
       densites{label_voisin(j)}(i,1) = P;
    end
 
    temps=toc(tstart)*(length(database_valid)-i);
    disp([num2str(i/length(database_valid)*100) '%. Estimation du temps restant : ' num2str(temps) 's']);
    
end


error_rate = error_rate/length(labels_valid);
temps_ecoule = toc;
disp(['Temps ecoule : ' num2str(temps_ecoule) 's']);

end

