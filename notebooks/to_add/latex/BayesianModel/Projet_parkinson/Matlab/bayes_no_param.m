function [label_out, P, mat_conf, densites]= bayes_no_param(database, label)
%function [label_out, P, mat_conf, densites]= bayes_no_param(database, label)

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

P_priori = (sum(labels_learning))/length_train;

npoints = 100; %npoints = length_train

%% Training

for i = 1:size(data_learning,2)
    
    [f_no(:,i),x_no(:,i)] = ksdensity(data_learning(~labels_learning,i),'kernel','box','npoints', npoints ,'support', 'positive', 'width',1/sqrt(length_train),'function', 'pdf'); 
    [f_pd(:,i),x_pd(:,i)] = ksdensity(data_learning(labels_learning,i),'kernel','box','npoints', npoints ,'support', 'positive','width',1/sqrt(length_train),'function', 'pdf');
    x = [x_no(:,i); x_pd(:,i)];
    SBR_P_x(:,i) = linspace(min(x),max(x),npoints);
    
end

densites = struct('D_no',f_no ,...
                  'SBR_no',x_no ,...
                  'D_pd',f_pd ,...
                  'SBR_pd',x_pd,...
                  'P_x', 0,...
                  'SBR_P_x', 0);

% Calcul de P_X

for i = 1:size(densites.D_no,1)

    for j = 1:size(database,2)
        
        [~,Id_no] = min(abs((ones(size(densites.D_no,1),1)*SBR_P_x(i,j)-densites.SBR_no(:,j)))); % Recuperation des points les plus proches
        [~,Id_pd] = min(abs((ones(size(densites.D_no,1),1)*SBR_P_x(i,j)-densites.SBR_pd(:,j))));
        P_X(i,j) = densites.D_no(Id_no,j)*(1-P_priori) + densites.D_pd(Id_pd,j)*P_priori;
    
    end 
end

densites = struct('D_no',f_no ,...
                  'SBR_no',x_no ,...
                  'D_pd',f_pd ,...
                  'SBR_pd',x_pd,...
                  'P_x', P_X,...
                  'SBR_P_x', SBR_P_x);

plot(densites.SBR_pd(:,1),densites.D_pd(:,1),'-r',densites.SBR_no(:,1),densites.D_no(:,1),'-g','Linewidth',2);
hold on;
plot(densites.SBR_P_x(:,1),densites.P_x(:,1),'--','Linewidth',1);

legend('Early Parkinson','Normal','P(x)');
title('Density function estimation of SBR of the Right Caudate');
xlabel('SBR of the Right Caudate');
hold on;
plot(densites.SBR_pd(:,1),densites.D_pd(:,1),'+r',densites.SBR_no(:,1),densites.D_no(:,1),'+g');
hold off;

%% Labelisation

for i = 1:length_test

    P_carac(:,:,i) = posteriori(data_test(i,:), densites, P_priori);
    P(i,:) = prod(P_carac(:,:,i),2); % Ici je fais la multiplication des probas, peut être faire l'addition ou choix par vote.
    [~, ind] = max(P(i,:)); 
    label_out(i) = ind(1)-1;
    mat_conf(labels_test(i)+1, label_out(i)+1) = mat_conf(labels_test(i)+1, label_out(i)+1) + 1;
    
end
   
label_out = logical(label_out);
end

function Pcx = posteriori(x, densites, P_priori)
    
Pcx = zeros(2,size(densites.D_no,2));
n = size(densites.D_no,1);
    
    for k=1:size(densites.D_no,2)
        
        [~,Id_no] = min(abs((ones(n,1)*x(k)-densites.SBR_no(:,k)))); % Recuperation des points les plus proches car calcul numérique !
        [~,Id_pd] = min(abs((ones(n,1)*x(k)-densites.SBR_pd(:,k))));
        [~,Id_P] = min(abs((ones(n,1)*x(k)-densites.SBR_P_x(:,k))));
        
        Pcx(1,k) = densites.D_no(Id_no,k)*(1-P_priori)/densites.P_x(Id_P,k); 
        Pcx(2,k) = densites.D_pd(Id_pd,k)*P_priori/densites.P_x(Id_P,k);
    
    end
    
end