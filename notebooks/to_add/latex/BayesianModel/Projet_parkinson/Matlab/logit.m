function [label_out, new_mat_conf, stats] = logit(database,label)
%function [label_out,new_mat_conf, param] = logit(database,label)

n = 75; %Nombre de points pour le tracé du likelihood
k = 1; %Choix de la caractéristique
nb_boxes = 50; % On divise notre base de donnée en n blocs
plot_x = {
    ['SBR of the Right Caudate'],...
    ['SBR of the Left Caudate'],...
    ['SBR of the Right Putamen'],...
    ['SBR of the Left Putamen']
    };
length_boxes = floor(length(label)/nb_boxes);


% for k = 1:size(database,2)
%     
%     m = min(database(:,k));
%     M = max(database(:,k));
%     larg = (M-m)/n;
%     p = zeros(1,n);
% 
%     for i=1:n
% 
%         p(i) = sum(label((database(:,k)>=(i-1)*larg+m & database(:,k)<i*larg+m)))/length(label((database(:,k)>=(i-1)*larg+m & database(:,k)<i*larg+m)));
% 
%     end
% 
%     figure;
%     plottools;
%     plot(m+larg/2:larg:M,p,'r.','markersize',15);
%     ylabel('Probability that parkinson occur'); xlabel(plot_x(k));
% 
% end

indices = randperm(length(label));
old_mat_conf = zeros(length(unique(label)));
new_mat_conf = zeros(length(unique(label)));

new_database = zeros(length(database),3);
new_database(:,1) = database(:,1).*database(:,2);
new_database(:,2:3) = database(:,3:4);

for i = 1:nb_boxes
     
    if i==nb_boxes
        box = indices(1+(i-1)*length_boxes:end);
    else
        box = indices(1+(i-1)*length_boxes:i*length_boxes); 
    end
%     
%     [B,dev,stats] = mnrfit(database(box(1:end-1),:), double(label(box(1:end-1))+1));
%     P = 1-1/(1+exp(-B(1)-B(2)*database(box(end),1)-B(3)*database(box(end),2)-B(4)*database(box(end),3)-B(5)*database(box(end),4)));
% 
%     if P > 0.5    
%         lab = true;    
%     else
%         lab = false;
%     end
% 
%     old_mat_conf(double(label(box(end))+1), double(lab+1)) = old_mat_conf(double(label(box(end))+1), double(lab+1)) + 1;
    
    [B,dev,stats] = mnrfit(new_database(box(1:end-1),:), double(label(box(1:end-1))+1)); % Pour matlab le logit est forcément croissant,  
% si (comme avec parkinson) la probabilité augmente avec le marqueur qui decroit alors il faut calculer l'inverse de la proba (car le logit matlab croit alors qu'il est censé decroitre !)
    P = 1-1/(1+exp(-B(1)-B(2)*new_database(box(end),1)-B(3)*new_database(box(end),2)-B(4)*new_database(box(end),3))); 

    if P > 0.5   
        lab = true;    
    else
        lab = false;
    end
    
    label_out(i) = lab;
    new_mat_conf(double(label(box(end))+1), double(lab+1)) = new_mat_conf(double(label(box(end))+1), double(lab+1)) + 1;
    
end

[B,dev,stats] = mnrfit(new_database, double(~label+1));

% ['Erreur avec les anciennes caractéristiques : ' num2str(100*sum(diag(rot90(old_mat_conf)))/sum(sum(old_mat_conf))) ' %'],
% old_mat_conf,
['Erreur avec les nouvelles caractéristiques : ' num2str(100*sum(diag(rot90(new_mat_conf)))/sum(sum(new_mat_conf))) ' %'],
['Taux d erreur Normal : ' num2str(100*(new_mat_conf(1,2))/sum((new_mat_conf(1,:)))) ' %'],
['Taux d erreur Parkinson : ' num2str(100*(new_mat_conf(2,1))/sum((new_mat_conf(2,:)))) ' %'],
% new_mat_conf,

end