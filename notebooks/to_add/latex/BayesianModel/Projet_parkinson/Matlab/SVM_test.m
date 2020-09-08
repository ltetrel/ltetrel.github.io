function [label_out, mat_conf, model] = SVM_test(database, label)
% function [label_out, mat_conf, model] = SVM_test(database, label)

%% Variables

choix_noyau = 0; %0 : lineaire      1 : Polynomial      2 : Gaussien        3 : Sigmoïd

C = 1e4;

switch choix_noyau
    case 0
    K_param = [1]; % Lineaire
    case 1
    K_param = [4; 1e2; 1e-2]; % Polynomial
    case 2
    K_param =0.08547; % Gaussien
    case 3
    K_param = [1e1; 5e-1]; % Sigmoid  
end

nb_boxes = 50; % On divise notre base de donnée en n blocs
length_boxes = floor(length(label)/nb_boxes);
mat_conf = zeros(length(unique(label)));

indices = randperm(length(label));
Cf = sum(~label)/sum(label);
Cf = 1-Cf;

if size(label,1) == 1             % Pour la fonction SVM
    label = double(label)';
end

for i=1:length(label)  
   if label(i) == 0  
      label(i) = -1;   
   end
end

switch choix_noyau
    case 0
    param=['-v 0 -t 0 -c ' num2str(C) ' -j ' num2str(Cf)]; % Lineaire
    case 1
    param=['-v 0 -t 1 -d ' num2str(K_param(1)) ' -s ' num2str(K_param(2)) ' -r ' num2str(K_param(3)) ' -c ' num2str(C) ' -j ' num2str(Cf)]; % Polynomial
    case 2
    param=['-v 0 -t 2 -g ' num2str(K_param) ' -c ' num2str(C) ' -j ' num2str(Cf)];  %Gaussien
    case 3
    param=['-v 3 -t 3 -s ' num2str(K_param(1)) ' -r ' num2str(K_param(2)) ' -c ' num2str(C) ' -j ' num2str(Cf)];  %Sigmoid 
end

%% Processus

tic;
for i = 1:nb_boxes
    tstart = tic;
    
    if i==nb_boxes
        box = indices(1+(i-1)*length_boxes:end);
    else
        box = indices(1+(i-1)*length_boxes:i*length_boxes); 
    end
    
    model = mexsvmlearn(database(box(1:end-1),:), label(box(1:end-1)), param); 
    [~,label_out(i,1)] = mexsvmclassify(database(box(end),:), label(box(end)), model);
    mat_conf((label(box(end))>=0)+1, (label_out(i,1)>=0)+1) = mat_conf((label(box(end))>=0)+1, (label_out(i,1)>=0)+1) + 1; %Bidouille pour transformer -1 en 1 et +1 en 2
    label_out(i,2) = box(end);
    label_out(i,1) = label_out(i,1)>=0;
    
    temps=toc(tstart)*(nb_boxes-i);
    disp([num2str(i/nb_boxes*100) '%. Estimation du temps restant : ' num2str(temps) 's']);
end


%% Pour tracé des frontières

% clc;
% disp('Calcul et trace de la figure...');
% model = mexsvmlearn(database(:,1:2), label, param);
% index_vs = find(model.a~=0);
% VS = database(index_vs,1:2);
% x1plot = linspace(min(database(:,1)), max(database(:,1)), 100)';
% x2plot = linspace(min(database(:,2)), max(database(:,2)), 100)';
% [X1, X2] = meshgrid(x1plot, x2plot);
% 
% for i = 1:100
%     tstart = tic;
%     
%     for j = 1:100
%         [~,vals(i,j)] = mexsvmclassify([x1plot(i), x2plot(j)], label(1), model);
%         vals(i,j) = vals(i,j)>=0;
%     end
%     
%     temps=toc(tstart)*(100-i);
%     disp([num2str(i/100*100) '%. Estimation du temps restant : ' num2str(temps) 's']);
% end
% 
% % Plot des frontières
% 
% figure;
% plot(database((label>=0),1), database((label>=0),2), 'rx', database(~(label>=0),1), database(~(label>=0),2), 'gx');
% legend('Early Parkinson','Normal');
% xlabel('SBR of the Right Putamen');
% ylabel('SBR of the Left Putamen');
% hold on
% contour(X1, X2, vals, [0 0], 'Color', 'b');
% hold on
% plot(VS(:,1),VS(:,2),'ob', 'linewidth', 0.1);
% alpha(0.1);
% hold off;
% 
% switch choix_noyau
%     case 0
%     title(['Decision boundary for Linear Kernel with C = ' num2str(C)]); 
%     case 1
%     title(['Decision boundary for Polynomial Kernel with C = ' num2str(C) ', d = ' num2str(K_param(1)) ', s = ' num2str(K_param(2)) ' and c = ' num2str(K_param(3))]); 
%     case 2
%     title(['Decision boundary for Gaussian Kernel with C = ' num2str(C) ' and g = ' num2str(K_param)]); 
%     case 3
%     title(['Decision boundary for Sigmoid Kernel with C = ' num2str(C) ', s = ' num2str(K_param(1)) ' and c = ' num2str(K_param(2))]); 
% end
% 
% temps_ecoule = toc;
% disp(['Temps ecoule : ' num2str(temps_ecoule) 's']);
end
