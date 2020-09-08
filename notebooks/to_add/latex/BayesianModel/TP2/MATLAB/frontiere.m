function frontiere(database, label)

chiffre = 0;
chiffre = chiffre+1;
Cf = sum((label == chiffre))/length(label);
K_param = 10;
C = 0.1;
param=['-v 0 -t 2 -g ' num2str(K_param) ' -c ' num2str(C) ' -j ' num2str(Cf)];

label = label';
for i=1:length(label)
   
    if label(i) == chiffre
        label(i) = 1;
    else
        label(i) = -1;
    end
    
end

clc;
disp('Calcul et trace de la figure...');
model = mexsvmlearn(database(:,1:2), label, param);
index_vs = find(model.a~=0);
VS = database(index_vs,1:2);
x1plot = linspace(min(database(:,1)), max(database(:,1)), 100)';
x2plot = linspace(min(database(:,2)), max(database(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);

for i = 1:100
    tstart = tic;
    
    for j = 1:100
        [~,vals(i,j)] = mexsvmclassify([x1plot(i), x2plot(j)], label(1), model);
        vals(i,j) = vals(i,j)>=0;
    end
    
    temps=toc(tstart)*(100-i);
    disp([num2str(i/100*100) '%. Estimation du temps restant : ' num2str(temps) 's']);
end

% Plot des frontières

figure;
plot(database((label==1),1), database((label==1),2), 'rx', database(~(label==1),1), database(~(label==1),2), 'gx');
legend(['Chiffre ' num2str(chiffre-1)],'Autres chiffres');
xlabel('1e carac');
ylabel('2e carac');
hold on
contour(X1, X2, vals, [0 0], 'Color', 'b','linewidth', 5);
hold on
plot(VS(:,1),VS(:,2),'ob', 'linewidth', 0.1);
alpha(0.1);
hold off;

title(['Frontière de décision entre le chiffre ' num2str(chiffre-1) ' et les autres pour un Kernel Gaussien']); 

end

