function marge(SBR, label, model)
%Demo sur l'utilisation de la fonction mexsvmlearn.dll
%conçu par Tom Briggs, interface matlab et SVMlight
%SVMlight a été developpé par  Thorsten Joachims en C.
fprintf('*** Formation de la base de donnée *** \n');
X=SBR;
Y=label;
fprintf('*** Apprentissage du SVM *** \n');

fprintf('*** Affichage des valeurs de alpha avec signe de Yi *** \n');
Yalpha=model.a

fprintf('*** Affichage de la valeur du biais  *** \n');
biais=-model.b

fprintf('*** Affichage de l indice des vecteurs de support *** \n');
index_vs = find(Yalpha~=0)

fprintf('*** Dessin de la frontière de décision ***\n');
%Calcul du noyau
% gauss = exp(-model.kernel_parm.rbf_gamma*((SBR(:,1) - SBR(:,2)).^2 ));
% %Récupérer les vecteurs de support 
% VS = SBR(index_vs,1:2);
% VS(:,3) = gauss(index_vs);
% %Afficher les données d'apprentissage
% plot3(SBR(label,1), SBR(label,2), gauss(label), 'rx', SBR(~label,1), SBR(~label,2), gauss(~label), 'gx');
% xlabel('SBR of the Right Putamen');
% ylabel('SBR of the Left Putamen');
% zlabel('Gaussian Kernel');
% legend('Normal','Early Parkinson');
% %Encercler en bleu les vecteurs de support
% hold on
% plot3(VS(:,1),VS(:,2),VS(:,3),'ob', 'linewidth', 2);
%Paramètres de la frontière de décision
% a1=Yalpha(index_vs)'*SBR(index_vs,1)
% hold on


gamma     = model.kernel_parm.rbf_gamma;
b         = biais;
points_x1 = linspace(min(SBR(:,1)), max(SBR(:,1)), 100);
points_x2 = linspace(min(SBR(:,2)), max(SBR(:,2)), 100);
[X1, X2]  = meshgrid(points_x1, points_x2);

% Initialize f
f = ones(length(points_x1), length(points_x2));

% Iter. all SVs
for i=1:length(index_vs)
    Yalpha_i = Yalpha(index_vs(i));
    sv_i    = [SBR(index_vs(i),1); SBR(index_vs(i),2);];
    for j=1:length(points_x1)
        for k=1:length(points_x2)
            x = [points_x1(j);points_x2(k)];
            f(j,k) = f(j,k) + Yalpha_i*kernel_func(gamma, x, sv_i);
        end
    end    
end

surf(X1,X2,f);
shading interp;
lighting phong;
alpha(.6)


end

function k = kernel_func(gamma, x, x_i)
    k = exp(-gamma*norm(x - x_i)^2);
end