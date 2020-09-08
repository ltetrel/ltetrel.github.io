clear all

%%% on charge les données %%%
load retine_10x10_learning
[nb_ex,nb_feat] = size(database);

%%% on calcule la moyenne %%%
moyenne = mean(database);

%%% on calcule la matrice de covariance %%%
S = cov(database);

%%% on calcule les vecteurs propres et les valeurs propres %%%
[vec_p, L] = eig(S);

%%% on trie les vecteurs propres selon les valeurs propres %%%
[val_p, ind] = sort(diag(L));
val_p = flipud(val_p);
ind = flipud(ind);
vec_p = vec_p(:,ind);

%%% on sauvegarde les vecteurs propres et la moyenne %%%
save acp_retine_10x10 vec_p moyenne

%%% on trace la variabilité exprimée en fonction du nombre de composantes %%%
figure;
plot(cumsum(val_p)/sum(val_p)*100,'-');

