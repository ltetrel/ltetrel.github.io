clear all

%%% on charge les données %%%
load retine_10x10_learning
[nb_ex,nb_feat] = size(database);

%%% on charge les vecteurs propres et la moyenne %%%
load acp_retine_10x10

%%% on projecte sur les n axes principaux %%%
n = 52;
database = (database-ones(nb_ex,1)*moyenne)*vec_p(:,1:n);

%%% on sauvegarde les nouvelles données %%%
save retine_10x10_acp_52_learning database labels