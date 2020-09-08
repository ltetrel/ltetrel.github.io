function output = profil (image,nb_l,nb_c)

%% output = profil (image,nb_l,nb_c)

output = ones(1,2*( nb_l+nb_c));

%% on redimensionne l’image dans une matrice nb_l x nb_c


%% on compte le nombre de pixels horizontalement (de la gauche vers la droite et dans le sens contraire)



%% on compte le nombre de pixels verticalement (du bas vers le haut et dans le sens contraire)


%% on concatène les résultats dans output