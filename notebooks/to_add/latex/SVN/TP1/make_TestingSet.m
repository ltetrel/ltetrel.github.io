clear all
path = 'Database/Test/';
nb_ex_class = 100;
data_test = zeros(nb_ex_class*10,100);
labels = zeros(1,nb_ex_class*10);
for i = 1:10
tic
fid = fopen([path 'listing_' num2str(i-1) '.txt'],'r');
for j = 1:nb_ex_class
tline = fgetl(fid);
image = imread([path num2str(i-1) '/' tline]);
data_test (nb_ex_class*(i-1)+j,:) = retine(image,10,10);
% data_test(nb_ex_class*(i-1)+j,:)= profil(image,5,5);
labels(nb_ex_class*(i-1)+j) = i;
end
fclose(fid);
toc
end
%%% on charge les vecteurs propres et la moyenne calcules avec la base d’apprentissage %%%
load acp_retine_10x10
%%% on projecte sur les n axes principaux %%%
n = 52;
data_test = (data_test-ones(nb_ex_class*10,1)*moyenne)*vec_p(:,1:n);
%%% on sauvegarde les nouvelles données %%%
save retine_10x10_acp_52_test data_test labels