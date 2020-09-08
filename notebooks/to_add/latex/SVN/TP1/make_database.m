clear all

path = 'Database/Learning/';

nb_ex_class = 100;
database = zeros(nb_ex_class*10,100);
labels = zeros(1,nb_ex_class*10);

for i = 1:10
    tic
    fid = fopen([path 'listing_' num2str(i-1) '.txt'],'r');
    for j = 1:nb_ex_class
        tline = fgetl(fid);
        image = imread([path num2str(i-1) '/' tline]);
        database(nb_ex_class*(i-1)+j,:) = retine(image,10,10);
        %database(nb_ex_class*(i-1)+j,:) = profil(image,10,10);
        labels(nb_ex_class*(i-1)+j) = i;
    end
    fclose(fid);
    toc
end

save retine_10x10_learning database labels
%save profil_10x10_learning database labels
