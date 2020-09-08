function database_create(dbpath)

listing = dir(dbpath);
data_train = uint8(zeros((length(listing)-2)*28,98304));
name_train = cell((length(listing)-2)*28,1);
data_test = uint8(zeros((length(listing)-2)*4,98304));
name_test = cell((length(listing)-2)*4,1);
n = 1;
m = 1;

for i=1:length(listing)-2
    dbpath_image = [listing(i+2).name '/'];
    listing_im = dir([dbpath dbpath_image]);
    choice = randperm(length(listing_im)-2,32);
    for j=1:28
        im = imread([dbpath dbpath_image listing_im(choice(j)+2).name]);
        im = rgb2gray(im);
        im = imresize(im,0.5);
        data_train(n,:) = im(:)';
        name_train{n} = listing_im(choice(j)+2).name;
        n=n+1;
    end
    for j= 29:32
        im = imread([dbpath dbpath_image listing_im(choice(j)+2).name]);
        im = rgb2gray(im);
        im = imresize(im,0.5);
        data_test(m,:) = im(:)';
        name_test{m} = listing_im(choice(j)+2).name;
        m=m+1;
    end
end

save('Data_im','data_train','name_train','data_test','name_test');

end