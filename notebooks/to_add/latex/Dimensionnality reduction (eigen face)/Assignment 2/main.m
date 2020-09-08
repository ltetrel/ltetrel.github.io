function main

%% Read files

% dbpath = './Color FERET Database/';
% database_create(dbpath);

% listing = dir('./');
% listing = char(listing.name);
% test = uint8(listing);
% I = false(size(test,1),1);
% n = uint8('.fig');
% for i=1:size(test,2)-3 
% I = (I|test(:, i) == n(1) & test(:, i+1) == n(2)  & test(:, i+2) == n(3)  & test(:, i+3) == n(4)); 
% end
% I = find(I);
% for i =1:16
%     tmp = uint8(listing(I(i),:)); 
%     tmp=char(tmp(tmp~=32));
%     openfig(tmp);
%     tmp = [tmp(1:end-3) 'pdf'];
%     saveas(gca,tmp);
%     system(['pdfcrop' ' ' tmp ' ' tmp]);
%     close all;
% end

%% 1.1 Principal component analysis

% Image size : 384x256
% load('Data_im_single','data_train');
% 
% [eigen_faces,V,v,weights_train] = eigen_face_gen(data_train);
% 
% figure;
% title('Bigest ten eigen faces');
% for i=1:10
%     subplot(2,5,i);
% % We multiply by 255 because we want to see te behaviour for maximum
% % variance intensity (255 because the resolution is 255 max with ppm)
%     imshow((vec2mat(eigen_faces(i,:),384)')*255);
%     xlabel(['u' num2str(i)])
% end
% 
% figure;
% semilogx(1:length(v),v/sum(v));
% xlabel('Eigen values'); 
% ylabel('Importance (%)');
% 
% save('Eigen_results','V','v','eigen_faces','weights_train');

%% 1.2 Reconstruction

% load('Eigen_results','eigen_faces','weights_train');
% load('Data_im_single','data_train');
% n_eigen = [1 2 5 10 20 50 100 size(eigen_faces,1)];
% choice_im = randi(size(eigen_faces,1),1,3);
% rec_error = zeros(1,length(n_eigen));
% 
% figure;
% subplot(6,3,1);
% imshow(vec2mat(data_train(choice_im(1),:),384)');
% subplot(6,3,2);
% imshow(vec2mat(data_train(choice_im(2),:),384)');
% subplot(6,3,3);
% imshow(vec2mat(data_train(choice_im(3),:),384)');
% j=1;
% 
% for i = 1:length(n_eigen)
%     ii = n_eigen(i);
%     [data_test_rec,rec_error(i)] = rec_face(data_train, eigen_faces, weights_train, ii);
%     if sum(ii == [1 5 10 20 50])>0
%         subplot(6,3,3*j+1);
%         imshow(vec2mat(data_test_rec(choice_im(1),:),384)');
%         xlabel([num2str(ii) ' eigen vector(s)']);
%         subplot(6,3,3*j+2);
%         imshow(vec2mat(data_test_rec(choice_im(2),:),384)');
%         xlabel([num2str(ii) ' eigen vector(s)']);
%         subplot(6,3,3*j+3);
%         imshow(vec2mat(data_test_rec(choice_im(3),:),384)');
%         xlabel([num2str(ii) ' eigen vector(s)']);
%         j=j+1;
%     end
% end
% 
% figure;
% plot(n_eigen,rec_error,'-+');
% ylabel('RMS reconstruction error (float pixel)');
% xlabel('Number of eigen values');

%% 1.3 Classification and recognition

% load('Eigen_results','eigen_faces','weights_train');
% load('Data_im_single');
% n_eigen = [1 2 5 10 20 50 100 size(eigen_faces,1)];
% dist_eucli = zeros(size(data_train,1),size(data_test,1));
% true_detect = false(length(n_eigen),size(data_test,1));
% label = zeros(length(n_eigen),size(data_test,1));
% 
% m = mean(data_train);
% A = data_test- ones(size(data_test,1),1)*m;
% weights_test = A*eigen_faces';
% 
% for i = 1:length(n_eigen)
%     ii = n_eigen(i);
%     label(i,:) = eig_classify(weights_train,weights_test,ii,true);
%     for j=1:length(label(i,:))
%         true_detect(i,j) = sum(name_test(j,1:5) == name_train(label(i,j),(1:5))) == 5;
%     end
% end
% 
% choice_im(1,:) = datasample(find(true_detect(6,:)),3);
% choice_im(2,:) = datasample(find(~true_detect(6,:)),3);
% error_rate = sum(~true_detect,2)/size(true_detect,2);
% 
% figure;
% semilogx(n_eigen,error_rate,'+-');
% xlabel('Number of eigen values');
% ylabel('Error rate (%)');
% 
% figure;
% %Correctly classified
% subplot(4,3,1);
% imshow(vec2mat(data_test(choice_im(1,1),:),384)');
% ylabel('Correctly classified');
% xlabel(name_test(choice_im(1,1),:),'Interpreter','none');
% subplot(4,3,2);
% imshow(vec2mat(data_test(choice_im(1,2),:),384)');
% ylabel('Correctly classified');
% xlabel(name_test(choice_im(1,2),:),'Interpreter','none');
% subplot(4,3,3);
% imshow(vec2mat(data_test(choice_im(1,3),:),384)');
% ylabel('Correctly classified');
% xlabel(name_test(choice_im(1,3),:),'Interpreter','none');
% subplot(4,3,4);
% imshow(vec2mat(data_train(label(6,choice_im(1,1)),:),384)');
% xlabel(name_train(label(6,choice_im(1,1)),:),'Interpreter','none');
% subplot(4,3,5);
% imshow(vec2mat(data_train(label(6,choice_im(1,2)),:),384)');
% xlabel(name_train(label(6,choice_im(1,2)),:),'Interpreter','none');
% subplot(4,3,6);
% imshow(vec2mat(data_train(label(6,choice_im(1,3)),:),384)');
% xlabel(name_train(label(6,choice_im(1,3)),:),'Interpreter','none');
% 
% % Misclassified
% subplot(4,3,7);
% imshow(vec2mat(data_test(choice_im(2,1),:),384)');
% ylabel('Misclassified');
% xlabel(name_test(choice_im(2,1),:),'Interpreter','none');
% subplot(4,3,8);
% imshow(vec2mat(data_test(choice_im(2,2),:),384)');
% ylabel('Misclassified');
% xlabel(name_test(choice_im(2,2),:),'Interpreter','none');
% subplot(4,3,9);
% imshow(vec2mat(data_test(choice_im(2,3),:),384)');
% ylabel('Misclassified');
% xlabel(name_test(choice_im(2,3),:),'Interpreter','none');
% subplot(4,3,10);
% imshow(vec2mat(data_train(label(6,choice_im(2,1)),:),384)');
% xlabel(name_train(label(6,choice_im(2,1)),:),'Interpreter','none');
% subplot(4,3,11);
% imshow(vec2mat(data_train(label(6,choice_im(2,2)),:),384)');
% xlabel(name_train(label(6,choice_im(2,2)),:),'Interpreter','none');
% subplot(4,3,12);
% imshow(vec2mat(data_train(label(6,choice_im(2,3)),:),384)');
% xlabel(name_train(label(6,choice_im(2,3)),:),'Interpreter','none');

%% 1.4 Probabilistic Face Recognition

% Assuming that the weight I (eigen space) are our data (and not the pixel
% grey value d) p(l=k|I)=p(I|l=k).p(l=k)/p(I)

% load('Eigen_results','eigen_faces','weights_train');
% load('Data_im_single');
% 
% m = mean(data_train);
% A = data_test- ones(size(data_test,1),1)*m;
% weights_test = A*eigen_faces';
% n_eigen = 5;
% true_detect = false(length(n_eigen),size(data_test,1));
% label = zeros(length(n_eigen),size(data_test,1));
% 
% listing = dir('./Color FERET Database');
% for i=3:54
%     name_t(i-2,:)=char(listing(i).name);
% end
% 
% label = eig_classify(weights_train,weights_test,n_eigen,false);
% for j=1:length(label)
%     true_detect(j) = sum(name_test(j,1:5) == name_t(label(j),(1:5))) == 5;
% end
% sum(true_detect)/length(true_detect)

%% 2.1 Eigenfaces for Pose Estimation

% load('Data_im','data','name');
% database_create_pose(data,name);

% load('Data_im_pose');
% name_pose = [-90 -50 -15 0 15 50 90];
% 
% for i=1:7
%     m{i} = mean(data_train{i});
%     [eigen_faces{i},V{i},v{i},weights_train{i}] = eigen_face_gen(data_train{i});
%     figure;
%     title(['Bigest ten eigen faces for pose ' num2str(name_pose(i)) '°']);
%     for j=1:10
%         subplot(2,5,j);
%         imshow((vec2mat(eigen_faces{i}(j,:),384)')*255);
%         xlabel(['u' num2str(j)])
%     end
% end
% 
% figure;
% for i=1:7
%     subplot(4,2,i);
%     imshow(vec2mat(m{i},384)');
% 	xlabel(['Mean image for pose ' num2str(name_pose(i)) '°']);
% end
% 
% save('Eigen_results_pose','eigen_faces','weights_train','V','v');

%% 2.2 Pose classification

% load('Data_im_pose');
% load('Eigen_results_pose');
% 
% n = {   {'pl'};...
%         {'hl','bi','bh','ra'};...
%         {'bg','ql','bf','rb'};...
%         {'fa','fb','ba','bj','bk'};...
%         {'be','qr','bd','rc'};...
%         {'bc','rd','bb','hr'};...
%         {'pr','re'} };
% name_u = uint8(name_test);
% true_label = zeros(size(name_test,1),1);
% for i=1:7
%     for j=1:length(n{i})
%         ind = sum(name_u(:,14:15) == uint8(ones(size(name_test,1),1)*double(uint8(char(n{i}{j})))),2)==2;
%         true_label(ind) = i;
%     end
% end
% 
% for i = 1:7
%     m = mean(data_train{i});
%     A = data_test - ones(size(data_test,1),1)*m;
%     weights_test{i} = A*eigen_faces{i}';
%     n_eigen(i,:) = [1 2 5 10 20 30 50 90 100 size(eigen_faces{i},1)];
% end
% 
% for i=1:10
%     mat_conf{i} = zeros(7);
% end
%     
% for i = 1:length(n_eigen)
%     for j = 1:7
%         [label_image_dist{i}(j,:),label_dist(j,:)] = eig_classify(weights_train{j},weights_test{j},n_eigen(j,i),true);  
%     end
%     [~,label(i,:)] = min(label_dist);
%     
%     for j=1:size(data_test,1)
%         mat_conf{i}(true_label(j),label(i,j)) = mat_conf{i}(true_label(j),label(i,j)) + 1;
%     end
% end
% 
% % for i=1:size(n_eigen,2)
% %     error_rate(i) = 1 - sum(diag(mat_conf{i}))/sum(sum(mat_conf{i}));
% % end
% % figure;
% % semilogx(n_eigen,error_rate,'+-');
% % xlabel('Number of eigen values');
% % ylabel('Error rate (%)');
% 
% true_detect = (label == ones(10,1)*(true_label'));
% choice_im(1,:) = datasample(find(true_detect(7,:)),3);
% choice_im(2,:) = datasample(find(~true_detect(7,:)),3);
% 
% choice_im_tmp = choice_im';
% label_num = [diag(label_image_dist{7}(label(7,choice_im_tmp),choice_im_tmp))'; label(7,choice_im_tmp)];
% 
% figure;
% %Correctly classified
% subplot(4,3,1);
% imshow(vec2mat(data_test(choice_im(1,1),:),384)');
% ylabel('Correctly classified');
% xlabel(name_test(choice_im(1,1),:),'Interpreter','none');
% subplot(4,3,2);
% imshow(vec2mat(data_test(choice_im(1,2),:),384)');
% ylabel('Correctly classified');
% xlabel(name_test(choice_im(1,2),:),'Interpreter','none');
% subplot(4,3,3);
% imshow(vec2mat(data_test(choice_im(1,3),:),384)');
% ylabel('Correctly classified');
% xlabel(name_test(choice_im(1,3),:),'Interpreter','none');
% subplot(4,3,4);
% imshow(vec2mat(data_train{label_num(2,1)}(label_num(1,1),:),384)');
% xlabel(name_train{label_num(2,1)}(label_num(1,1),:),'Interpreter','none');
% subplot(4,3,5);
% imshow(vec2mat(data_train{label_num(2,2)}(label_num(1,2),:),384)');
% xlabel(name_train{label_num(2,2)}(label_num(1,2),:),'Interpreter','none');
% subplot(4,3,6);
% imshow(vec2mat(data_train{label_num(2,3)}(label_num(1,3),:),384)');
% xlabel(name_train{label_num(2,3)}(label_num(1,3),:),'Interpreter','none');
% 
% % Misclassified
% subplot(4,3,7);
% imshow(vec2mat(data_test(choice_im(2,1),:),384)');
% ylabel('Misclassified');
% xlabel(name_test(choice_im(2,1),:),'Interpreter','none');
% subplot(4,3,8);
% imshow(vec2mat(data_test(choice_im(2,2),:),384)');
% ylabel('Misclassified');
% xlabel(name_test(choice_im(2,2),:),'Interpreter','none');
% subplot(4,3,9);
% imshow(vec2mat(data_test(choice_im(2,3),:),384)');
% ylabel('Misclassified');
% xlabel(name_test(choice_im(2,3),:),'Interpreter','none');
% subplot(4,3,10);
% imshow(vec2mat(data_train{label_num(2,4)}(label_num(1,4),:),384)');
% xlabel(name_train{label_num(2,4)}(label_num(1,4),:),'Interpreter','none');
% subplot(4,3,11);
% imshow(vec2mat(data_train{label_num(2,5)}(label_num(1,5),:),384)');
% xlabel(name_train{label_num(2,5)}(label_num(1,5),:),'Interpreter','none');
% subplot(4,3,12);
% imshow(vec2mat(data_train{label_num(2,6)}(label_num(1,6),:),384)');
% xlabel(name_train{label_num(2,6)}(label_num(1,6),:),'Interpreter','none');

%% 2.3 Probabilistic Pose Classification

load('Data_im_pose');
load('Eigen_results_pose');

n = {   {'pl'};...
        {'hl','bi','bh','ra'};...
        {'bg','ql','bf','rb'};...
        {'fa','fb','ba','bj','bk'};...
        {'be','qr','bd','rc'};...
        {'bc','rd','bb','hr'};...
        {'pr','re'} };
name_u = uint8(name_test);
true_label = zeros(size(name_test,1),1);
for i=1:7
    for j=1:length(n{i})
        ind = sum(name_u(:,14:15) == uint8(ones(size(name_test,1),1)*double(uint8(char(n{i}{j})))),2)==2;
        true_label(ind) = i;
    end
end

for i = 1:7
    m = mean(data_train{i});
    A = data_test - ones(size(data_test,1),1)*m;
    weights_test{i} = A*eigen_faces{i}';
end
    
for j = 1:7
    [~,label_dist(j,:)] = eig_classify(weights_train{j},weights_test{j}, 1,false);  
end
[~,label] = min(label_dist);

ind = [79 83 24 165 64 106];
figure;
for i=1:6
    subplot(2,3,i);
    stem(1:7,label_dist(:,ind(i)));
    xlabel('Pose label');
    ylabel(['Probability of ' name_test(ind(i),:)],'Interpreter','none');
end

mat_conf = zeros(7);

for j=1:size(data_test,1)
    mat_conf(true_label(j),label(j)) = mat_conf(true_label(j),label(j)) + 1;
end

end


