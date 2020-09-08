function main

%% EM-GMM : To use if you want to test (just change the variable feat and nb_k)

% im_in = 'horse.jpg';
% im_in = imread(im_in);
% size_im = size(im_in);
% color = {[1; 0; 0], [0; 1; 0], [0; 0; 1], [0.4; 0.2; 0.2], [0.2; 0.4; 0.2], [0.2; 0.2; 0.4]};
% name_feat = {'grey', 'RGB','Lab'};
%
% feat = 1; % 1:grey; 2:rgb; 3:Lab
% nb_k = 4;
%
% M = feature_space(im_in,feat);
% M = double(M);
% [y,label,model,loglklh] = EM_GMM(M,nb_k,true);
% disp(['For feature ' name_feat{feat} ' and ' num2str(nb_k) ' gaussians component, a maximum log-likelihood : ' num2str(single(loglklh(end)))]);
% im_out = label_show(label, size_im, color);
%
% if size(M,2)==1
%     figure;
%     for k=1:nb_k
%         [f,xi] = ksdensity(M(label==k));
%         plot(xi,f,'Linewidth',2,'color',color{k});
%         hold on;
%         y = normpdf(1:255,model.means(k),sqrt(model.covariances(k)));
%         plot(1:255,y,'-','color',color{k});
%         hold on;
%     end
%     xlabel('Grey');
% elseif size(M,2)==2
%     figure;
%     z = cell(2,1);
%     for k=1:nb_k
%         for i=1:size(M,1)
%             z{2}(M(i,1)+1,M(i,2)+1) = y{2}(i,k);
%         end
%         surf(1:max(M(:,2))+1,1:max(M(:,1))+1,z{2},'FaceColor',color{k}');
%         hold on;
%     end
%     xlabel('a');
%     ylabel('b');
% elseif size(M,2)==3
%     figure;
%     for k=1:nb_k
%         plot_ellipsoid(model.covariances(:,:,k),model.means(k,:),color{k});
%         hold on;
%     end
%     xlabel('Red');
%     ylabel('Green');
%     zlabel('Blue');
% end

%% Part I : Influence of parameters and dataspace in EM-GMM

% im_in = 'horse.jpg';
% im_in = imread(im_in);
% size_im = size(im_in);
color = {[1; 0; 0], [0; 1; 0], [0; 0; 1], [0.4; 0.2; 0.2], [0.2; 0.4; 0.2], [0.2; 0.2; 0.4]};
% name_feat = {'grey', 'RGB','Lab'};
%
% figure;
% n=1;
% for feat=1:3
%     for nb_k=[2 4 6]
%         M = feature_space(im_in,feat);
%         M = double(M);
%         [y,label,model,loglklh,~,~,time] = EM_GMM(M,nb_k,true);
%         disp(['For feature ' name_feat{feat} ' and ' num2str(nb_k) ' gaussians component, a maximum log-likelihood : ' num2str(loglklh(end))]);
%         subplot(3,3,n)
%         im_out = label_show(label, size_im, color);
%         xlabel(['Processing of one iteration in ' num2str(single(time)) ' s.']);
%         n=n+1;
%     end
% end

%% Part II a. : 5-fold cross validation

im_in = 'horse.jpg';
im_in = imread(im_in);
size_im = size(im_in);
name_feat = {'grey', 'RGB','Lab'};

param = [2:8];
nb_boxes = 5;
figure;

% for feat = 1:3
%     M = feature_space(im_in,feat);
%     M = double(M);
%     scores = zeros(length(param),1);
%     length_boxes = floor(size(M,1)/nb_boxes);
%     indices = randperm(size(M,1));
%     box_test = indices(1+(nb_boxes-1)*length_boxes:end);
%
%     for i=param
%         modelScores = zeros(nb_boxes-1,1);
%         for j=1:nb_boxes-1
%             box_train = indices(1+(j-1)*length_boxes:j*length_boxes);
%             [~,~,model] = EM_GMM(M(box_train,:),i,true);
%             [~,~,~,lklh_gmm] = expectation(M(box_test,:),model);
%             modelScores(j) = mean(log(lklh_gmm));
%             modelScoresGmm(:,j) = log(lklh_gmm);
%             disp(['Computing box no ' num2str(j) '/' num2str(nb_boxes-1) ' with a mean log-likelihood : ' num2str(modelScores(j))]);
%         end
%         ScoresGmm(:,i) = mean(modelScoresGmm,2);
%         scores(i) = mean(modelScores);
%         disp(['For feature and ' num2str(i) ' gaussians component, a mean log-likelihood : ' num2str(scores(i))]);
%     end
%
%     scores(scores==0) = [];
%     [~,mn] = min(scores);
%     [~,mx] = max(scores);
%     figure;
%     hist([ScoresGmm(:,mn+1) ScoresGmm(:,mx+1)],100);
%     plot(param,scores,'+','Linewidth',2,'MarkerSize',10,'color',color{feat}); 
%     hold on; 
%     plot(param,scores,'color',color{feat});
%     hold on;
% end
%% Part II b. : Bayesian information criterion

for feat = 1:3
    
    M = feature_space(im_in,feat);
    M = double(M);
    scoresBIC = zeros(length(param),1);
    
    for i=param
        disp(['Computing BIC criteria ' num2str(find(param==i)) '/' num2str(length(param)) ' for feature ' name_feat{feat} ' and ' num2str(i) ' gaussians component.']);
        [~,~,~,~,lklh_gmm] = EM_GMM(M,i,true);
        scoresBIC(i) = -2*(sum(log(lklh_gmm)))+i*log(size(M,1));
    end
    
    scoresBIC(scoresBIC==0) = [];
    plot(param,scoresBIC,'+','Linewidth',2,'MarkerSize',10,'color',color{feat}); 
    hold on; 
    plot(param,scoresBIC,'color',color{feat}); 
    hold on;
    
end
%% Part III : Segmentation evaluation

% % 1. Set the relative path to the dataset main directory
% % assume that the Weizmann dataset folder is in the same directory as this code
% DBpath = './Weizmann dataset';
%
% % 2. Set SysType to 'win' or 'unix', based on your OS
% SysType = 'unix';
% l=dir(DBpath);
% switch lower(SysType)
%     case 'win'
%         Sep='\';
%     case 'unix'
%         Sep='/';
%     otherwise
%         Sep='\';
% end;
% % Load image names
% Lpath = load(strcat(DBpath,Sep,'img_list.mat'));
% Lpath = Lpath.fls;
% % Set the num of clusters
% numClusters = 2;
%
% % 3. For each image, do
% for i=1:length(Lpath)
%
% %     Lpath(i).name = 'b20nature_landscapes129';
%     disp(['Processing of image ' Lpath(i).name]);
%     % load the image
%     imgPath = strcat(DBpath,Sep,Lpath(i).name,Sep,'src_color',Sep,Lpath(i).name,'.png');
%     fileName = Lpath(i).name;
%     I = imread(imgPath);
%     [nrows, ncols, nchannels] = size(I);
%     binaryMap = false(nrows,ncols);
%
%     M = feature_space(I,1);
%     M = double(M);
%     centers = (quantile(M,[0.25 0.75])+rand(1,numClusters))';
%     idx = kmeans(M,numClusters,'Start',centers);
%     binaryMap = vec2mat(idx==1,nrows)';
%     SegResultsSubPath = 'my_kmeans_intensity';
%     outputDir = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath);
%     mkdir(outputDir);
%     outputImagePath = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath,Sep,fileName,'.png');
%     imwrite(binaryMap, outputImagePath);
%
%     M = feature_space(I,2);
%     M = double(M);
%     centers = quantile(M,[0.25 0.75])+rand(numClusters,size(M,2));
%     idx = kmeans(M,numClusters,'Start',centers);
%     binaryMap = vec2mat(idx==1,nrows)';
%     SegResultsSubPath = 'my_kmeans_RGB';
%     outputDir = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath);
%     mkdir(outputDir);
%     outputImagePath = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath,Sep,fileName,'.png');
%     imwrite(binaryMap, outputImagePath);
%
%     M = feature_space(I,3);
%     M = double(M);
%     centers = quantile(M,[0.25 0.75])+rand(numClusters,size(M,2));
%     idx = kmeans(M,numClusters,'Start',centers);
%     SegResultsSubPath = 'my_kmeans_Lab';
%     binaryMap = vec2mat(idx==1,nrows)';
%     outputDir = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath);
%     mkdir(outputDir);
%     outputImagePath = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath,Sep,fileName,'.png');
%     imwrite(binaryMap, outputImagePath);
%
%
%     M = feature_space(I,1);
%     M = double(M);
%     [~,label] = EM_GMM(M,numClusters,false);
%     binaryMap = vec2mat(label==1,nrows)';
%     SegResultsSubPath = 'my_gmm_intensity';
%     outputDir = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath);
%     mkdir(outputDir);
%     outputImagePath = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath,Sep,fileName,'.png');
%     imwrite(binaryMap, outputImagePath);
%
%     M = feature_space(I,2);
%     M = double(M);
%     [~,label] = EM_GMM(M,numClusters,false);
%     binaryMap = vec2mat(label==1,nrows)';
%     SegResultsSubPath = 'my_gmm_RGB';
%     outputDir = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath);
%     mkdir(outputDir);
%     outputImagePath = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath,Sep,fileName,'.png');
%     imwrite(binaryMap, outputImagePath);
%
%     M = feature_space(I,3);
%     M = double(M);
%     [~,label] = EM_GMM(M,numClusters,false);
%     binaryMap = vec2mat(label==1,nrows)';
%     SegResultsSubPath = 'my_gmm_Lab';
%     outputDir = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath);
%     mkdir(outputDir);
%     outputImagePath = strcat(DBpath,Sep,fileName,Sep,SegResultsSubPath,Sep,fileName,'.png');
%     imwrite(binaryMap, outputImagePath);
%
% end

% Evaluate the performance scores
% K-means in intensity space
% scores_kmeansgray = ComputeFMeasure(DBpath, 'my_kmeans_intensity', SysType);
% scores_kmeansrgb = ComputeFMeasure(DBpath, 'my_kmeans_RGB', SysType);
% scores_kmeanslab = ComputeFMeasure(DBpath, 'my_kmeans_Lab', SysType);
%
% scores_gmmgray = ComputeFMeasure(DBpath, 'my_gmm_intensity', SysType);
% scores_gmmrgb = ComputeFMeasure(DBpath, 'my_gmm_RGB', SysType);
% scores_gmmlab = ComputeFMeasure(DBpath, 'my_gmm_Lab', SysType);
%
% scores_their_gmmlab = ComputeFMeasure(DBpath, 'EMlab', SysType);
% scores_their_gmmrgb = ComputeFMeasure(DBpath, 'EMrgb', SysType);

%% Statistical analysis

% load('scores');
% % scores_all {row : image; col: measure; method}
% % method {gmmgrey gmmglab gmmrgb kmeansgrey kmeanslab kmeansrgb}
%
% anova1(scores_Fmeas,{'GMM.g','GMM.lab','GMM.rgb','K-means.g','K-means.lab', 'K-means.rgb'});
%
% %ANOVA between GMM and K-means
% [~,t,~] = anova1([[scores_all(:,1,1); scores_all(:,1,2); scores_all(:,1,3)] [scores_all(:,1,4);scores_all(:,1,5);scores_all(:,1,6)]], {'GMM', 'K-means'});
% ylabel('F-measure');
%
% %ANOVA between Grey, Lab and rgb
% [~,t,~] = anova1([[scores_all(:,1,1); scores_all(:,1,4)] [scores_all(:,1,2);scores_all(:,1,5)] [scores_all(:,1,3);scores_all(:,1,6)]] , {'Grey', 'Lab','rgb'});
% ylabel('F-measure');
%
% test=cell(6,6,3);
% for meas = 1:3 %Fmeasure, Recall and Precision
%     for i=1:6
%         [~,t] = kstest((scores_all(:,meas,i)-mean(scores_all(:,meas,i)))/std(scores_all(:,meas,i)));
%         test{i,i,meas} = [t , median(scores_all(:,meas,i)), var(scores_all(:,meas,i)), quantile(scores_all(:,meas,i),[.05 0.95])];
%         for j=(i+1):6
%             [~,t,~] = anova1([scores_all(:,meas,i) scores_all(:,meas,j)], {'(a)', '(b)'}, 'off');
%             test{i,j,meas}=[t{2,6}, mean(scores_all(:,meas,i))-mean(scores_all(:,meas,j)), var(scores_all(:,meas,i))-var(scores_all(:,meas,j))];
%         end
%     end
% end

end