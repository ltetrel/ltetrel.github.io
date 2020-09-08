function main

patient = read_file('training_001');

%% Demo

% im = patient.mr_T1.data(:,:,10);
% pix_size = patient.mr_T1.header.PixelSize;
% s = size(im);
%
% %with downsampling in x
% d = 5;
% im_d = imresize(im(:,1:d:s(1)), s, 'bicubic');
% figure;
% imshow(im_d);
% title('downsampled');
%
% im_d_column = im_d(:);
%
% [X,Y,grid_point] = im_2_grid(im_d, pix_size,1);
%
% %project moving grid into fixed image boundaries
% bound_constraint = [min(grid_point(1,:))-0.001 max(grid_point(1,:))+0.001];
% % bound_constraint = [-inf inf];
%
% figure;
% mesh(X,Y,im_d);
% colormap('gray');
% xlabel('x (mm)');
% ylabel('y (mm)');
% zlabel('grey value');
% title('im');
% hidden off;
% view(0,90);
%
% Tr = rigid_2_matrix([100 100 217]);
% grid_point_m = Tr*grid_point;
%
% [im_d_t,X_t,Y_t] = grid_2_im(grid_point_m,im_d_column,bound_constraint);
%
% figure;
% imshow(imresize(im_d_t,1));
% title('im transformed');
%
% figure;
% mesh(X_t,Y_t,im_d_t);
% colormap('gray');
% xlabel('x (mm)');
% ylabel('y (mm)');
% zlabel('grey value');
% title('im transformed');
% hidden off;
% view(0,90);

%% Main
figure;
for n = 1:100
    % random selection of slice
    s = size(patient.mr_T1.data);
    slice = randi(s(3)-4); % every slice but not the last one too little
    im = patient.mr_T1.data(:,:,slice);
    
    pix_size = patient.mr_T1.header.PixelSize;
    s = size(im);
    n_point_p_grid = 8; % Number of pixel (MIN 5) in 1 grid case in one direction
    noise = 0.1; % Noise parameter for GP
    l = 6.25; %window size for GP
    
    % % removing a little background
    % t1 = find(diff(sum(im<quantile(im(:),0.85),2)));
    % t2 = find(diff(sum(im<quantile(im(:),0.85),1)));
    % b = min([t1(1) t2(1) s(1)-t1(end) s(2)-t2(end)])-15;
    % im = im(b:end-b, b:end-b);
    s = size(im);
    
    %Image downsampling by 5, randomly horizontal or vertical
    d = 5;
    dir_d = rand();
    if dir_d>0.5
        im_d = imresize(im(:,1:d:s(1)), s, 'bicubic');
    else
        im_d = imresize(im(1:d:s(2),:), s, 'bicubic');
    end
    
    %% Fixed image and moving image creation
    
    [X_f,Y_f,grid_point,im_d] = im_2_grid(im_d, pix_size, 1);
    im_d_column = im_d(:);
    
    %project moving grid into fixed image boundaries
    bound_constraint = [min(grid_point(1,:))-1 max(grid_point(1,:))+1];
    % bound_constraint = [-inf inf];
    
    Tr = [(([size(im,1) size(im,2)].*pix_size)/50).*rand(1,2) - [size(im,1) size(im,2)].*pix_size/100 ...
        360*rand()-180];
    Tr(1:2) = 0;
    Tr = rigid_2_matrix(Tr);
    grid_point_m = Tr*grid_point;
    
    [im_m,X_m,Y_m] = grid_2_im(grid_point_m,im_d_column,bound_constraint);
    
    %% Grid creation
    
    [X_m,Y_m,grid_point_m,im_m] = im_2_grid(im_m, pix_size, n_point_p_grid);
    [X_f,Y_f,grid_point,im_d] = im_2_grid(im_d, pix_size, n_point_p_grid);
    
    im_m_column = im_m(:);
    im_d_column = im_d(:);
    
    %% Random transformation and projection of the moving image into fixed image
    
    for i = 1:100
        
        disp([ num2str(rat(((i*350/100-180)/180),0.1)) 'pi']);
        T = rigid_2_matrix([0 0 i*359/100-180])
        grid_point_m_f = inv(T)*grid_point_m;
        
        %% Interpolation of the pixel values of moving image into fixed image
        
        [Mu,Sigma] = param_GP(grid_point,grid_point_m_f,l,noise,im_m_column);
        Sigma_t = zeros(length(Mu));
        Sigma_t = Sigma + noise^2*eye(length(Mu));
        %     Sigma_t(1:size(Sigma_t,1)+1:numel(Sigma_t)) = diag(Sigma + noise^2*eye(length(Mu))).^2;
        %     Similarity = log(((2*pi)^(-length(Mu)/2))*(det(Sigma_t))^(-0.5))- ...
        %         0.5*(im_d_column-im_m_column)'*inv(Sigma_t)*(im_d_column-im_m_column);
        Similarity(n,i) = - 0.5*(im_d_column-Mu)'*inv(Sigma_t)*(im_d_column-Mu);
        sse(n,i) = sum((matrix_2_rigid(T) - matrix_2_rigid(Tr)).^2);
        mtre(n,i) = mean(sqrt(sum((grid_point_m_f - inv(Tr)*grid_point_m).^2)));
        
        [Y_t,X_t] = ndgrid(sort(grid_point_m_f(2,:)),sort(grid_point_m_f(1,:)));
        Y_t = flipud(Y_t);
        [ind,D] = knnsearch(grid_point_m_f(1:2,:)',[X_t(:) Y_t(:)],'Distance','cityblock');
        im_t = im_m_column(ind);
        im_t(im_t<0) = 0.01;
        im_t = single(vec2mat(im_t,length(grid_point_m_f(2,:))))';
        
        Vq_cubic = interp2(X_t,Y_t,im_t,X_f,Y_f,'spline',0);
        Similarity_cubic(n,i)= sum(sum((im_d-Vq_cubic).^2));
        
        Vq_nearest = interp2(X_t,Y_t,im_t,X_f,Y_f,'nearest',0);
        Similarity_nearest(n,i)= sum(sum((im_d-Vq_nearest).^2));    
        
    end
    disp(['Iteration : ' num2str(n) '/' num2str(100)]);
%     grid_point_m_f = inv(Tr)*grid_point_m;
%     [Mu,Sigma] = param_GP(grid_point,grid_point_m_f,l,noise,im_m_column);
%     Sigma_t = Sigma + noise^2*eye(length(Mu));
%     Similarity2 = - 0.5*(im_d_column-Mu)'*inv(Sigma_t)*(im_d_column-Mu);
%     Tr2 = matrix_2_rigid(Tr);
%     
%     [Y_t,X_t] = ndgrid(sort(grid_point_m_f(2,:)),sort(grid_point_m_f(1,:)));
%     Y_t = flipud(Y_t);
%     [ind,D] = knnsearch(grid_point_m_f(1:2,:)',[X_t(:) Y_t(:)]);
%     im_t = im_m_column(ind);
%     im_t = single(vec2mat(im_t,length(grid_point_m_f(2,:))))';
%     
%     Vq_cubic = interp2(X_t,Y_t,im_t,X_f,Y_f,'spline',0);
%     Similarity2_cubic = sum(sum((im_d-Vq_cubic)^2));
%     
%     Vq_nearest = interp2(X_t,Y_t,im_t,X_f,Y_f,'nearest',0);
%     Similarity2_nearest = sum(sum((im_d-Vq_nearest)^2));
%     
%     [im_m_f_obs,X_m_f_obs,Y_m_f_obs] = grid_2_im(grid_point_m_f,im_m_column,bound_constraint);
%     figure;
%     subplot(2,3,1)
%     imshow(Vq_nearest);
%     xlabel('NN Resampled values');
%     subplot(2,3,2)
%     imshow(Vq_cubic);
%     xlabel('Cubic Resampled values');
%     subplot(2,3,3)
%     imshow(vec2mat(Mu,sqrt(length(Mu)))',[]);
%     xlabel('GP Resampled values');
%     subplot(2,3,5)
%     imshow(im_m_f_obs,[]);
%     xlabel('transformed Moving image');
%     subplot(2,3,4)
%     imshow(im_d,[]);
%     xlabel('Fixed image');
%     
%     Tr = matrix_2_rigid(Tr);
%     figure;
%     subplot(1,2,1);
%     loglog(sse(n,:), Similarity_cubic(n,:),'+');
%     hold on;
%     xL = get(gca,'XLim');
%     line(xL,[Similarity2_cubic Similarity2_cubic],'Color','r');
%     xlabel('Registration Error (mm²)');
%     ylabel('SSD with cubic');
%     hold off
%     
%     subplot(1,2,2);
%     semilogy([1:100]*359/100-180, Similarity_cubic(n,:),'+');
%     hold on;
%     yL = get(gca,'YLim');
%     line([Tr(3) Tr(3)],yL,'Color','r');
%     xlabel('Rotation (°)');
%     ylabel('SSD with cubic');
%     hold off
%     drawnow;
%     %%
%     figure;
%     subplot(1,2,1);
%     loglog(sse(n,:), Similarity_nearest(n,:),'+');
%     hold on;
%     xL = get(gca,'XLim');
%     line(xL,[Similarity2_nearest Similarity2_nearest],'Color','r');
%     xlabel('Registration Error (mm²)');
%     ylabel('SSD with NN');
%     hold off
%     
%     subplot(1,2,2);
%     semilogy([1:100]*359/100-180, Similarity_nearest(n,:),'+');
%     hold on;
%     yL = get(gca,'YLim');
%     line([Tr(3) Tr(3)],yL,'Color','r');
%     xlabel('Rotation (°)');
%     ylabel('New similarity measure');
%     hold off
%     drawnow;
%     %%
%     figure;
%     subplot(1,2,1);
%     loglog(sse(n,:), Similarity(n,:),'+');
%     hold on;
%     xL = get(gca,'XLim');
%     line(xL,[Similarity2 Similarity2],'Color','r');
%     xlabel('Registration Error (mm²)');
%     ylabel('New similarity measure');
%     hold off
%     
%     subplot(1,2,2);
%     semilogy([1:100]*359/100-180, Similarity(n,:),'+');
%     hold on;
%     yL = get(gca,'YLim');
%     line([Tr(3) Tr(3)],yL,'Color','r');
%     xlabel('Rotation (°)');
%     ylabel('New similarity measure');
%     hold off
%     drawnow;
    
end

save('res','Similarity','Similarity_nearest','Similarity_cubic','sse','mtre')

% [~,ind] = min(Similarity,[],2);

%% Figure trace

% figure;
% subplot(2,2,1)
% imshow(im);
% xlabel(['slice no' num2str(slice)]);
%
% if dir_d>0.5
%     subplot(2,2,2)
%     imshow(im_d);
%     xlabel('downsampled in x direction');
% else
%     subplot(2,2,2)
%     imshow(im_d);
%     xlabel('downsampled in y direction');
% end
%
% subplot(2,2,3);
% imshow(imresize(im_m,1));
% xlabel(['im transformed by ' num2str(matrix_2_rigid(Tr)) '°']);
%
% figure;
% mesh(X_f,Y_f,im_d);
% colormap('gray');
% xlabel('x (mm)');
% ylabel('y (mm)');
% zlabel('grey value');
% title('original grid I_F','interpreter','tex');
% hidden off;
% view(0,90);
%
% figure;
% mesh(X_m,Y_m,im_m);
% colormap('gray');
% xlabel('x (mm)');
% ylabel('y (mm)');
% zlabel('grey value');
% title('transformed grid I_M','interpreter','tex');
% hidden off;
% view(0,90);

% figure;
% [f,xi] = ksdensity(diag(Sigma));
% [~,ind] = max(abs(diff(f)));
% c = [(cumsum(f)/sum(f))'  (normpdf(xi,xi(ind),0.1)/max(normpdf(xi,xi(ind),0.1)))' 1-(cumsum(f)/sum(f))'];
% load color;
% surf(X_f,Y_f,vec2mat(Mu,sqrt(length(Mu)))',vec2mat(diag(Sigma),sqrt(length(diag(Sigma))))');
% xlabel('x (mm)');
% ylabel('y (mm)');
% zlabel('grey value');
% title(['Grid resampling at position I_F' ],'interpreter','tex');
% colormap(c);
% cb = colorbar;
% cb.Label.String = 'Interpolation uncertainty';
% view(0,90);
% pause(0.01);
% drawnow;

% [im_m_f_obs,X_m_f_obs,Y_m_f_obs] = grid_2_im(grid_point_m_f,im_m_column,bound_constraint);
% %
% figure;
% subplot(2,2,1)
% imshow(vec2mat(Mu,sqrt(length(Mu)))',[]);
% xlabel('Resampled values');
% subplot(2,2,2)
% imshow(im_m_f_obs,[]);
% xlabel('transformed Moving image');
% subplot(2,2,3)
% imshow(im_d,[]);
% xlabel('Fixed image');

% figure;
% subplot(1,2,1);
% loglog(sse, Similarity,'+');
% hold on;
% xL = get(gca,'XLim');
% line(xL,[Similarity2 Similarity2],'Color','r');
% xlabel('Registration Error');
% ylabel('New similarity measure');
%
% subplot(1,2,2);
% semilogy([1:100]*359/100-180, Similarity,'+');
% hold on;
% semilogy(Tr2(3), Similarity2,'r+','Linewidth',5);
% xlabel('Ascending rotation');
% ylabel('New similarity measure');

% figure; semilogy([1:500]*359/500-180, mtre(62,:),'linewidth',1.5); hold on; semilogy([1:500]*359/500-180, Similarity_nearest(62,:),'r','linewidth',1.5); hold on; semilogy([1:500]*359/500-180, Similarity_cubic(62,:),'Color',[0.75 0.75 0],'linewidth',1.5); hold on; semilogy([1:500]*359/500-180, (-1).*Similarity(62,:),'Color',[0 0.6 0],'linewidth',1.5); ylim([0 1000]); ylabel('Similarity'); xlabel('Rotation (°)'); legend('mTRE','SSD (NN)', 'SSD (SPL)', 'GP');

end

