function [X,Y,grid_point,im] = im_2_grid(im, size_pix, n_point)

s = size(im);
x_dist = size_pix(1)/2:size_pix(1)*n_point:size_pix(1)*s(1)-size_pix(1)/2;
y_dist = size_pix(2)/2:size_pix(2)*n_point:size_pix(2)*s(2)-size_pix(2)/2;
xcenter = (size_pix(1)*s(1)-size_pix(1)/2)/2; %Rotation at the center of the image
ycenter = (size_pix(2)*s(2)-size_pix(2)/2)/2;
x_dist = x_dist - xcenter;
y_dist = y_dist - ycenter;

Y = zeros(length(y_dist),length(x_dist));
X = zeros(length(y_dist),length(x_dist));
[Y,X] = ndgrid(y_dist,x_dist);
Y = flipud(Y);
grid_point = [X(:)'; Y(:)'; ones(1,length(Y(:)))];
im = im(1:n_point:end, 1:n_point:end);

end

