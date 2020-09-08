function [im,X_t,Y_t] = grid_2_im(grid_point,im_d_column,bound_constraint)

grid_point(:,sum(bound_constraint(1)<[grid_point(1,:)' grid_point(2,:)'] & [grid_point(1,:)' grid_point(2,:)']<bound_constraint(2),2)~=2) = nan;
length_im = sqrt(length(im_d_column));
x_dist = linspace(min(grid_point(1,:)),max(grid_point(1,:)),length_im);
y_dist = linspace(min(grid_point(2,:)),max(grid_point(2,:)),length_im);
[Y_t,X_t] = ndgrid(y_dist,x_dist);
Y_t = flipud(Y_t);

im = zeros(length(y_dist),length(x_dist),'single');
[ind,D] = knnsearch(grid_point(1:2,:)',[X_t(:) Y_t(:)]);
im = im_d_column(ind);

im(sum(bound_constraint(1)<[X_t(:) Y_t(:)] & [X_t(:) Y_t(:)]<bound_constraint(2),2)~=2) = NaN;
im = vec2mat(im,length_im)';

end

