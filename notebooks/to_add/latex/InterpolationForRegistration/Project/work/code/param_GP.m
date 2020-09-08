function [Mu,Sigma] = param_GP(grid_point,grid_point_m_f,l,noise,im_m_column);


K_XsX = cov_function(grid_point,grid_point_m_f,l);
K_XsXs = cov_function(grid_point,grid_point,l);
% K_XX = K_XsXs; if both same grid
K_XXs = cov_function(grid_point_m_f,grid_point,l);

Mu = K_XsX*inv(K_XsXs + (single(noise)^2)*eye(length(K_XsXs)));
Sigma = K_XsXs - Mu*K_XXs;

Mu = Mu*im_m_column;

end

