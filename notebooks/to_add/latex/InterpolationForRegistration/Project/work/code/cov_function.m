function K = cov_function(grid_star,grid,l)

[mesh_x_star, mesh_x] = ndgrid(grid_star(1,:),grid(1,:));
[mesh_y_star, mesh_y] = ndgrid(grid_star(2,:),grid(2,:));

K = exp( -( (mesh_x_star - mesh_x).^2 + ...
            (mesh_y_star - mesh_y).^2 )/ ...
                                (2*l^2) );            
                            
% for i = 1:length(grid_star(1,:));
%     for j = i:length(grid(1,:));
%         K(i,j) = exp( -sqrt(sum((grid_star(1:2,i) - ...
%                                 grid(1:2,j)).^2)) / ...
%                                 (2*l^2) );
%     end
% end
% 
% K = K+K';
% K(1:length(grid_star(1,:))+1:numel(K)) = K(1:length(grid_star(1,:))+1:numel(K))/2;

end

