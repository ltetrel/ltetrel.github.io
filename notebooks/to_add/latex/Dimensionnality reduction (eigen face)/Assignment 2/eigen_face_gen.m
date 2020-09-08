function [eigen_faces,V,v,weights_train] = eigen_face_gen(data_train)

m = mean(data_train);
% imshow(vec2mat(m,384)');
A = data_train - ones(size(data_train,1),1)*m;

% We know from linears algebra theory that for a PxQ matrix, the maximum
% number of non-zero eigenvalues that the matrix can have is min(P-1,Q-1).
% Since the number of training images (P) is usually less than the number
% of pixels (M*N), the most non-zero eigenvalues that can be found are equal
% to P-1. So we can calculate eigenvalues of A'*A (a PxP matrix) instead of
% A*A' (a M*NxM*N matrix). It is clear that the dimensions of A*A' is much
% larger that A'*A. So the dimensionality will decrease.

[V v] = eig(A*A');
v = diag(v);
v = v(end:-1:1);
V = V(:,end:-1:1);
V(:,v<0)=[]; 
v(v<0)=[]; 
eigen_faces = V'*A;

%Normalization of eigen faces
nrm = sqrt(v)*ones(1,size(eigen_faces,2));
eigen_faces = eigen_faces./nrm; 

weights_train = A*eigen_faces';

end

