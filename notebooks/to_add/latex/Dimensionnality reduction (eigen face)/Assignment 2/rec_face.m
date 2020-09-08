function [data_test_rec,rec_error] = rec_face(data_train, eigen_faces, weights_train, n_eigen)

m = mean(data_train);
rec_error = 0;

% Reconstruction into pixel space
data_test_rec = weights_train(:,1:n_eigen)*eigen_faces(1:n_eigen,:) + ones(size(eigen_faces,1),1)*m;
rec_error = sqrt(sum(sum((data_test_rec-data_train).^2)))/size(eigen_faces,2);

end

