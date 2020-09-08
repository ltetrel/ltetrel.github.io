function output = overlap_v2(database,labels)
% output = overlap_v2(database,labels)
% data => [nb_samples,nb_features]
% labels => [1,nb_samples];
database=database';
[nb_features,nb_samples] = size(database);
decision = zeros(1,nb_samples);

for i=1:nb_samples
index_prototypes = [(1:i-1) (i+1:nb_samples)];
prototypes = database(:,index_prototypes);
test_sample = database(:,i);

%Distance euclidienne
distances = sum((prototypes - test_sample*ones(1,nb_samples-1)).^2);

[distance_nn,index_nn] = min(distances);
decisions(i) = labels(index_prototypes(index_nn));
end

output = sum(labels ~= decisions)/nb_samples;