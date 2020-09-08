function output = euclidienne(database, labels)

[nb_samples, nb_features] = size(database);
output = 0;

for i = 1:nb_samples
    i,
    distances = zeros(1,nb_samples);
    for j = 1:nb_samples
        distances(j) = sum((database(i,:)-database(j,:)).^2);
    end
    distances(i) = inf;
    [distance_nn,index_nn] = min(distances);
    if (labels(index_nn) ~= labels(i))
        output = output+1;
    end
end

output = output/nb_samples;

end

