function output = correlation(database, labels)

[nb_samples, nb_features] = size(database);
output = 0;

for i = 1:nb_samples
    i,
    distances = zeros(1,nb_samples);
    mi = mean(database(i,:));
    stdi = std(database(i,:));
    for j = 1:nb_samples
        mj = mean(database(j,:));
        stdj = std(database(j,:));
        distances(j) = sum((database(i,:)-mi).*(database(j,:)-mj))/(nb_features*stdi*stdj);
    end
    distances(i) = inf;
    [distance_nn,index_nn] = min(distances); 
    if (labels(index_nn) ~= labels(i))
        output = output+1;
    end
end

output = output/nb_samples; 

end



