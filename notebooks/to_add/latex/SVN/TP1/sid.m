function output = sid(database,labels)

database=database';
[nb_features, nb_samples] = size(database);
decisions = zeros(1,nb_samples);

tic;
for i=1:nb_samples

    tstart = tic;
    iteration = i,
    index_prototypes = [(1:i-1) (i+1:nb_samples)];
    prototypes = database(:,index_prototypes);
    test_sample = database(:,i);
    
    t = size(prototypes);
    s1 = ones(t(1),1) * sum(prototypes);
    pl = prototypes./s1;
    A = test_sample*ones(1,nb_samples-1);
    s2 = ones(t(1),1) * sum(A);
    ql = A./s2;
    
    Dsi = sum(pl.*log(pl./ql));
    Dsj = sum(ql.*log(ql./pl));
    
    distances = Dsi + Dsj;
    
    temps=toc(tstart)*(nb_samples-i);
    disp(['Estimation du temps restant : ' num2str(temps) 's']);
    toc;
end

[distance_nn,index_nn] = min(distances);
decisions(i) = labels(index_prototypes(index_nn));

output = sum(labels ~= decisions)/nb_samples;
end
