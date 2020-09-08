function div = divergence(database, labels)

div = zeros(10,10);

for carac=1:100
    
    for i=1:10
        
        for j=i+1:10
            database_1 = database(labels==i,carac);
            database_2 = database(labels==j,carac);
            mu_1 = mean(database_1,1);
            covar_1 = std(database_1,1);
            mu_2 = mean(database_2,1);
            covar_2 = std(database_2,0,1);
            fisher = mean((abs(mu_1  - mu_2).^2)./(covar_1.^2 + covar_2.^2));
            div(i,j,carac) = fisher;
            div(j,i,carac) = div(i,j,carac);
        end
        
    end
    
end

end