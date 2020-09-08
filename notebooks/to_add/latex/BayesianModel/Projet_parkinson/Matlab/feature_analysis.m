function [div, ovlap] = feature_analysis(SBR,label)
%[div, ovlap] = feature_analysis(SBR,label)

% Ne marche que pour 2 classes
nb_classes = 2;
[nb_pts,nb_feat] = size(SBR);

%% Calcul divergence

div = zeros(nb_classes,nb_classes);

for carac=1:nb_feat
    
    for i=1:nb_classes
        
        for j=i+1:nb_classes
            database_1 = SBR(label,carac);
            database_2 = SBR(~label,carac);
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

%% Calcul overlap

database=SBR';
[nb_features,nb_samples] = size(database);
decisions = zeros(1,nb_samples);

for i=1:nb_features
    
    for j=1:nb_samples
        index_prototypes = [(1:j-1) (j+1:nb_samples)];
        prototypes = database(i,index_prototypes);
        test_sample = database(i,j);
        
        %Distance euclidienne
        distances = ((prototypes - test_sample*ones(1,nb_samples-1)).^2);
        
        [~,index_nn] = min(distances);
        decisions(j) = label(index_prototypes(index_nn));
    end
    
    ovlap(i) = sum(label ~= decisions)/nb_samples;
    
end

%%

titl = {
    ['Density function estimation of SBR of the Right Caudate'],...
    ['Density function estimation of SBR of the Left Caudate'],...
    ['Density function estimation of SBR of the Right Putamen'],...
    ['Density function estimation of SBR of the Left Putamen']
    };
notch_titl = {
    ['Notch boxes of SBR of the Right Caudate'],...
    ['Notch boxes of SBR of the Left Caudate'],...
    ['Notch boxes of SBR of the Right Putamen'],...
    ['Notch boxes of SBR of the Left Putamen']
    };

for i=1:length(label)
    
    if label(i) == true
        notch_box_label{i} = ['Early Parkinson'];
    else
        notch_box_label{i} = ['Normal'];
    end
    
end

for i=1:nb_feat
       
    figure; 
    plottools;
    
    subplot(1,2,1)
    
    [f1,xi1] = ksdensity(SBR(label,i),'kernel','normpdf','width',0.05);
    [f2,xi2] = ksdensity(SBR(~label,i),'kernel','normpdf','width',0.05);
    plot(xi1,f1,'-r',xi2,f2,'-g','Linewidth',2);
    legend('Early Parkinson','Normal');
    hold on;
    plot(xi1,f1,'+r',xi2,f2,'+g');
    title(titl(i));
    [f,xi] = ksdensity(SBR(label,i),'kernel','box','width',1/sqrt(length(SBR))); 
    hold on;
    bar(xi,f,'r'); 
    hold on; 
    [f,xi] = ksdensity(SBR(~label,i),'kernel','box','width',1/sqrt(length(SBR))); 
    bar(xi,f,'g'); 
    alpha(0.2);
    
    subplot(1,2,2);
    boxplot(SBR(:,i),notch_box_label,'notch','on');
    title(notch_titl(i));

end

end

