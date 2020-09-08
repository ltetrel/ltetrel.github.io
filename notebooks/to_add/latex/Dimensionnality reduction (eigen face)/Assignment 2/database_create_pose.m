function database_create_pose(data,name)

data_train = cell(1,7);
name_train = cell(1,7);
% order -90,-50,-15,0,15,50,90
n = {   {'pl'};...
        {'hl','bi','bh','ra'};...
        {'bg','ql','bf','rb'};...
        {'fa','fb','ba','bj','bk'};...
        {'be','qr','bd','rc'};...
        {'bc','rd','bb','hr'};...
        {'pr','re'} };
name_u = uint8(name);
data_test = [];
name_test = [];

for i=1:7
    data_t = [];
    name_t = [];
    for j=1:length(n{i})
        ind = sum(name_u(:,14:15) == uint8(ones(1664,1)*double(uint8(char(n{i}{j})))),2)==2;
        data_t = [data_t; data(ind,:)];
        name_t = [name_t ;name(ind,:)];
    end
    ind_rnd = randperm(size(name_t,1));
    q = round(length(ind_rnd)*0.9);
    data_train{i} = data_t(ind_rnd(1:q),:);
    name_train{i} = char(name_t(ind_rnd(1:q),:));
    data_test = [data_test; data_t(ind_rnd(q+1:end),:)];
    name_test = [name_test; char(name_t(ind_rnd(q+1:end),:))];
end

save('Data_im_pose','data_train','name_train','data_test','name_test');

end