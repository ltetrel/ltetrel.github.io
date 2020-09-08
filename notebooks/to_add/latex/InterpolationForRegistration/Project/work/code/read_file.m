function patient = read_file(name)
% Read all raw data from one patient

listing = dir(['Data\' name]);

for i=1:length(listing)-2

    patient.(listing(i+2).name) = struct('header',char(),'data', zeros(5,5,5,'single'));
    n = char(fieldnames(patient));
    if sum(uint8(n(i,1:2)) == uint8('tr'))~=2
        patient.(listing(i+2).name).header = helperReadHeaderRIRE(['Data\' name '\' listing(i+2).name '\header.ascii']);  
        patient.(listing(i+2).name).data = multibandread(['Data\' name '\' listing(i+2).name '\image.bin'],[patient.(listing(i+2).name).header.Rows, patient.(listing(i+2).name).header.Columns, patient.(listing(i+2).name).header.Slices],'int16=>single', 0, 'bsq', 'ieee-be' );
        for j=1:size(patient.(listing(i+2).name).data,3)
            patient.(listing(i+2).name).data(:,:,j) = patient.(listing(i+2).name).data(:,:,j)./(max(max(patient.(listing(i+2).name).data(:,:,j))));
        end
    end
end

end

