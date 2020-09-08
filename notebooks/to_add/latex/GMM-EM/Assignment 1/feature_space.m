function M = feature_space(im_in,c)
% c = 1 : Intenisty image
% c = 2 : rgb image
% c = 3 : L*a*b* image

if c==1
    
    M = rgb2gray(im_in);
    M = M(:);
    
elseif c==2
    
    tmp = im_in(:,:,1);
    M(:,1) = tmp(:);
    tmp = im_in(:,:,2);
    M(:,2) = tmp(:);
    tmp = im_in(:,:,3);
    M(:,3) = tmp(:);
    
else
    
    colorTransform = makecform('srgb2lab');
    lab = applycform(im_in, colorTransform);
%     lab = im2uint8(rgb2lab(im_in));
%     tmp  = lab(:, :, 1);  % Extract the A image.
%     M(:,1) = tmp(:);
    tmp  = lab(:, :, 2);  % Extract the A image.
    M(:,1) = tmp(:);
    tmp = lab(:, :, 3);  % Extract the B image.
    M(:,2) = tmp(:);
    
end

end