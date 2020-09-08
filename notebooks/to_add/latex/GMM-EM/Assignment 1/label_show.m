function im_label = label_show(label, size_im, color)

im_label = zeros(size_im(1),size_im(2),3,'single');
label2 = vec2mat(label,size_im(1))'; 

for i=1:size_im(1)
    for j=1:size_im(2)
        im_label(i,j,:)=color{label2(i,j)};
    end
end

imshow(im_label);

end