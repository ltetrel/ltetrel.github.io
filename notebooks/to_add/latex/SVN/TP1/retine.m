function output = retine(im,nb_l,nb_c)
% output = retine(im,nb_l,nb_c)

output = ones(1,nb_l*nb_c);

%% Centralisation de l'image

taille_image1 = size(im);
m = max(taille_image1) ;

image_2 = true(m,m);
image_2(floor((m - taille_image1(1))/2)+1 : taille_image1(1)+floor((m - taille_image1(1))/2) , floor((m - taille_image1(2))/2)+1 : taille_image1(2)+floor((m - taille_image1(2))/2) ) = im;

%% Zoom de l'image

image_3 = imresize(image_2, [nb_l*m nb_c*m]);

%% Calcul des caractéristiques

cnt=1;
for j=1:nb_c
    
    for i=1:nb_l
        
    zone = image_3( m*(i-1)+1 :i*m , m*(j-1)+1 :j*m );
    ratio = sum(sum(zone))/m^2;
    output(cnt) = ratio;
    cnt=cnt+1;
    
    end
end

end

    