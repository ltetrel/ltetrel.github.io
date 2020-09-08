% function vint = GPInterp(I,nX,nY,sz)
% Gaussian process interpolation
% I : image
% nX : number of sample in X direction
% nY : number of sample in Y direction
clear all
close all

sz = [5 5 50 100];

nX=sz(1);
nY=sz(2);

w = sz(3);
h = sz(4);
dx = w/(nX-1);
dy = h/(nY-1);
subSample = 5;
margex = dx/subSample;
margey = dy/subSample;

[imw,imh] = meshgrid(0:dx:w,0:dy:h);
[xi,yi] = meshgrid(0:margex:w, 0:margey:h);

% randMux = zeros(size(imw));
% randMuy = zeros(size(imw));
% randMuz = zeros(size(imw));
% TREcov = zeros(3,3,size(imw,1),size(imw,2));


dimX = size(imw,1);
dimY = size(imh,2);

sampx = floor(linspace(1,w,nX));
sampy = floor(linspace(1,h,nY));

z=0;
for i=1:length(sampx)
    for j=1:length(sampy)
        z = z + 1;
        sampMat(z,:) = [sampx(i) sampy(j)];
        sigma(z) = 1;
        obs(z) = i*j;
        if i == j
            obs(z) = 0;
        end
    end
end

sigma_p = 10;
l = 15;

for i=1:length(sampMat)
    for j=1:i
        s1 = sampMat(i,:);
        s2 = sampMat(j,:);
        K(i,j) = squaredExpo(s1,s2,sigma_p,l);
        K(j,i) = K(i,j);
        
        if i==j
            K(i,j) = K(i,j) + sigma(i)^2;
        end
        
    end
end

invK = K^-1;

tic

for x=1:w
    for y=1:h
        s = [x y];

        % weight function
        intV = 0;
        for i=1:length(K)
            f=0;
            for j=1:length(K)
                sj = sampMat(j,:);
                f = f + invK(i,j) * squaredExpo(sj,s,sigma_p,l);
            end

            phi(i) = f;
            intV = intV + obs(i)*phi(i);
        end
        
        I(x,y) = intV;

    end
end

display('Gaussian process interpolation time :');
t = toc;
datestr(t/24/3600,'HH:MM:SS')


Orig = zeros([w h]);
z=0; i=1; j=1;
for z=1:length(sampMat)
    c = sampMat(z,:);
    Orig(c(1),c(2)) = obs(z);
    
    beforeInt(i,j) = obs(z);
    j=j+1;
    if j==nY+1
        i=i+1;
        j=1;
    end
    
end

surf(Orig,'EdgeColor','none');
hold on
surf(I,'EdgeColor','none');
shading interp

[imw,imh] = meshgrid(sampx,sampy);
[xi,yi] = meshgrid(1:w, 1:h);

tic

afterInt = interp2(imw,imh,beforeInt,xi,yi);

display('Bilinear interpolation time :');
t = toc;
datestr(t/24/3600,'HH:MM:SS')

figure
surf(Orig,'EdgeColor','none');
hold on
surf(afterInt','EdgeColor','none');
shading interp

figure
surf(Orig,'EdgeColor','none');
hold on
surf(afterInt','EdgeColor','b');
shading interp
surf(I,'EdgeColor','r');
shading interp


axis equal