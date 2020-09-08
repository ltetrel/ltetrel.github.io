function mat = rigid_2_matrix(x_form)
%x_form en degre [x y theta_z]

trans = zeros(3);
roll = zeros(3);

x_form(3) = x_form(3)*pi/180;

trans(1,1)=1;
trans(2,2)=1;
trans(3,3)=1;
trans(1,3)=x_form(1);
trans(2,3)=x_form(2);

roll(3,3)=1;
roll(1,1)=cos(x_form(3));
roll(2,2)=cos(x_form(3));
roll(1,2)=-sin(x_form(3));
roll(2,1)=sin(x_form(3));

mat = trans*roll;

end
