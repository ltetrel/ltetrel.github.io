function x_form = matrix_2_rigid(mat)
%x_form en degre

x_form(1)=mat(1,3);
x_form(2)=mat(2,3);
x_form(3) = ( atan2(mat(2,1),mat(1,1)) )*180/pi;

end