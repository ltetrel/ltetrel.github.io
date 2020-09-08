function k = squaredExpo(s1,s2,sigma_p,l)

d = sqrt((s1(1) - s2(1))^2 + (s1(2) - s2(2))^2);

k = sigma_p^2 * exp(-1/(2*l^2)*d^2);

end