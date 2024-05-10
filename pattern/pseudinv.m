function y=pseudinv(x);
% PSEUDINV - Calcula la pseudoinversa de x
%
%          y=pseudinv(x);
% 

y=inv(x'*x+1e-6*eye(size(x'*x)))*x';