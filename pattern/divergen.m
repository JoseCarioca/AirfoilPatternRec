function d=divergen(m1,c1,m2,c2);
% DIVERGEN - Calcula la divergencia de dos normales multivariadas
%
%           d=divergen(m1,c1,m2,c2)
%
%           (Tou & Gonzalez, pg. 293)
%

%	Copyright (c) Pedro L. Galindo (1998)

invc1=inv(c1);
invc2=inv(c2);

d = 0.5 * trace((c1-c2)*(invc2-invc1)) + ...
    0.5 * trace((invc1+invc2)*(m1-m2)*(m1-m2)');
