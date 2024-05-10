function W=bayesbon(m1,C1,m2,C2,P1,P2,C);
% BAYESBON - Calcula la interseccion de dos normales 2D
%
%            W=bayesbon(media1,cova1,media2,cova2,P1,P2,C);
%
%            P1,P2 : probabilidades a priori
%            C     : matriz de costes
%

%	Copyright (c) Pedro L. Galindo (1998)


if nargin==4,
   C=[0 1;1 0];  % Matriz de costes
   P1=1;         % Prob. apriori clase 1
   P2=1;         % Prob. apriori clase 2
elseif nargin~=7,
   help bayesbon
   return
end;

C1inv = inv(C1);
C2inv = inv(C2);

% h(x) = x' * Q * x + V' *x + V0 :

Q  = C2inv-C1inv;
V = 2*(C1inv * m1 - C2inv * m2);
V0 = m2' * C2inv * m2 - m1' * C1inv * m1 + ...
     log( det(C2) / det(C1) * ...
          (C(2,1)-C(1,1))^2 * P1^2 / ((C(1,2)-C(2,2))^2 * P2^2) ...
         );

% a*x.^2 + b*x^2 + c*x*y + d*x + e*y + f = 0;

a = Q(1,1); 
b = Q(2,2);
c = Q(1,2)+Q(2,1);
d = V(1);
e = V(2);
f = V0;

W=[a b c d e f]';
