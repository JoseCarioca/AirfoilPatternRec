function [Sb,Sw,Sm]=scatter(x,c);
% SCATTER  - Cálculo de las matrices de dispersión de clases
%
%   [Sb,Sw,Sm]=scatter(x,c)

%	Copyright (c) Pedro L. Galindo (1998)

[d,n] = size(x);
clases = allclass(c);
nc = length(clases);

tm = meanpat(x);     

[nr,m,v] = clstats(x,c);

Sb = zeros(d);
Sw = zeros(d);

for i=1:nc
  mi = m(:,i);
  Sb = Sb + nr(i) * (mi-tm) * (mi-tm)';

  aux = myclass(x,c,clases(i));
  Sw = Sw + cov(aux')*(size(aux,2)-1);
end

Sm=Sb+Sw;

return

mediaglobal=meanpat(x);
M=[];
V=[];
NC  = size(x,1);
Sw = zeros(NC,NC);
Sb = zeros(NC,NC);
numclases=0;
for i=allclass(c),
   indices=find(c==i);
   N = length(indices);
   if N>=1,
      numclases=numclases+1;
      patrones=x(:,indices);
      mediaparcial = meanpat(patrones);
      M=[M mediaparcial];
      V=[V stdpat(patrones)];

      t = patrones - mediaparcial*ones(1,N);
      Sw = Sw + t * t';

      t = mediaparcial - mediaglobal;
      Sb = Sb + N * (t * t');
   end;
end;

Sm = Sb + Sw;

