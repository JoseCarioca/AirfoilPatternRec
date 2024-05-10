function y=allclass(t)
% ALLCLASS - Devuelve un vector con las clases existentes en t ordenadas ascendentemente
%
%          y=allclass(t)
%
%          t = targets
%          y = clases existentes en t ordenadas
%
%          Es muy util para realizar bucles de la forma    
%              for i = allclass(t), ...

%	Copyright (c) Pedro L. Galindo (1998)

j=1;
for i=min(t):max(t),
   indices = find(t==i);
   if length(indices)>0,
      y(j)=t(indices(1));
      j=j+1;      
   end;
end;