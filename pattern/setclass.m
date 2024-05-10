function clasec = setclass(x,y,centroides)
% SETCLASS - Asigna la clase correspondiente a los centroides
%
%    clases = setclass(x,y,centroides);
%

%	Copyright (c) Pedro L. Galindo (1998)

numcentroides = size(centroides,2);
numpatrones = size(x,2);

% Asignar la clase a cada centroide
clases = mycenter(x,centroides);
for i=1:numcentroides,
   indices = find(clases==i);
   if isempty(indices),
      error('Quedó una clase vacia');
   end;
   salidas = y(indices);
   clasec(i) = moda(salidas);
end;