function d = dg_avg(x,y)
% DG_AVG   - Distancia intergrupos = distancia media entre cada pareja de puntos
%             Se puede utilizar tambien como distancia de un punto a un grupo.
%
%	d = dg_avg(x,y)
%
%         x  = conjunto de patrones del grupo 1 (por columnas)
%         y  = conjunto de patrones del grupo 2 (por columnas)
%              (y puede ser tambien un patron individual) 
%         d  = Distancia del grupo 1 al grupo 2

%	Copyright (c) Pedro L. Galindo (1998)

suma = 0;
for i=1:size(y,2),
   suma = suma + sum(d_euclid(x,y(:,i)));
end;
d = suma / (size(x,2)*size(y,2));