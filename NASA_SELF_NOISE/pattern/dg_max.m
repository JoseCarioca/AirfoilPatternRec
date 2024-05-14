function [d,i1,i2] = dg_max(x,y)
% DG_MAX   - Distancia intergrupos = maxima distancia entre sus patrones
%             Se puede utilizar tambien como distancia de un punto a un grupo.
%
%	[d,i1,i2] = dg_max(x,y)
%
%         x  = conjunto de patrones del grupo 1 (por columnas)
%         y  = conjunto de patrones del grupo 2 (por columnas)
%         d  = Distancia del grupo 1 al grupo 2
%         i1 = indice del patron del grupo x mas cercano al grupo y
%         i2 = indice del patron del grupo y mas cercano al grupo x

%	Copyright (c) Pedro L. Galindo (1998)

maximo=-1e10;
for i=1:size(y,2),
   d=d_euclid(x,y(:,i));
   [valor,posicion]=max(d);
   if valor>maximo,
      maximo=valor;
      i1=posicion;
      i2=i;
   end;
end;

d=maximo;