function d = dg_means(x,y)
% DG_MEANS - Distancia intergrupos = distancia entre las medias de los dos grupos
%             Se puede utilizar tambien como distancia de un punto a un grupo.
%
%	d = dg_means(x,y)
%
%         x  = conjunto de patrones del grupo 1 (por columnas)
%         y  = conjunto de patrones del grupo 2 (por columnas)
%         d  = Distancia del grupo 1 al grupo 2

%	Copyright (c) Pedro L. Galindo (1998)

if size(x,2)==1,
   media1=x;
else,
   media1=meanpat(x);
end;

if size(y,2)==1,
   media2=y;
else,
   media2=meanpat(y);
end;

d=d_euclid(media1,media2);
