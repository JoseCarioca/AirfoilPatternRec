function d = dg_means2(x,y)
% DG_MEANS2 - Distancia intergrupos = distancia media ponderada entre las medias de los 2 grupos
%             Se puede utilizar tambien como distancia de un punto a un grupo.
%
%	d = dg_means2(x,y)
%
%         x  = conjunto de patrones del grupo 1 (por columnas)
%         y  = conjunto de patrones del grupo 2 (por columnas)
%         d  = Distancia del grupo 1 al grupo 2

%	Copyright (c) Pedro L. Galindo (1998)

M = size(x,2);
N = size(y,2);

if M==1,
   media1=x;
else,
   media1=meanpat(x);
end;

if N==1,
   media2=y;
else,
   media2=meanpat(y);
end;


d=sqrt(M*N/(M+N)) * d_euclid(media1,media2);
