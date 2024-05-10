function d = d_minkow(x,y)
% D_BHATTA - Distancia de Bhattacharyya de un conjunto de vectores(x) a un vector(y)
%
%	d = d_bhatta(x,y)
%
%         x = conjunto de patrones (por columnas)
%         y = vector de referencia
%         d = vector fila con las distancias de cada vector de x a y

%	Copyright (c) Pedro L. Galindo (1998)

if size(x,1)~=size(y,1),
   error('Parametros incorrectos');
end;

aux=sqrt(x)-sqrt(y)*ones(1,size(x,2));
d=sqrt(sum(aux.*aux));