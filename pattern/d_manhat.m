function d = d_manhat(x, y)
% D_MANHAT - Distancia de Manhattan de un conjunto de vectores(x) a un vector(y)
%
%	d = d_manhat(x,y)
%
%         x = conjunto de patrones (por columnas)
%         y = vector de referencia
%         d = vector fila con las distancias de cada vector de x a y

%	Copyright (c) Pedro L. Galindo (1998)

if size(x,1)~=size(y,1),
   error('Parametros incorrectos');
end;

aux=x-y*ones(1,size(x,2));
d=sum(abs(aux));