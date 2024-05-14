function d = d_euclin(x, y, v)
% D_EUCLIN - Distancia euclidea normalizada por la raiz cuadrada de la varianza.
%            Tambien se denomina distancia Karl-Pearson.
%
%
%	d = d_euclin(x,y,varianzas)
%
%         x = conjunto de patrones (por columnas)
%         y = vector de referencia
%         varianzas = vector columna de varianza de cada componente
%         d = vector fila con las distancias euclideas normalizadas de cada vector de x a y

%	Copyright (c) Pedro L. Galindo (1998)


if size(x,1)~=size(y,1),
   error('Parametros incorrectos');
end;

aux=x-y*ones(1,size(x,2));

d=sum(aux.*aux./(sqrt(tocol(v))*ones(1,size(x,2))));
