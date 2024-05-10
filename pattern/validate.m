function tasa = validate(x,y,metodo,n)
% VALIDATE - Metodos cross-validation & leave-one-out
%
%    tasa=validate(x,y,metodo,n);
%
%    Por ejemplo,
%
%         tasa = validate(x,y,'knnrate(x1,y1,x2,y2,7)',12);
%
%    Es obligatorio utilizar los nombres x1,y1,x2,y2
%    en la descripcion del metodo a utilizar.
%
%    Si n = numero de columnas de x, el metodo cross-validation
%    degenera en el metodo "leave-one-out"

%	Copyright (c) Pedro L. Galindo (1998)


tasamedia = 0;

ultimo = size(x,2);

inicio=1;
incremento = round(ultimo / n);

for i=1:n,
   fin = min(inicio + incremento - 1,ultimo);
   indices = inicio:fin;
   indicestrn = [1:inicio-1 fin+1:ultimo];
   indicestst = inicio:fin;
   x1 = x(:,indicestrn);
   y1 = y(:,indicestrn);
   x2 = x(:,indicestst);
   y2 = y(:,indicestst);
   eval(['tasaparcial=' metodo ';']);
   tasamedia = tasamedia + tasaparcial;
   inicio = inicio + incremento;
end;

tasa = tasamedia / n;
