function [y,n] = moda(x)
% MODA     - Moda de un vector
%
%    La moda es el número que más se repite en x
%
%    [y,n]=moda(x);
%
%    n es el numero de veces que aparece dicho numero
%

%	Copyright (c) Pedro L. Galindo (1998)

x=sort(x);
valores = x(find(diff(x)~=0));
valores=[valores x(length(x))];
total(length(valores))=0;
for i=1:length(valores),
   total(i)=length(find(x==valores(i)));
end;

posiciones=cumsum(total);
[valor,posic]=max(total);

y=x(posiciones(posic));
n = valor;
