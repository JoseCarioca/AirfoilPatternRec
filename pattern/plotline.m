function plotline(W,cadena);
% PLOTLINE - Dibuja una recta, dada su ecuacion
%
%     plotline(W);
%     plotline(W,cadena);
%
%          Si W es un vector de longitud 2, W=[m b], y dibuja la recta y=mx+b;
%          Si W es un vector de longitud 3, W=[a b c], y dibuja la recta ax+by+c=0;
%

%	Copyright (c) Pedro L. Galindo (1998)

if nargin==1,
   cadena='w';
end;
   
if length(W)==2,
   m=W(1);
   b=W(2);
else,
   a=W(1);
   b=W(2);
   c=W(3);
end;

x=-10:0.1:10;

if length(W)==2,
   y=m*x+b;
   indices=find(abs(y)<=10);
   x=x(indices);
   y=y(indices);
else
   if b==0,
      y=-10:0.1:10;
      x=-c/a*ones(size(x));
   else,
      y=(-c-a*x)/b;
      indices=find(abs(y)<=10);
      x=x(indices);
      y=y(indices);
   end;
end;

plot(x,y,cadena);