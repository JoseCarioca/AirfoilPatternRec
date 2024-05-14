function plotquad(W,cadena,limites);
% PLOTQUAD - Dibuja la conica de coeficientes a,b,c,d,e,f 
%
%          plotquad(w);
%          plotquad(w,cadena);
%          plotquad(w,[-5:0.1:5]);
%          plotquad(w,cadena,[-5:0.1:5]);
%
%            w = [a b c d e f], donde   
%               a = termino en x^2
%               b = termino en y^2
%               c = termino en x*y
%               d = termino en x
%               e = termino en y
%               f = termino independiente
%            cadena = estilo de linea
%            [-5:0.1:5] : indica los limites en x, y la precision deseada
%            Los limites en y dependen de W y de los limites en x.

%	Copyright (c) Pedro L. Galindo (1998)

if nargin==1,
   cadena='g';
   limites = -5:0.005:5;
elseif nargin==2,
   if isstr(cadena),
      limites = -5:0.005:5;
   else,
      limites=cadena;
      cadena='g';
   end;
elseif nargin==3,
   if ~isstr(cadena),
      aux=cadena;
      cadena=limites;
      limites=aux;
   end;
else,
   help plotquad
   break;
end;

a=W(1);
b=W(2);
c=W(3);
d=W(4);
e=W(5);
f=W(6);

if a==0 & c==0 & d==0,
   x=limites;
   y=roots([b e f]);
   y = y ( find(y==real(y)) );
   if ishold,
      for i=1:length(y),
         plot(x,y(i)*ones(size(x)),cadena);
      end;
   else,
      hold on;
      for i=1:length(y),
         plot(x,y(i)*ones(size(x)),cadena);
      end;
      hold off;
   end;
   return
end;

if b==0 & c==0 & e==0,
   y=limites;
   x=roots([a d f]);
   x = x ( find(x==real(x)) );
   if ishold,
      for i=1:length(x),
         plot(x(i)*ones(size(y)),y,cadena);
      end;
   else,
      hold on;
      for i=1:length(x),
         plot(x(i)*ones(size(y)),y,cadena);
      end;
      hold off;
   end;
   return;
end;


% Resolucion de la grafica en el eje x
x=limites;

A = b;                   % Termino en y^2
B = c*x + e;             % Termino en y
C = a*x.^2 + d*x + f;    % Termino independiente de y

if A==0,
  if any(B==0),
    x = x(find(B~=0));
    B = B(find(B~=0));
    C = C(find(B~=0));
  end
  y1 = -C ./ B;
  y2 = y1;
  x1 = x;
  x2 = x;
else
  D = sqrt(B.^2 - 4*A*C);
  y1 = (-B+D)/(2*A);
  y2 = (-B-D)/(2*A);
  ind = find((real(y1)~=0)&(abs(imag(y1))< 1e-5));
  x1 = x(ind);
  x2 = x(ind);
  y1 = y1(ind);
  y2 = y2(ind);
end

n1 = length(x1);
n2 = length(x2);
[dummy1,break1] = max(x1(2:n1)-x1(1:n1-1));
[dummy2,break2] = max(x2(2:n2)-x2(1:n2-1));

if (dummy1<0.15 | dummy2<0.15),
  xn1 = x1;
  xn2 = x2;
  yn1 = y1;
  yn2 = y2;
else
  xn1 = [x1(1:break1) x2(break2:-1:1)];
  xn2 = [x1(n1:-1:break1+1) x2(break2+1:n2)];
  yn1 = [y1(1:break1) y2(break2:-1:1)];
  yn2 = [y1(n1:-1:break1+1) y2(break2+1:n2)];
end;

indices1=find(abs(yn1)<100);
indices2=find(abs(yn2)<100);

xn1=xn1(indices1);
yn1=yn1(indices1);

xn2=xn2(indices2);
yn2=yn2(indices2);



if length(yn1)>2,
   diferencias=abs(diff(yn1));
   dif_media = mean(diferencias);
   indices = find(diferencias>100*dif_media);
   yn1(indices)=yn1(indices)*nan;
end;

if length(yn2)>2,
   diferencias=abs(diff(yn2));
   dif_media = mean(diferencias);
   indices = find(diferencias>100*dif_media);
   yn2(indices)=yn2(indices)*nan;
end;


   [valor1,indice1]=min(xn1);
   [valor2,indice2]=min(xn2);

   [valor3,indice3]=max(xn1);
   [valor4,indice4]=max(xn2);

   xaux = [xn1 xn2];
   yaux = [yn1 yn2];

   margenx=(max(xaux)-min(xaux))/4;
   margeny=(max(yaux)-min(yaux))/4;

   if ishold,
      plot(xn1,yn1,cadena);
      plot(xn2,yn2,cadena);

      if abs(yn1(indice1)-yn2(indice2))<margeny & abs(valor2-valor1)<margenx,
         plot([valor1 valor2],[yn1(indice1) yn2(indice2)],cadena);
      end;
      if abs(yn1(indice3)-yn2(indice4))<margeny & abs(valor4-valor3)<margenx,
         plot([valor3 valor4],[yn1(indice3) yn2(indice4)],cadena);
      end;          
   else,
      hold on;
      plot(xn1,yn1,cadena);
      plot(xn2,yn2,cadena);

      if abs(yn1(indice1)-yn2(indice2))<margeny & abs(valor2-valor1)<margenx,
         plot([valor1 valor2],[yn1(indice1) yn2(indice2)],cadena);
      end;

      if abs(yn1(indice3)-yn2(indice4))<margeny & abs(valor4-valor3)<margenx,
         plot([valor3 valor4],[yn1(indice3) yn2(indice4)],cadena);
      end;
      hold off;
   end;


