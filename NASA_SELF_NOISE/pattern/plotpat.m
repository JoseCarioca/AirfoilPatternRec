function plotpat(x,y,cadena)
% PLOTPAT  - Dibuja un conjunto de pares x/y de patrones bidimensionales
%
%    plotpat(x);             
%    plotpat(x,y);
%    plotpat(x,'.g');
%    plotpat(x,y,'x'); En este caso, cadena solo puede ser 'o' , 'x' , '*' , ...
%  

%	Copyright (c) Pedro L. Galindo (1998)

if nargin==1,
   caso=1;
   cadena='og';
elseif nargin==2
   if ~isstr(y),
      caso=2;
      cadena='o';
   else,
      caso=1;
      cadena=y;
   end;
elseif nargin==3,
   caso=2;
else,
   error('Numero de argumentos invalido');
end;

% s=rand('state');
% rand('seed',3);
h=[hsv;hot;gray];
h=h(1:7:64*2,:);
h=shuffle(h')';
colores=[1 0 0;0 1 0;0 0 1;1 1 0;1 0 1;0 1 1;0.5 0.5 0.5;1 0.5 0;0.5 0 1;...
         1 0 0.5;0 0.5 1;h];
% rand('state',s);

if caso==2,
   if size(x,1)==1,
      y = y - min(y);
      numclases = max(y);
      
      hold on;
      for i=0:numclases,
         indices = find(y==i);
         h = plot(x(1,indices),zeros(1,length(indices)),cadena);
         indice=1+mod(i,length(colores));
         set(h,'Color',colores(indice,:));
      end;
      hold off;
   else,
      y = y - min(y);
      numclases = max(y);

      hold on;
      for i=0:numclases,
         indices = find(y==i);
         h = plot(x(1,indices),x(2,indices),cadena);
         indice=1+mod(i,length(colores));
         set(h,'Color',colores(indice,:));
     end;
      hold off;
   end;
else,
   plot(x(1,:),x(2,:),cadena);
end;
