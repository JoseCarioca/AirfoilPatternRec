function centroides = lvq1(x,y,centroides,alpha,numiterac)
% LVQ1     - Metodo LVQ1
%
%    centroides=lvq1(x,y,centroides,alpha,numiteraciones);
%

%	Copyright (c) Pedro L. Galindo (1998)

if nargin<5,
   numiterac=1;
end;
if nargin<4,
   alpha=0.1;
end;
if nargin<3,
   error('Error en el numero de argumentos');
end;

numcentroides = size(centroides,2);
numpatrones = size(x,2);

clasec = setclass(x,y,centroides);

for i=1:numiterac,
   [micentro,d]=mycenter(x,centroides);
   deltacentroide=zeros(size(centroides));
   factor(numcentroides)=0;
   for j=1:numpatrones,
      patron=x(:,j);
      p=micentro(j);
      clase_ganadora = clasec(micentro(j));
      if y(j)==clase_ganadora,  % Clasificacion correcta
         deltacentroide(:,p)= deltacentroide(:,p) + (patron - centroides(:,p));
      else,                     % Clasificacion erronea
         deltacentroide(:,p)= deltacentroide(:,p) - (patron - centroides(:,p));      
      end;
      factor(p)=factor(p)+1;
   end;
   centroides = centroides + alpha * deltacentroide / factor(p);
end;