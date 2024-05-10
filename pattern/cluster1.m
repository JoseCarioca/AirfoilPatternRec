function c = cluster1(x,k,particion,distancia)
% CLUSTER1 - Algoritmo de clustering aglomerativo jerarquico   !! O(n^2) !!
%            "Basic Agglomerative Hierarchical Clustering"
%
%       clases = cluster1(x,k)
%       clases = cluster1(x,k,p0) 
%       clases = cluster1(x,k,'d_max');
%       clases = cluster1(x,k,p0,'d_min')
%
%         x      = conjunto de patrones
%         P0     = particion inicial
%         k      = numero de clusters a localizar
%         clases = categoria de cada patron de x
%
%       Como distancia es posible usar cualquier distancia intergrupo, es
%       decir, dg_min, dg_max, dg_avg, dg_mean, dg_mean2

%	Copyright (c) Pedro L. Galindo (1998)

if nargin==2,
   distancia='dg_max';
   particion=1:size(x,2);   % Cada patron es una clase
elseif nargin==3,
   if isstr(particion),
      distancia=particion;
      particion=1:size(x,2);
   else,      
      distancia='dg_max';
   end;
elseif nargin~=4,
   help cluster1;
   error('Parametros incorrectos')
end;

[d,n] = size(x);
nc = k;
c = particion;
nk = n;

while nk>nc,
  %-----------------------------------------------------
  % Buscar los dos clusters más cercanos
  %-----------------------------------------------------
  dmin = 1e20;
  for i=1:nk-1,
    for j=i+1:nk,
      d = feval(distancia,myclass(x,c,i),myclass(x,c,j));
      if d<dmin, 
         dmin=d; 
         imin=i; 
         jmin=j; 
      end
    end
  end
  %-----------------------------------------------------
  % Mezclar los clusters imin y jmin
  %-----------------------------------------------------
  for i=1:n,
    if c(i)==jmin, 
       c(i)=imin; 
    end
    if c(i)==nk, 
       c(i)=jmin; 
    end
  end
  %-----------------------------------------------------
  % Decrementar el numero de clases
  %-----------------------------------------------------
  nk = nk - 1
end

return



if nargin==2,
   distancia='dg_max';
   p0=1:length(x);   % Cada patron es una clase
elseif nargin==3,
   if isstr(p0),
      distancia=p0;
      p0=1:length(x);
   else,      
      distancia='dg_max';
   end;
elseif nargin~=4,
   help cluster1;
   error('Parametros incorrectos')
end;

N = length(allclass(p0));

for numclases=N:-1:k+1,
   numclases
   minimoglobal = 1e10;

   categorias = allclass(p0);
   M = length(categorias);

   % Para cada pareja de clases, buscar la minima distancia de la clase C1 a la C2
   for i=1:M-1,
      indice1=find(p0==categorias(i));
      clase1=x(:,indice1);
      for j=i+1:M,
         indice2=find(p0==categorias(j));
         clase2=x(:,indice2);
         minimoparcial=feval(distancia,clase1,clase2);
         if minimoparcial<minimoglobal, 
            patron1=categorias(i);
            patron2=categorias(j);
            minimoglobal=minimoparcial;
         end;
      end;
   end
   % Fusionar las clases del patron1 y patron2
   indice2=find(p0==patron2);
   p0(indice2)=patron1*ones(1,length(indice2));   
end;

% Renumerar las clases de 1 hasta numclases
% =========================================
y=p0;
j=1;
for i=allclass(p0),
   indices=find(p0==i);
   y(indices)=j*ones(1,length(indices));
   j=j+1;
end;
