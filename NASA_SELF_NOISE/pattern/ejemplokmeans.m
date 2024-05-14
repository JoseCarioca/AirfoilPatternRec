clear all
close all
N = 1000;
K = 7;

% Generacion de x
r = 4+rand(1,N);
alpha = 2*pi*rand(1,N);
x=[r.*cos(alpha);r.*sin(alpha)];

% Determinacion inicial de prototipos
c=x(:,1:K);

for i=1:100,
   plot(x(1,:),x(2,:),'.'), axis equal, hold on;
   plot(c(1,:),c(2,:),'or','LineWidth',3);hold off;
   
   % Determinacion del prototipo mas cercano a cada dato
   for k=1:K,
       d(k,:)=d_euclid(x,c(:,k));
   end;     
   [basura,prototipo]=min(d);
   
   % Actualizacion de los prototipos
   for k=1:K,
       c(:,k)=meanpat(x(:,find(prototipo==k)));
   end;   
   
   % Mostrar la suma total de las distancias
   title(num2str(sum(basura)))
   pause
end;
