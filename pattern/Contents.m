% Pattern Recognition Toolbox.
% Version 0.1  22-Oct-1999
% Copyright (c) Pedro L. Galindo Riaño
%
% Operaciones con un patrón
% -------------------------
%   torow    - Convierte un vector en un vector fila
%   tocol    - Convierte un vector en un vector columna
%
% Operaciones con una fila de salidas
% -----------------------------------
%   moda     - Calcula el elemento que más veces aparece
%
% Operaciones con un conjunto de vectores
% ---------------------------------------
%   shuffle  - Desordena un conjunto de vectores           
%   norma    - Calcula la norma de un conjunto de patrones
%
% Calculo de valores elementales
% ------------------------------
%   minpat   - Calcula el minimo de un conjunto de patrones
%   maxpat   - Calcula el maximo de un conjunto de patrones
%
% Operaciones elementales sobre un grupo de vectores
% -----------------------------------------------------------
%   addpat   - Suma a un grupo de vectores un vector dado
%   subpat   - Resta a un grupo de vectores un vector dado
%   mulpat   - Multiplica a un grupo de vectores por un vector dado
%   divpat   - Divide a un grupo de vectores por un vector dado
%   zeromean - Resta a un grupo de vectores su vector media
%   normalize - Normaliza un grupo de vectores al intervalo 0..1
%   denormalize - Realiza la operacion inversa a normalize
%
% Calculo de valores estadisticos
% -------------------------------
%   sumpat   - Calcula la suma de un conjunto de patrones
%   meanpat  - Calcula la media de un conjunto de patrones
%   stdpat   - Calcula la desv. standard de un conjunto de patrones
%   covpat   - Calcula la matriz de covarianza de un "  "   "
%
% Transformaciones de espacios
% ----------------------------
%   whiten   - Calcula la transf. de media 0 y covarianza unidad
%   fisher   - Calcula la transf. de Fisher(clasificacion optima)
%   pca      - Calcula la transf. PCA(varianza maxima)
%   scatter  - Cálculo de las matrices de dispersion(Sb, Sw, Sm)
%
% Funciones de dibujo
% -------------------
%   plotpat  - Dibuja un conjunto de patrones de varias clases        
%   plotline - Dibuja una recta, dada su ecuación
%   plotbon  - Dibuja la intersección de dos normales 2D
%   plotN2D  - Dibuja una normal 2D con una elipse de pdf fija
%   plotN3D  - Dibuja una normal 2D con varias elipses de pdf fija
%   plotquad - Dibuja una conica de coeficientes a,b,c,d,e,f 
%   bayesbon - Calcula la interseccion de dos normales 2D
%
% Generacion de nubes de puntos
% -----------------------------
%   randnorm - Genera una nube de puntos normal 2D
%
% Funciones de clasificacion
% --------------------------
%   clstats  - Calcula el nº de puntos, media y matriz de covarianza de cada clase
%   allclass - Calcula qué clases diferentes hay en un vector de salidas
%   myclass  - Extrae todos los vectores de una clase dada
%   setclass - Asigna a cada centroide (procedente de k-medias, por ej.) una clase
%   mycenter - Asigna cada vector al centroide más cercano (usa dist. euclidea)
%
% Algoritmos de clustering no supervisado
% ---------------------------------------
%   kmeans   - Algoritmo de las k-medias
%
% Algoritmos de clustering jerarquico
% -----------------------------------
%   cluster1 - Clustering aglomerativo jerarquico
%   cluster2 - Clustering divisivo jerarquico
%   cluster1 - Clustering max-min
%
% Funciones de distancia de un vector a un conjunto de vectores
% -------------------------------------------------------------
%   d_bhatta - Distancia de Bhattacharaya
%   d_coseno - Distancia coseno
%   d_euclid - Distancia euclidea
%   d_eucli2 - Cuadrado de la distancia euclidea
%   d_euclin - Distancia euclidea normalizada
%   d_mahal  - Distancia de Mahalanobis
%   d_manhat - Distancia de Manhattan
%   d_minkow - Distancia de Minkowsky
%
% Funciones de distancia intergrupos
% ----------------------------------
%   dg_avg   - Distancia media entre cada pareja de puntos
%   dg_max   - Distancia maxima entre cada pareja de puntos
%   dg_min   - Distancia minima entre cada pareja de puntos
%   dg_means - Distancia entre las medias de cada grupo
%   dg_means2 - Distancia ponderada entre las medias de cada grupo

% Distancia entre dos normales multivariadas
% ------------------------------------------
%   bhatta   - Distancia de Bhattacharaya entre dos normales multiv.
%   chernoff - Distancia de Chernoff entre dos normales multiv.
%   divergen - Calcula la divergencia entre dos normales multiv.
%
% Criterios de clustering
% -----------------------
%   J1       - Traza de Sw = Suma de errores al cuadrado
%   J2       - Criterio de Hotelling = (-1) * traza(inv(Sw)*Sb) 
%   J3       - Criterio = log(|C1|^n1 * |C2|^n2 * ...)
%   J4       - Criterio de Wilks = det(Sw)
%
% Calculo de outliers
% -------------------
%   chitest  - Determinacion de outliers segun el metodo chi-test
%
% Funciones varias
% ----------------
%   gauss    - Evalua una distribucion Gaussiana multivariada
%   dpr      - Distancia de un punto a una recta
%   circulo  - Devuelve N puntos del circulo de centro y radio dados
%   crossval - Subconjunto de patrones i-esimo de los N posibles
%   pseudinv - Pseudoinversa de x
%   parzen   - Ventana de Parzen
