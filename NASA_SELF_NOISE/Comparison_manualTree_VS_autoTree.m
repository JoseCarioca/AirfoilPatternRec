close all, clc;

%% Libreria, datos y variables entrega
addpath("./pattern");
data = readtable("AirfoilSelfNoise.csv");
images_corr = false; %flag para mostrar e imprimir los mapas de correlacion
images_plot = true; %flag para mostrar e imprimir las gráficas
CV = 10; %iteraciones Cross Validation
%Errores de cada entrenamiento (duplicated a ver)
ErrTree = zeros(1,CV);

ErrTreeManual = zeros(1,CV);


x = table2array(data(:, 1:5))';
d = table2array(data(:, :))';


%% Preproccess
unique_chord      = unique(x(3, :));
unique_chord_data = [];
unique_chord = sort(unique_chord);
for chord = unique_chord
    unique_chord_data = [unique_chord_data d(:, d(3, :) == chord) ];
end

unique_angle    = unique(x(2, :));
unique_velocity = unique(x(4, :));

%% Aplicamos strouhal 
%  Con esto podemos aproximar un SSPL
% Fluid properties (for air at standard conditions)
rho = 1.225; % kg/m^3 (density of air)
mu = 1.81e-5; % Pa.s (dynamic viscosity of air)
c_air = 343; % m/s (speed of sound in air at 20 degrees Celsius)
USSF = @(f, St) (1 / f) * (f / St)^(-5/3) * (1 - exp(-f / St));
for it = 1:size(unique_chord_data, 2)
    f_d = unique_chord_data(1, it);
    c_d = unique_chord_data(3, it);
    v_d = unique_chord_data(4, it);
    a_d = unique_chord_data(2, it);
    s_d = unique_chord_data(5, it);
    if a_d <= 0
        a_d = 1;
    end

    M = v_d / c_air;
    St = (f_d * c_d) / v_d;
    SPL = 10 * log10(USSF(f_d, St) * (v_d / c_air) * (c_d / s_d));
    unique_chord_data(7, it) = SPL - 10 .* log10((v_d/100).^5 .* ((f_d .* c_d)/s_d.^2));
end

%1 = frec 2=angle 3= chord 4= vel 5 =thickness
x = unique_chord_data(1:5, :);
pca_transform = pca(unique_chord_data([2 5], :), 1);
x(6, :) = pca_transform * unique_chord_data([2 5], :);
x(7, :) = unique_chord_data(7, :);
y = unique_chord_data(6, :);

for i=1:CV
    [tr_x, ts_x, tr_y, ts_y] = crossval(x, y, CV, i);

    %% manual vs auto tree
    %surrogate off + bag 17 ; sur on + bag 16.9
    tree = RegressionTree.template('Reproducible',true); %prueba no se de estos params
    Mdl = fitrensemble(tr_x',tr_y','OptimizeHyperparameters','auto','Learners',tree);
    tree2 = RegressionTree.template('Surrogate','on','MaxNumSplits',1,'MinLeaf',1,'PredictorSelection','interaction-curvature'); %prueba no se de estos params
    Mdl2 = fitrensemble(tr_x',tr_y','Method','Bag','NumLearningCycles',100);
    ypTree = predict(Mdl,  ts_x');
    % parametros como predictorSelector, surrogate o NumCycles > 100
    % parecen no sonseguir una mejora significariva
    ErrTree(i) = MSE(ts_y, ypTree');
    disp("Reg Tree Auto:"+ ErrTree(i));

    ErrTreeManual(i) = MSE(ts_y, predict(Mdl2,  ts_x')');
    disp("Manual Reg Tree:"+ ErrTree(i));
end
figure,
plot(ErrTree);
xlabel("iteration"); ylabel("Erro"); hold on;
plot(ErrTreeManual);  legend("Auto","Manual");
hold off;
