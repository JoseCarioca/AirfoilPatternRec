close all, clc;

%% Libreria, datos y variables entrega
addpath("./pattern");
data = readtable("AirfoilSelfNoise.csv");
images_corr = false; %flag para mostrar e imprimir los mapas de correlacion
images_plot = true; %flag para mostrar e imprimir las gráficas
CV = 10; %iteraciones Cross Validation
%Errores de cada entrenamiento (duplicated a ver)
ErrTree = zeros(1,CV);
ErrInv = zeros(1,CV); 
ErrInv2 = zeros(1,CV); 
ErrFitlm = zeros(1,CV); 
%% Store errors and their coefficients:
pinv_error_coefs = cell(1, CV);
regr_error_coefs = cell(1, CV);
fitl_error_coefs = cell(1, CV);


%% Mapa de Correlacion de los datos originales
if images_corr
correlacion = corrcoef(data{:,:});
figure,
heatmap(data.Properties.VariableNames,data.Properties.VariableNames,correlacion);
title('Correlation map of given data');

    print("CorrelationMap0.png", '-dpng', '-r300')
end


%% Labels
%  Frequency
%  Angle of attack
%  Chord length
%  Free-stream velocity
%  Section-side displacement thickness
%  Scaled Sound Pressure Level

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

%% Poner a 1 si quieres sacar las imagenes y guardarlas
if images_corr
    ShowFigures(unique_chord_data, unique_chord, unique_velocity, unique_angle);
end

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

%% no se por que hay dos socorro
if images_corr
    corr_table = corrcoef([x([1 3 4 5 6 7],:)' y']); %correlacion de los datos contando con la salida
    disp(corr_table);
    figure,
    heatmap(corr_table); %, 'Colormap', 'jet', 'ColorLimits', [-1, 1], 'CellLabelColor', 'none'
    title('Mapa Correlacion con SPL Form');
end

if images_corr
    corr_table = corrcoef([x' y']); %correlacion de los datos contando con la salida
    disp(corr_table);
    % Create heatmap
    xtitle = {'Freq' 'Angle Attack' 'Chord Length' 'Velocity' 'Section Side Disp' 'Angle SSD Transform' 'SPL Form' 'SSPL'};
    heatmap(xtitle,xtitle,corr_table); %, 'Colormap', 'jet', 'ColorLimits', [-1, 1], 'CellLabelColor', 'none'
    title('Mapa de correlacion');
end

%% CROSS VALIDATION
for i=1:CV
    [tr_x, ts_x, tr_y, ts_y] = crossval(x, y, CV, i);

    %% REGRESSION TREE
    %surrogate off + bag 17 ; sur on + bag 16.9
    %tree = RegressionTree.template('Reproducible',true); %prueba no se de estos params
    %Mdl = fitrensemble(tr_x',tr_y','OptimizeHyperparameters','auto','Learners',tree);
    tree = RegressionTree.template('Surrogate','on','MaxNumSplits',1,'MinLeaf',1,'PredictorSelection','interaction-curvature'); %prueba no se de estos params
    Mdl = fitrensemble(tr_x',tr_y','Method','LSBoost','NumLearningCycles',200);
    ypTree = predict(Mdl,  ts_x');
    % parametros como predictorSelector, surrogate o NumCycles > 100
    % parecen no sonseguir una mejora significariva
    ErrTree(i) = MSE(ts_y, ypTree');
    ErrTree;
    disp("RMSE Reg Tree:"+ ErrTree(i));

    if images_plot
        figure;
        screen_size = get(0, 'ScreenSize');
        set(gcf, 'Position', screen_size);
        plot(ts_y, 'r'); hold on; plot(ypTree', 'b');
        xlabel("Frequency"); ylabel("SSPL");
        legend('Test', 'Prediction');
        print("RegTREE"+i+".png", '-dpng', '-r300')
    end

    %% PseudoInversa 
    A = [ tr_x(7, :)' tr_x(4, :)' tr_x(3, :)' tr_x(1, :)' (tr_x(5, :)') (tr_x(6, :)') (tr_x(2, :)') ones(size(tr_x, 2), 1)];

    p = pinv(A) * tr_y';

    y_pred = ts_x(7, :) .* p(1) + ts_x(4, :) .* p(2) + ts_x(3, :) .* p(3) + ts_x(1, :) .* p(4) + ts_x(5, :) .* p(5) + ts_x(6, :) .* p(6) + ts_x(2, :) .* p(7) + p(8);

    ErrInv(i) = MSE(ts_y, y_pred);
    disp("RMSE (Our PINV): " + ErrInv(i) );

    pinv_err = MSE(ts_y, y_pred);

    if images_plot
        figure;
        screen_size = get(0, 'ScreenSize');
        set(gcf, 'Position', screen_size);
        plot(ts_y, 'r'); hold on; plot(y_pred, 'b');
        xlabel("Frequency"); ylabel("SSPL");
        legend('Test', 'Prediction');hold off;
        print("RegressionIt"+i+".png", '-dpng', '-r300')
    end

    %% PseudoInversa 2 (la misma solucion)
    A2 = [ tr_x' ones(size(tr_x, 2), 1)];

    p = pinv(A2) * tr_y';

    y_pred2 = [ ts_x' ones(size(ts_x, 2), 1)]*p;

    ErrInv2(i) = MSE(ts_y, y_pred2');
    disp("RMSE INV metodo 2: " +  ErrInv2(i));

    %% Funcion Regress matlab
    mdl    = regress(tr_y', tr_x', 0.1);
    r_pred = ts_x(1, :) .* mdl(1) + ts_x(2, :) .* mdl(2) + ts_x(3, :) .* mdl(3) + ts_x(4, :) .* mdl(4) + ts_x(5, :) .* mdl(5) + ts_x(6, :) .* mdl(6) + ts_x(7, :) .* mdl(7);

    r_pred = (y_pred + r_pred) ./ 2;

    regress_err = MSE(ts_y, r_pred);

    regr_error_coefs{i} = {regress_err, mdl};

    disp("RMSE (Regress):" + regress_err);

    if images_plot
        figure;
        screen_size = get(0, 'ScreenSize');
        set(gcf, 'Position', screen_size);
        plot(ts_y, 'r'); hold on; plot(r_pred, 'b');
        xlabel("Frequency"); ylabel("SSPL");
        legend('Test', 'Prediction');hold off;
        print("RegressIt"+i+".png", '-dpng', '-r300')
    end

    pp     = fitlm(tr_x', tr_y', 'quadratic');

    p_pred = predict(pp,  ts_x');

    fitlm_err = MSE(ts_y, p_pred');

    fitl_error_coefs{i} = {fitlm_err, pp};

    disp("RMSE (PolyFit): " + fitlm_err);

    %prueba fitlm sin alpha 'duplicado'
    y_fitlmPred = predict(fitlm(tr_x([1 3 4 5 6 7],:)', tr_y', 'quadratic'),  ts_x([1 3 4 5 6 7],:)');
    ErrFitlm(i) = MSE(ts_y, y_fitlmPred');
    disp("RMSE (PolyFit) sin alpha: " + ErrFitlm(i));
    if images_plot
        figure;
        screen_size = get(0, 'ScreenSize');
        set(gcf, 'Position', screen_size);
        plot(ts_y, 'r'); hold on; plot(p_pred, 'b');
        xlabel("Frequency"); ylabel("SSPL");
        legend('Test', 'Prediction');hold off;
        print("PolyFitIt"+i+".png", '-dpng', '-r300')

        figure;
        screen_size = get(0, 'ScreenSize');
        set(gcf, 'Position', screen_size);
        plot(ts_y-p_pred', 'g');
        xlabel("Frequency"); ylabel("Error");
        legend('residuals');hold off;
        print("resfitlmIt"+i+".png", '-dpng', '-r300')
    end

end


%%  AQUI MOSTRAR RESULTADOS DE RMSE MEDIO ABSOLUTO RELATIVO, GRAFICAS...
%disp("RMSE suma TRee: " +  ErrTotalTree);
figure,
plot(ErrTree);
xlabel("iteration"); ylabel("Error Reg Tree"); hold on;
plot(ErrInv2);
plot(ErrFitlm);


%% ELASTIC NET ¿SE DEJA?
for i=1:CV
    [tr_x, ts_x, tr_y, ts_y] = crossval(data{:,1:end-1}', data{:,end}', CV, i);

    if 0

       %% Lambda pred: 
        %[X, mu, sigma] = zscore(x);
        alpha = 0.5;
    
        ErrElNet = zeros(1,100);
        % ElasticNet
        %[B, FitInfo] = lasso(X, y', 'Alpha', alpha);
        [B, FitInfo] = lassoglm(tr_x', tr_y','gamma','Alpha',0.5); % 10-fold cross-validation
        yPred = ts_x' * B; % matriz 150x100
        for j = 1:size(yPred,2)
            ErrElNet(j) = MSE(ts_y, yPred(:,j)');
            disp("RMSE Lasso:"+ ErrElNet(j));
        end
        [m,index] = min(ErrElNet);
        figure;
        screen_size = get(0, 'ScreenSize');
        set(gcf, 'Position', screen_size);
        plot(ts_y, 'r'); hold on; plot(yPred', 'b');
        xlabel("Frequency"); ylabel("SSPL");
        legend('Test', 'Prediction');
        print("RegElasticNet"+i+".png", '-dpng', '-r300')
    end

end


