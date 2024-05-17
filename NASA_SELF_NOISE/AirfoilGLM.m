clear all, close all, clc;

addpath("./pattern");

data = readtable("AirfoilSelfNoise.csv");

correlacion = corrcoef(data{:,:});
figure,
heatmap(data.Properties.VariableNames,data.Properties.VariableNames,correlacion);
title('Correlation map of given data');
print("CorrelationMap0.png", '-dpng', '-r300')
 CV = 10;
for i=1:CV
    [tr_x, ts_x, tr_y, ts_y] = crossval(data{:,1:end-1}', data{:,end}', CV, i);

    %% REGRESSION TREE

    tree = fitrtree(data);

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
    % 
    % A = [ tr_x' ones(size(tr_x, 2), 1)];
    % 
    % p = pinv(A) * tr_y';
    % 
    % y_pred = [ ts_x' ones(size(ts_x, 2), 1)]*p;
    % 
    % disp("RMSE INV original data: " +  MSE(ts_y, y_pred'));

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
if 0
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

CV = 10;
%1 = frec 2=angle 3= chord 4= vel 5 =thickness
x = unique_chord_data(1:5, :);
pca_transform = pca(unique_chord_data([2 5], :), 1);
x(6, :) = pca_transform * unique_chord_data([2 5], :);
x(7, :) = unique_chord_data(7, :);
y = unique_chord_data(6, :);

corr_table = corrcoef([x([1 3 4 5 6 7],:)' y']); %correlacion de los datos contando con la salida
disp(corr_table);
%% alpha y delta estan transformados... ver el significado y editar mapa acorde
figure,
heatmap(corr_table); %, 'Colormap', 'jet', 'ColorLimits', [-1, 1], 'CellLabelColor', 'none'
title('Mapa de correlacion');

ErrInv = zeros(1,CV); ErrInv2 = zeros(1,CV); ErrFitlm = zeros(1,CV); 

corr_table = corrcoef([x' y']); %correlacion de los datos contando con la salida
disp(corr_table);

% Create heatmap
xtitle = {'Freq' 'Angle Attack' 'Chord Length' 'Velocity' 'Section Side Disp' 'Angle SSD Transform' 'SPL Form' 'SSPL'};
% alpha y delta estan transformados... ver el significado y editar mapa acorde
heatmap(xtitle,xtitle,corr_table); %, 'Colormap', 'jet', 'ColorLimits', [-1, 1], 'CellLabelColor', 'none'
title('Mapa de correlacion');

%% Store errors and their coefficients:
pinv_error_coefs = cell(1, CV);
regr_error_coefs = cell(1, CV);
fitl_error_coefs = cell(1, CV);

for i=1:CV
    [tr_x, ts_x, tr_y, ts_y] = crossval(x, y, CV, i);
    
    A = [ tr_x(7, :)' tr_x(4, :)' tr_x(3, :)' tr_x(1, :)' (tr_x(5, :)') (tr_x(6, :)') (tr_x(2, :)') ones(size(tr_x, 2), 1)];

    p = pinv(A) * tr_y';

    y_pred = ts_x(7, :) .* p(1) + ts_x(4, :) .* p(2) + ts_x(3, :) .* p(3) + ts_x(1, :) .* p(4) + ts_x(5, :) .* p(5) + ts_x(6, :) .* p(6) + ts_x(2, :) .* p(7) + p(8);
    ErrInv(i) = MSE(ts_y, y_pred);
    disp("RMSE (Our PINV): " + ErrInv(i) );

    pinv_err = MSE(ts_y, y_pred);

    pinv_error_coefs{i} = {pinv_err, p};

    disp("RMSE (Our PINV): " +  pinv_err);

    if 1
        figure;
        screen_size = get(0, 'ScreenSize');
        set(gcf, 'Position', screen_size);
        plot(ts_y, 'r'); hold on; plot(y_pred, 'b');
        xlabel("Frequency"); ylabel("SSPL");
        legend('Test', 'Prediction');hold off;
        print("RegressionIt"+i+".png", '-dpng', '-r300')
    end


    A2 = [ tr_x' ones(size(tr_x, 2), 1)];

    p = pinv(A2) * tr_y';

    y_pred = [ ts_x' ones(size(ts_x, 2), 1)]*p;
    ErrInv2(i) = MSE(ts_y, y_pred');
    disp("RMSE INV original data: " +  ErrInv2(i));

    % lda = fitcdiscr( tr_x', tr_y');
    % lda_pred = predict(lda, ts_x')';
    % 
    % disp("RMSE (fitcdiscr):" + MSE(ts_y, lda_pred));

    mdl    = regress(tr_y', tr_x', 0.1);
    r_pred = ts_x(1, :) .* mdl(1) + ts_x(2, :) .* mdl(2) + ts_x(3, :) .* mdl(3) + ts_x(4, :) .* mdl(4) + ts_x(5, :) .* mdl(5) + ts_x(6, :) .* mdl(6) + ts_x(7, :) .* mdl(7);

    r_pred = (y_pred + r_pred) ./ 2;

    regress_err = MSE(ts_y, r_pred);

    regr_error_coefs{i} = {regress_err, mdl};

    disp("RMSE (Regress):" + regress_err);

    if 1
        figure;
        screen_size = get(0, 'ScreenSize');
        set(gcf, 'Position', screen_size);
        plot(ts_y, 'r'); hold on; plot(r_pred, 'b');
        xlabel("Frequency"); ylabel("SSPL");
        legend('Test', 'Prediction');hold off;
        print("RegressIt"+i+".png", '-dpng', '-r300')
    end


    %pp     = fitlm(tr_x', tr_y', 'quadratic');

    pp     = fitlm(tr_x', tr_y', 'quadratic');

    %p_pred = predict(pp,  ts_x');

    %disp("RMSE (PolyFit): " + MSE(ts_y, p_pred'));

    fitlm_err = MSE(ts_y, p_pred');

    fitl_error_coefs{i} = {fitlm_err, pp};

    disp("RMSE (PolyFit): " + fitlm_err);

    %prueba fitlm sin alpha 'duplicado'
    y_fitlmPred = predict(fitlm(tr_x([1 3 4 5 6 7],:)', tr_y', 'quadratic'),  ts_x([1 3 4 5 6 7],:)');
    ErrFitlm(i) = MSE(ts_y, y_fitlmPred');
    disp("RMSE (PolyFit) sin alpha: " + ErrFitlm(i));
    if 0
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
        
    % ts_unique_chord      = unique(ts_x(3, :));
    % ts_unique_chord_data = [];
    % ts_unique_chord = sort(ts_unique_chord);
    % ts_data = ts_x;
    % ts_data(7, :) = ts_y;
    % for chord = ts_unique_chord
    %     ts_unique_chord_data = [ts_unique_chord_data ts_data(:, ts_data(3, :) == chord) ];
    % end
    % 
    % ts_unique_angle    = unique(ts_x(2, :));
    % ts_unique_velocity = unique(ts_x(4, :));
    % for chord = ts_unique_chord
    %     for v = ts_unique_velocity
    %         for a = ts_unique_angle
    %             values = ts_unique_chord_data(:, ts_unique_chord_data(3, :) == chord);
    %             values = values(:, values(4, :) == v);
    %             values = values(:, values(2, :) == a);
    %             if ~isempty(values)
    %                 y_d    = values(7, :);
    %                 y_pred = values(6, :).^5 .* p(1) + values(4, :).^4 .* p(2) + values(3, :) .^ 3 .* p(3) + values(5, :).^2  .* p(4) + values(1, :) .* p(5) + p(6);
    %                 figure, plot(y_d, 'r'); hold on; plot(y_pred, 'b');
    %                 xlabel("Frequency"); ylabel("SSPL");
    %                 legend('Test', 'Prediction');hold off;
    %             end
    %         end
    %     end
    % end
end

figure,
plot(Er_rel*100); xlabel("iteration"); ylabel("Error relativo");

