clear all, close all, clc;

addpath("./pattern");

data = readtable("AirfoilSelfNoise.csv");

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

% Strouhal number
% -10 .* log((unique_velocity(1)/100).^5 .* f_d .* chord)

%% Poner a 1 si quieres sacar las imagenes y guardarlas
if 0
    ShowFigures(unique_chord_data, unique_chord, unique_velocity, unique_angle);
end

%% Aplicamos strouhal 
%  Con esto podemos aproximar un SSPL
for it = 1:size(unique_chord_data, 2)
    f_d = unique_chord_data(1, it);
    c_d = unique_chord_data(3, it);
    v_d = unique_chord_data(4, it);
    unique_chord_data(7, it)=  -10 .* log((v_d/100).^5 .* f_d .* c_d);
end

CV = 10;

x = unique_chord_data(1:4, :);
pca_transform = pca(unique_chord_data([2 5], :), 1);
x(5, :) = pca_transform * unique_chord_data([2 5], :);
x(6, :) = unique_chord_data(7, :);
y = unique_chord_data(6, :);

for i=1:CV
    [tr_x, ts_x, tr_y, ts_y] = crossval(x, y, CV, i);

    % A = ones(size(tr_x, 2), size(tr_x, 1) + 1);
    % j = size(tr_x, 1);
    % for ii = 1:size(tr_x, 1)
    %     A(:, ii) = tr_x(j, :);
    %     j = j - 1;
    % end
    
    A = [ (tr_x(6, :)').^5 tr_x(4, :)'.^4 tr_x(3, :)'.^3 (tr_x(5, :)') tr_x(1, :)' ones(size(tr_x, 2), 1)];

    p = pinv(A) * tr_y';

    y_pred = ts_x(6, :).^5 .* p(1) + ts_x(4, :).^4 .* p(2) + ts_x(3, :) .^ 3 .* p(3) + ts_x(5, :) .* p(4) + ts_x(1, :) .* p(5) + p(6);
    figure;
    screen_size = get(0, 'ScreenSize');
    set(gcf, 'Position', screen_size);
    plot(ts_y, 'r'); hold on; plot(y_pred, 'b');
    xlabel("Frequency"); ylabel("SSPL");
    legend('Test', 'Prediction');hold off;
    print("RegressionIt"+i+".png", '-dpng', '-r300')

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

