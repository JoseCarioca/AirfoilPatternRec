function [] = ShowFigures(unique_chord_data, unique_chord, unique_velocity, unique_angle)
    for chord = unique_chord 
        figure;
        screen_size = get(0, 'ScreenSize');
        set(gcf, 'Position', screen_size);
        row = 1;
        for v = unique_velocity
            for a = unique_angle
                
                values = unique_chord_data(:, unique_chord_data(3, :) == chord);
                values = values(:, values(4, :) == v);
                values = values(:, values(2, :) == a);
                if ~isempty(values)
                    f_d = values(1, :);
                    y_d = values(6, :);
                    subplot(1, length(unique_velocity), row);
                    plot(f_d, y_d); hold on;
                end
            end
            title("Chord = " + chord + " Velocity = " + v);
            legend(arrayfun(@num2str, unique_angle, 'UniformOutput', false));
            xlabel("Frequency");
            ylabel("SSPL");
            hold off;
            row = row + 1;
        end
        print("Chord" + chord + ".png", '-dpng', '-r300')
    end
end

