clear all, close all, clc;
addpath("pattern\");

directory       = "./harth12";
files           = dir(directory);

total_table = table();

%% Procesamos cada fichero
for f = 3:length(files)
    data = readtable(fullfile(directory, files(f).name));
    % conversion a scalar
    deis = data.timestamp.Hour * 60 * 60 + data.timestamp.Minute * 60 + data.timestamp.Second;
    data.timestamp = deis;

    if any(data.Properties.VariableNames == "Var1")
        data = removevars(data, "Var1");
    end
    if any(data.Properties.VariableNames == "index")
        data = removevars(data, "index");
    end    
    total_table = [total_table; data];

end

writetable(total_table, "ALL_FILES12.csv");

