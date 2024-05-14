clearvars -except d1 d2,
close all, clc;
%addpath("./pattern");
% Load data from CSV files
d1 = readtable('student-mat.csv');d2 = readtable('student-por.csv');

% Specify variables for merging
mergeVars = {'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', ...
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet'};

% Merge the tables
d3 = [d1; d2];
d3.school = replace(d3.school, {'GP','MS'},{'0','1'}); %
d3.sex = replace(d3.sex, {'F','M'},{'0','1'}); %female or male
d3.address = replace(d3.address, {'U','R'},{'0','1'}); %urban or rural
d3.famsize = replace(d3.famsize, {'LE3','GT3'},{'0','1'}); %<=3 or >3 members
d3.Pstatus = replace(d3.Pstatus, {'A','T'}, {'0','1'}); %living apart or together (parents)
% cols 9,10,11,12 vamos a dejarlas de momento
d3.schoolsup = replace(d3.schoolsup, {'no','yes'},{'0','1'}); %extra school support
d3.famsup = replace(d3.famsup, {'no','yes'},{'0','1'}); %family educ. support
d3.paid = replace(d3.paid, {'no','yes'},{'0','1'}); %extra paid lessons
d3.activities = replace(d3.activities, {'no','yes'},{'0','1'});
d3.nursery = replace(d3.nursery, {'no','yes'},{'0','1'});
d3.higher = replace(d3.higher, {'no','yes'},{'0','1'}); %wants to take higher edu
d3.internet = replace(d3.internet, {'no','yes'},{'0','1'});
d3.romantic = replace(d3.romantic, {'no','yes'},{'0','1'});
%[d3.age, ~] = grp2idx(d3.age);
%[uniqueRows, ~, idx] = unique(d3(:,[1:17 19:end]));
% Display number of rows
disp(size(d3, 1)); % Should display 382 students -> 662 uniques
d3Incomp = d3(:,[1:8 13:end]);

for i = 1:size(d3Incomp,2)
    celda(:,i) = str2double(d3Incomp{:,i});
    if(isnan(celda(:,i)))
        celda(:,i) = d3Incomp{:,i};
    end
end
d4 = unique(d3(:,[1:12 22])); %segun los researchers columnas clave para datos (quita datos repetidos)
writetable(d3(:,[1:8 13:end]),"AlcoholSetIncomplete.csv");

save("ValoresAlcohol.mat","celda");