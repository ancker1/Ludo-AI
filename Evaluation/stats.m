clc;clear;format compact;
file = 'QLearner_vs_semismart.txt';
disp(file)
data = textread(file) ./ 1000;
datatemp = data;
x  = (data - mean(data)) / std(data); % std

[h,p] = kstest(x);% Could not reject normality.´
if h == 0
    disp('Normality could not be rejected')
else
    disp('Normality is rejected!')
end
mu0 = 0.50;
[h,p] = ttest(datatemp - mu0);
if h == 0
    disp('Results are insignificant')
else
    disp('Results are significant!')
end