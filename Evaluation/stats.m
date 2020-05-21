clc;clear;format compact;
file = 'semismart_vs_random.txt';
disp(file)
data = textread(file) ./ 1000;

x  = (data - mean(data)) / std(data); % std

[h,p] = kstest(x);% Could not reject normality.´
if h == 0
    disp('Normality could not be rejected')
else
    disp('Normality is rejected!')
end
mu0 = 0.50;
[h,p] = ttest(data - mu0);
if h == 0
    disp('Results are insignificant')
else
    disp('Results are significant!')
end