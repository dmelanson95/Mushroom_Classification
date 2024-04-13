%% Mushroom Classification
% By David Melanson
% Dataset found at https://www.kaggle.com/datasets/vishalpnaik/mushroom-classification-edible-or-poisonous/data

clc; clear all; close all;
dir = 'C:\Users\dpfab\Documents\School\S2024\Neural Networks';

%% Read dataset
data = readtable(fullfile(dir, 'mushroom_nums.csv'));

%split class column into two for those poisonous and those edible
p = data(:, 1) == 0;
p.Properties.VariableNames = "Poisonous";
e = data(:, 1) == 1;
e.Properties.VariableNames = "Edible";
results = [p e];

% class {'p': 0, 'e': 1}
% cap-shape {'b': 0, 'c': 1, 'p': 2, 'o': 3, 'f': 4, 'x': 5, 's': 6}
% cap-color {'w': 0, 'y': 1, 'e': 2, 'b': 3, 'n': 4, 'k': 5, 'o': 6, 'u': 7, 'l': 8, 'r': 9, 'g': 10, 'p': 11}
% does-bruise-or-bleed {'t': 0, 'f': 1}
% gill-color {'b': 0, 'w': 1, 'y': 2, 'e': 3, 'n': 4, 'k': 5, 'o': 6, 'u': 7, 'f': 8, 'r': 9, 'g': 10, 'p': 11}
% stem-color {'b': 0, 'w': 1, 'y': 2, 'e': 3, 'n': 4, 'k': 5, 'o': 6, 'u': 7, 'l': 8, 'f': 9, 'r': 10, 'g': 11, 'p': 12}
% has-ring {'f': 0, 't': 1}
% habitat {'w': 0, 'h': 1, 'u': 2, 'l': 3, 'm': 4, 'g': 5, 'p': 6, 'd': 7}
% season {'a': 0, 'w': 1, 's': 2, 'u': 3}

%% partition data for training and testing
n = height(data);
hpartition = cvpartition(n, 'Holdout', 0.1);
idxTrain = training(hpartition);
train_data = table2array(data(idxTrain, 2:12))';
train_result = table2array(results(idxTrain, :))';
idxTest = test(hpartition);
test_data = table2array(data(idxTest, 2:12))';
test_result = table2array(results(idxTest, :))';

%% initialize network

net = feedforwardnet([50 25 11]);
net.trainFcn = 'trainscg';
net.divideFcn = '';
net.trainParam.show = 10;
net.trainParam.epochs = 2000;
net.trainParam.goal = 0;
net = configure(net, train_data, train_result);
view(net);

%% train network
net = train(net, train_data, train_result);

%% calculate accuracy
[~, I] = max(net(test_data));
I = I-1;
[~, t_I] = max(test_result);
t_I = t_I-1;
error = 100-(100*sum(I == t_I)/numel(I));
fprintf('error = %.2f%%\n\r', error)