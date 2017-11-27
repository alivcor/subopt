clear all
close all
rand('state', 10);
addpath(genpath('UGM'));                                                            
addpath(genpath('libsvm-3.21/matlab'));
addpath(genpath('SFO'))

% load cora.mat
load saved/alldata.mat
load saved/links.mat

% alldata = Feature;
% links = Link;
alldata = double(alldata);
links = double(links);
NFeat = size(alldata,2)-2;  % No. of features
NData = size(alldata,1);    % No. of data points

% Divide the data in train-test
ind = crossvalind('kfold', alldata(:,end), 10);

trainset = alldata(ind~=1,:);
testset = alldata(ind==1,:);

[accuracy_vec, n_label_vec] = get_performance(trainset, testset, links);
plot(cumsum(n_label_vec), accuracy_vec, '--o', 'LineWidth', 1);
xlabel('Number of manually labeled samples')
ylabel('Accuracy')