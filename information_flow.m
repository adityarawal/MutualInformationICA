%Highway network
clear all; close all;clc;
dirname = 'logs/resnet/';
F = dir(strcat(dirname, '1475883330_*.txt'));
h = {};
for i = 1:length(F)
    h{i} = dlmread(strcat(dirname, F(i).name));
end

%%Node v/s layer mutual information
%%Topmost (final layer) is softmax with size = number of classes = 10
% MImatrix = zeros(length(h)-1, size(h{2},2));
% for i = length(h):-1:2
%     i
%     t = h{i};
%     for j = 1:size(t,2)
%         MImatrix(length(h)-i+1,j) = MIxnyn(h{1}', t(:,j)');
%     end
% end

%%Layer v/s layer mutual information
%%Topmost layer (ignored here) is softmax with size = number of classes = 10

layerMI = zeros(length(h)-1, 1); %Ignoring the topmost layer
count = 1;
for i = length(h)-1:-1:1
    i
    layerMI(count) = MIxnyn(h{1}',h{i}')
    count = count+1 ;
end

imagesc(layerMI);            %# Create a colored plot of the matrix values
colorbar
colormap(flipud(summer));  %# Change the colormap to gray (so higher values are
title('resnet')
f = 0;

