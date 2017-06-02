clear all; close all;clc;
format longg;

input_size = 784;
layer_size = [10 10 10];
T = 1.0;

target = [0 0 0 0 0 0 0 1 0 0]';
x = zeros(784,1);
x = dlmread('Images_1476828441_0.txt');

w1 = ones(input_size, layer_size(1))*1;
b1 = ones(layer_size(1),1)*1.0;

w2 = ones(layer_size(1), layer_size(2))*1;
b2 = ones(layer_size(2),1)*1.0;

w3 = ones(layer_size(2), layer_size(3))*1;
b3 = ones(layer_size(3),1)*1.0;

h1_raw = x'*w1 + b1';
h1 = max(0,h1_raw); %Relu

h2_raw = h1*w2 + b2';
h2 = (max(0,h2_raw))*T + h1*T; %Residual Network

h3_raw = h2*w3 + b3';
h3 = h3_raw; %Identity before softmax

y = softmax(h3');

dy = (y-target)'; %Num classes x 1

dw3 = h2' * dy;
db3 = dy;

dh2 = dy * w3';

dw2 = h1' * dh2;
db2 = dh2;





