clear all; close all;clc;
dirname = 'logs/resnet/';
scaling_factor = '2.0';
fname = strcat(scaling_factor, '_hidden_grads_*.txt');
F = dir(strcat(dirname, fname));

h = {};
for i = 1:length(F)
    h{i} = dlmread(strcat(dirname, F(i).name));
end

max_grad_value = 100; %For plotting finite values, upper bound the gradient value

num_layers = length(h);
num_samples = length(h{1});

grad_mat = zeros(num_samples,num_layers);
for i=1:num_layers
    h_mat = abs(h{i}); %num_samples x num_node per layer
    grad_mat(:,i) = min(max_grad_value, max(h_mat,[],2)); %Find maximum value of gradient in a layer and upper bound it
end

figure;
hold on;
layer_num = []
for i=1:2:num_layers
    plot(1:num_samples,grad_mat(:,i), 'LineWidth', 2);
    layer_num = [layer_num i];
end
legendCell = cellstr(num2str(layer_num','layer=%d'));
legend(legendCell);
title(['Max Layer Gradient in Resnet with Scaling Factor ' scaling_factor]);
xlabel('Number of Training Samples', 'fontsize', 10);
ylabel('Maximum Absolute gradient value in a layer', 'fontsize', 10);
t = 0;