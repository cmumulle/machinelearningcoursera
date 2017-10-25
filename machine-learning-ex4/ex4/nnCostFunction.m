function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ===================================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ====================== Part 1 ===========================================
% Feedforward the neural network and return the cost in the variable J

I = eye(num_labels);        % Initialize the needed identity matrix
Y = zeros(m, num_labels);   % Initialize the vectorized examples
for i=1:m
  Y(i, :)= I(y(i), :);      % For all training examples, select the right row in the identity matrix
end

a1 = [ones(m, 1) X];        % Add ones to the X data matrix

z2 = a1 * Theta1';          % Calculate the hidden layer inputs
a2 = sigmoid(z2);           % Calculate the activation values

n = size(a2, 1);            % Number of rows in the a2 matrix
a2 = [ones(n, 1) a2];       % Add ones to the a2 data matrix

z3 = a2 * Theta2';          % Calculate output layer activation values
a3 = sigmoid(z3);           % Calculate output layer values
h = a3;                     % Assign the hypothesis

% Calculte regularization penalty
p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));
% Note: Theta1(:, 2:end) excludes the theta values for the bias at pos 1

% Calculate J, stepwise for the stupid
J0 = (-Y) .* log(h) - (1-Y) .* log(1-h);    % Calculate m x num_labels matrix of errors for each training set
J1 = sum(J0, 2);                            % Sum errors by row for each training set, giving an m x 1 matrix
J = sum(J1) / m + lambda * p / (2 * m);     % Sum over all rows of J1 and add regularization penalty

% ====================== Part 2 ===========================================
% backpropagation algorithm to compute the gradients Theta1_grad and Theta2_grad

delta1 = zeros(size(Theta1));       % Initialize delta1
delta2 = zeros(size(Theta2));       % Initialize delta2

for t = 1:m
% Go through all the training examples individually
    
    a1t = a1(t,:)' ;    % ALL FOLLOWING LINES ARE TRANSPOSED !!!
	a2t = a2(t,:)' ;
	z3t = a3(t,:)' ;
	Yt = Y(t,:)';       % Select the t-th training example, forward propagated values from before

	d3t = z3t - Yt;     % Calculate error in output layer, column vector with 10 elements

	z2t = [1; Theta1 * a1t];                        % Activation values in a column vector with 26 elements (incl. bias)      
	d2t = Theta2' * d3t .* sigmoidGradient(z2t);    % Theta2' * d3t gives column with 26 elements, element-wise multiplication

	delta1 = delta1 + d2t(2:end) * a1t';            % Skip bias element d0
	delta2 = delta2 + d3t * a2t';           

end;

% ====================== Part 3 ===========================================
% Regularization with the cost function (done above) and gradients, AFTER
% gradients computing

Theta1NoBias = Theta1(:, 2:end);    % Get Theta1 without the biases in 1st column
Theta2NoBias = Theta2(:, 2:end);    % Get Theta2 without the biases in 1st column

Theta1ZeroBias = [ zeros(size(Theta1, 1), 1) Theta1NoBias ];
Theta2ZeroBias = [ zeros(size(Theta2, 1), 1) Theta2NoBias ];
Theta1_grad = (1 / m) * delta1 + (lambda / m) * Theta1ZeroBias;
Theta2_grad = (1 / m) * delta2 + (lambda / m) * Theta2ZeroBias;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
