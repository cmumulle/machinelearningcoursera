function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Cost function calculation

h = sigmoid(X * theta);                                 % calculate the hypotheses 

u = size(theta, 1);                                     % store the length of theta

J = 1/m * (-y' * log(h) - (1 - y)' * log (1 - h)) ...
  + lambda / (2 * m) * sum(theta(2:u) .^2);             % calculate the cost function, don't regularize theta(0)

% Gradients calculation as arguments for fminunc

grad = 1/m * X' * (sigmoid(X * theta) - y) ...
  + lambda / m * theta;                                 % gradient calculation without alpha
temp = 1/m * X' * (sigmoid(X * theta) - y);             % store grad matrix in temp matrix
grad(1) = temp(1);                                      % overwrite 1st position theta(0), not regularized

% =============================================================

end
