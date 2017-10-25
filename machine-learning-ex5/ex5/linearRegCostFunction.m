function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
u = size(theta, 1);      

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

prediction = X * theta;                 % Calculate the hypothesis
delta = prediction - y;                 % Calculate delta prediction vs. actuals
sqrErrors = (delta).^2;                 % Compute the squarred errors
J = 1/(2*m) * sum(sqrErrors);           % Compute the cost function without regularization

reg_penalty = lambda/(2 * m) * sum(theta(2:u).^2); % calculate the regularization penalty
J = J + reg_penalty;                    % Add to the cost function

theta1 = [0 ; theta(2:end, :)];         % make theta1 a 2-dim vector with 0 at first position
grad = (1/m)*(X'*delta) + (lambda/m)*theta1;    % calculate gradients in a single line (with theta1(1) = 0)

% =========================================================================

grad = grad(:);

end
