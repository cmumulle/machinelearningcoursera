function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    m = length(y);                      % number of training examples
    delta = zeros(2,1);                 % initialize delta
    
    prediction = X * theta;             % hypothesis for each training example
    difference = prediction - y;        % difference of hypotesis and actuals
    
    delta = 1/m * X' * difference;      % calculate delta to adjust theta
    
    theta = theta - alpha * delta;      % adjust theta


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

% disp(J_history);

end
